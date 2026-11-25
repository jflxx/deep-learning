"""scripts/E3_analisis.py

Genera todas las tablas y figuras del E3 a partir de las carpetas de corridas
(results/E3_corridas/<id>/) y de la partición geográfica. No necesita GPU.

Uso:
    python scripts/E3_analisis.py [--corridas results/E3_corridas] [--rgb-dir data/EuroSAT_RGB]

Salidas (macro-F1 y diferencias en puntos porcentuales):
    results/C2_semillas.csv, C3_mcnemar_exacto.csv, C4_fase_c.csv, C4_veredicto_a4.json,
    C4b_interna_vs_geo.csv, C5_f1_por_clase_geo.csv, C6_f1_por_region.csv, C7_ablaciones.csv
    figures/C4_interno_vs_geo.png, C5_f1_por_clase_geo.png, C5_confusion_geo.png,
    C6_mapa_errores.png, C8_ejemplos_fallo_geo.png (solo si existe --rgb-dir)
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from eurosat_lora.analisis import (
    SEMILLAS, cargar_corridas, comparacion_interna_vs_geo, f1_por_clase_geo, f1_por_region,
    tabla_ablaciones, tabla_fase_c, tabla_mcnemar, tabla_semillas, veredicto_a4,
)

CLASES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
          "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
# Paleta categórica validada (slots 1 y 2) y tintas de texto
COLOR = {"base": "#257de0", "lora": "#f47230"}
NOMBRE = {"base": "Linear probing (congelado)", "lora": "LoRA r=8 (Q, V; bloques 11–12)"}
TXT, TXT2, GRID = "#191919", "#666666", "#e6e6e6"
AZULES = ["#ffffff", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": TXT2, "axes.labelcolor": TXT, "xtick.color": TXT2,
    "ytick.color": TXT2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def fig_interno_vs_geo(c2: pd.DataFrame, ruta: str) -> None:
    grupos = [("interna", "test", "Partición interna\n(test aleatorio)"),
              ("geo", "test_dentro", "Geográfica\ntest dentro"),
              ("geo", "test_fuera", "Geográfica\ntest fuera (regiones nuevas)")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for gi, (particion, conjunto, _) in enumerate(grupos):
        sub = c2[(c2.particion == particion) & (c2.conjunto == conjunto)]
        semillas = sub[~sub.semilla.isin(["media", "desv"])]
        media = sub[sub.semilla == "media"].iloc[0]
        for k, cond in enumerate(("base", "lora")):
            x = gi + (k - 0.5) * 0.3
            ax.scatter([x] * len(semillas), semillas[f"f1_{cond}"], s=36, color=COLOR[cond],
                       edgecolor="white", linewidth=1.5, zorder=3, alpha=0.85)
            ax.hlines(media[f"f1_{cond}"], x - 0.11, x + 0.11, color=COLOR[cond], linewidth=2.5, zorder=4)
        ax.annotate(f"Δ = {media['delta']:+.2f} pts", (gi, max(media.f1_base, media.f1_lora) + 0.9),
                    ha="center", color=TXT, fontsize=9)
    ax.set_ylim(top=ax.get_ylim()[1] + 1.2)
    ax.set_xticks(range(3), [g[2] for g in grupos])
    ax.set_ylabel("macro-F1 (%)")
    ax.set_title("Macro-F1 por conjunto de prueba (3 semillas; línea = media)", color=TXT, loc="left")
    for cond in ("base", "lora"):
        ax.scatter([], [], color=COLOR[cond], s=36, label=NOMBRE[cond])
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.savefig(ruta)
    plt.close(fig)


def fig_f1_por_clase(c5: pd.DataFrame, ruta: str) -> None:
    c5 = c5.sort_values("base_fuera")
    y = np.arange(len(c5))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, conj, titulo in ((axes[0], "dentro", "Test dentro (regiones de entrenamiento)"),
                             (axes[1], "fuera", "Test fuera (regiones nuevas)")):
        ax.hlines(y, c5[f"base_{conj}"], c5[f"lora_{conj}"], color=GRID, linewidth=2, zorder=2)
        for cond in ("base", "lora"):
            ax.scatter(c5[f"{cond}_{conj}"], y, color=COLOR[cond], s=40, zorder=3,
                       edgecolor="white", linewidth=1.5, label=NOMBRE[cond])
        ax.set_title(titulo, color=TXT, loc="left", fontsize=10)
        ax.set_xlabel("F1 por clase (%), media de 3 semillas")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, c5["clase"])
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def _matriz(salida: str, cond: str) -> np.ndarray:
    total = np.zeros((10, 10), dtype=int)
    for s in SEMILLAS:
        d = pd.read_csv(os.path.join(salida, f"geo_{cond}_r8_b10-11_q+v_s{s}", "predicciones_test_fuera.csv"))
        total += confusion_matrix(d["real"], d["predicho"], labels=range(10))
    return total


def fig_confusion(salida: str, ruta: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azules", AZULES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, cond in zip(axes, ("base", "lora")):
        cm = _matriz(salida, cond)
        fila = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(fila, cmap=cmap, vmin=0, vmax=100)
        ax.set_title(f"{NOMBRE[cond]} · test fuera (3 semillas sumadas)", color=TXT, loc="left", fontsize=10)
        ax.set_xticks(range(10), CLASES, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(10), CLASES, fontsize=8)
        ax.set_xlabel("predicho"); ax.set_ylabel("real"); ax.grid(False)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                            color="white" if fila[i, j] > 55 else TXT)
    fig.text(0.01, -0.02, "Color = % de la fila (recall por clase). Número = imágenes.", color=TXT2, fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def _desenlace(salida: str, semilla: int = 42) -> pd.DataFrame:
    b = pd.read_csv(os.path.join(salida, f"geo_base_r8_b10-11_q+v_s{semilla}", "predicciones_test_fuera.csv"))
    l = pd.read_csv(os.path.join(salida, f"geo_lora_r8_b10-11_q+v_s{semilla}", "predicciones_test_fuera.csv"))
    ok_b, ok_l = b.predicho == b.real, l.predicho == l.real
    out = b[["archivo", "real"]].copy()
    out["pred_base"], out["pred_lora"] = b.predicho, l.predicho
    out["desenlace"] = np.select([ok_b & ok_l, ~ok_b & ok_l, ok_b & ~ok_l],
                                 ["ambos aciertan", "solo LoRA acierta", "solo base acierta"], "ambos fallan")
    return out


def fig_mapa_errores(salida: str, coords: pd.DataFrame, split: dict, ruta: str) -> None:
    d = _desenlace(salida).merge(coords[["archivo_rgb", "lon", "lat"]], left_on="archivo", right_on="archivo_rgb")
    fuera = set(split["test_fuera"])
    resto = coords[~coords.archivo_rgb.isin(fuera)]
    estilos = [("ambos aciertan", "#c9c9c9", 6), ("solo base acierta", COLOR["base"], 16),
               ("solo LoRA acierta", COLOR["lora"], 16), ("ambos fallan", TXT, 16)]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(resto.lon, resto.lat, s=2, color="#ececec", label="train / val / test dentro", zorder=1)
    for nombre, color, s in estilos:
        sub = d[d.desenlace == nombre]
        ax.scatter(sub.lon, sub.lat, s=s, color=color, zorder=3 if nombre != "ambos aciertan" else 2,
                   edgecolor="white" if s > 6 else "none", linewidth=0.5, label=f"{nombre} ({len(sub)})")
    ax.set_xlabel("longitud"); ax.set_ylabel("latitud")
    ax.set_title("Test fuera (regiones nuevas), semilla 42: ¿quién acierta?", color=TXT, loc="left")
    ax.legend(loc="upper left", markerscale=1.5, fontsize=8)
    fig.savefig(ruta)
    plt.close(fig)


def fig_ejemplos(salida: str, rgb_dir: str, ruta: str, por_fila: int = 6) -> None:
    from PIL import Image
    d = _desenlace(salida)
    vegetacion = {CLASES.index(c) for c in ("AnnualCrop", "HerbaceousVegetation", "Pasture", "PermanentCrop")}
    filas = []
    for nombre in ("solo LoRA acierta", "solo base acierta", "ambos fallan"):
        sub = d[d.desenlace == nombre]
        veg = sub[sub.real.isin(vegetacion)]
        elegidos = pd.concat([veg.head(por_fila // 2), sub[~sub.index.isin(veg.index)]]).head(por_fila)
        filas.append((nombre, elegidos))
    fig, axes = plt.subplots(3, por_fila, figsize=(2.1 * por_fila, 7.6))
    for r, (nombre, sub) in enumerate(filas):
        for c in range(por_fila):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(sub):
                continue
            f = sub.iloc[c]
            ax.imshow(Image.open(os.path.join(rgb_dir, f.archivo)))
            ax.set_title(f"real: {CLASES[f.real]}\nbase: {CLASES[f.pred_base]}\nLoRA: {CLASES[f.pred_lora]}",
                         fontsize=7, color=TXT)
        axes[r, 0].text(-0.15, 0.5, nombre, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=10, color=TXT)
    fig.suptitle("Ejemplos de test fuera (semilla 42), con prioridad a clases de vegetación", color=TXT, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corridas", default="results/E3_corridas")
    ap.add_argument("--split-geo", default="splits/split_geo.json")
    ap.add_argument("--coords", default="splits/eurosat_coords.csv")
    ap.add_argument("--rgb-dir", default="data/EuroSAT_RGB/EuroSAT_RGB")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

    df = cargar_corridas(args.corridas)
    n = df["id"].nunique()
    assert n == 20, f"Se esperaban 20 corridas y hay {n}"

    c2 = tabla_semillas(df); c2.to_csv("results/C2_semillas.csv", index=False, float_format="%.4f")
    c3 = tabla_mcnemar(args.corridas); c3.to_csv("results/C3_mcnemar_exacto.csv", index=False)
    c4 = tabla_fase_c(df); c4.to_csv("results/C4_fase_c.csv", index=False, float_format="%.4f")
    ver = veredicto_a4(c4)
    with open("results/C4_veredicto_a4.json", "w", encoding="utf-8") as f:
        json.dump(ver, f, indent=2, ensure_ascii=False)
    comparacion_interna_vs_geo(df).to_csv("results/C4b_interna_vs_geo.csv", index=False, float_format="%.4f")
    c5 = f1_por_clase_geo(args.corridas, CLASES); c5.to_csv("results/C5_f1_por_clase_geo.csv", index=False, float_format="%.4f")
    c7 = tabla_ablaciones(df); c7.to_csv("results/C7_ablaciones.csv", index=False, float_format="%.4f")

    fig_interno_vs_geo(c2, "figures/C4_interno_vs_geo.png")
    fig_f1_por_clase(c5, "figures/C5_f1_por_clase_geo.png")
    fig_confusion(args.corridas, "figures/C5_confusion_geo.png")

    if os.path.exists(args.split_geo) and os.path.exists(args.coords):
        split = json.load(open(args.split_geo, encoding="utf-8"))
        coords = pd.read_csv(args.coords)
        f1_por_region(args.corridas, split["region"]).to_csv("results/C6_f1_por_region.csv", index=False, float_format="%.4f")
        fig_mapa_errores(args.corridas, coords, split, "figures/C6_mapa_errores.png")
    else:
        print("Aviso: falta la partición geográfica o las coordenadas; se omiten C6.")
    if os.path.isdir(args.rgb_dir):
        fig_ejemplos(args.corridas, args.rgb_dir, "figures/C8_ejemplos_fallo_geo.png")
    else:
        print(f"Aviso: no existe {args.rgb_dir}; se omite C8.")

    pd.set_option("display.width", 200)
    print(c2.round(2).to_string(index=False))
    print("\nVeredicto A4:", json.dumps(ver, indent=2, ensure_ascii=False))
    print("\n", c7.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

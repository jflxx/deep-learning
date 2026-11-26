"""informe/figuras_paper.py — Figuras en inglés para el informe final (formato WAIMLAp).

Uso (desde la raíz del repo):  python informe/figuras_paper.py
Lee results/E3_corridas/, splits/ y data/EuroSAT_RGB; escribe informe/figuras/*.png.
Mismos datos que scripts/E3_analisis.py; solo cambian los textos.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from eurosat_lora.analisis import SEMILLAS, cargar_corridas, f1_por_clase_geo, tabla_semillas
from eurosat_lora.datos import encontrar_raiz_clases

OUT = "informe/figuras"
CORR = "results/E3_corridas"
CLASES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
          "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
COLOR = {"base": "#257de0", "lora": "#f47230"}
NOMBRE = {"base": "Linear probing (frozen)", "lora": "LoRA r=8 (Q, V; blocks 11–12)"}
TXT, TXT2, GRID = "#191919", "#666666", "#e6e6e6"
AZULES = ["#ffffff", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
    "axes.edgecolor": TXT2, "axes.labelcolor": TXT, "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def mapa_particion(coords, split):
    conj = {}
    for k in ("train", "test_dentro"):
        for a in split[k]:
            conj[a] = "train"
    for a in split["val"]:
        conj[a] = "val"
    for a in split["test_fuera"]:
        conj[a] = "fuera"
    coords = coords.assign(c=coords.archivo_rgb.map(conj), region=coords.archivo_rgb.map(split["region"]))
    capas = [("train", "#b8b8b8", "training regions (train + in-region test)"),
             ("val", "#257de0", "validation regions"),
             ("fuera", "#f47230", "held-out regions (out-of-region test)")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c, color, nombre in capas:
        s = coords[coords.c == c]
        ax.scatter(s.lon, s.lat, s=4, c=color, edgecolors="none", label=f"{nombre}: {len(s):,} images")
    for r, s in coords.groupby("region"):
        ax.annotate(str(r), (s.lon.mean(), s.lat.mean()), xytext=(0, 8), textcoords="offset points",
                    fontsize=7, ha="center", color=TXT,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=TXT2, lw=0.5, alpha=0.9))
    ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°)")
    ax.legend(loc="upper left", fontsize=8, markerscale=2.5)
    fig.savefig(f"{OUT}/fig_particion_geo.png"); plt.close(fig)


def puntos_por_conjunto(c2):
    grupos = [("interna", "test", "Internal split\n(random test)"),
              ("geo", "test_dentro", "Geographic split\nin-region test"),
              ("geo", "test_fuera", "Geographic split\nout-of-region test")]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for gi, (p, conj, _) in enumerate(grupos):
        sub = c2[(c2.particion == p) & (c2.conjunto == conj)]
        sem = sub[~sub.semilla.isin(["media", "desv"])]
        med = sub[sub.semilla == "media"].iloc[0]
        for k, cond in enumerate(("base", "lora")):
            x = gi + (k - 0.5) * 0.3
            ax.scatter([x] * len(sem), sem[f"f1_{cond}"], s=30, color=COLOR[cond], edgecolor="white", lw=1.2, zorder=3)
            ax.hlines(med[f"f1_{cond}"], x - 0.11, x + 0.11, color=COLOR[cond], lw=2.5, zorder=4)
        ax.annotate(f"Δ = {med['delta']:+.2f} pp", (gi, max(med.f1_base, med.f1_lora) + 0.8), ha="center", fontsize=9, color=TXT)
    ax.set_ylim(top=ax.get_ylim()[1] + 1.0)
    ax.set_xticks(range(3), [g[2] for g in grupos]); ax.set_ylabel("macro-F1 (%)")
    for cond in ("base", "lora"):
        ax.scatter([], [], color=COLOR[cond], s=30, label=NOMBRE[cond])
    ax.legend(loc="lower left", fontsize=8); ax.grid(axis="x", visible=False)
    fig.savefig(f"{OUT}/fig_macro_f1_conjuntos.png"); plt.close(fig)


def f1_clase(c5):
    c5 = c5.sort_values("base_fuera"); y = np.arange(len(c5))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    for ax, conj, t in ((axes[0], "dentro", "In-region test"), (axes[1], "fuera", "Out-of-region test")):
        ax.hlines(y, c5[f"base_{conj}"], c5[f"lora_{conj}"], color=GRID, lw=2, zorder=2)
        for cond in ("base", "lora"):
            ax.scatter(c5[f"{cond}_{conj}"], y, color=COLOR[cond], s=30, zorder=3, edgecolor="white", lw=1.2, label=NOMBRE[cond])
        ax.set_title(t, loc="left", fontsize=10, color=TXT); ax.set_xlabel("per-class F1 (%), mean of 3 seeds")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, c5.clase, fontsize=8)
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_f1_por_clase.png"); plt.close(fig)


def confusion():
    cmap = LinearSegmentedColormap.from_list("azules", AZULES)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    corto = ["AnnCrop", "Forest", "HerbVeg", "Highway", "Industr.", "Pasture", "PermCrop", "Resid.", "River", "SeaLake"]
    for ax, cond in zip(axes, ("base", "lora")):
        cm = np.zeros((10, 10), int)
        for s in SEMILLAS:
            d = pd.read_csv(f"{CORR}/geo_{cond}_r8_b10-11_q+v_s{s}/predicciones_test_fuera.csv")
            cm += confusion_matrix(d.real, d.predicho, labels=range(10))
        fila = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(fila, cmap=cmap, vmin=0, vmax=100)
        ax.set_title(NOMBRE[cond], fontsize=9, loc="left", color=TXT)
        ax.set_xticks(range(10), corto, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(10), corto, fontsize=6.5); ax.grid(False)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=4.8, color="white" if fila[i, j] > 55 else TXT)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_confusion_fuera.png"); plt.close(fig)


def ejemplos(raiz, por_fila=6):
    from PIL import Image
    b = pd.read_csv(f"{CORR}/geo_base_r8_b10-11_q+v_s42/predicciones_test_fuera.csv")
    l = pd.read_csv(f"{CORR}/geo_lora_r8_b10-11_q+v_s42/predicciones_test_fuera.csv")
    okb, okl = b.predicho == b.real, l.predicho == l.real
    d = b[["archivo", "real"]].assign(pb=b.predicho, pl=l.predicho,
        des=np.select([~okb & okl, okb & ~okl], ["Only LoRA correct", "Only frozen correct"], "Both wrong"))
    d = d[~(okb & okl)]
    veg = {CLASES.index(c) for c in ("AnnualCrop", "HerbaceousVegetation", "Pasture", "PermanentCrop")}
    fig, axes = plt.subplots(3, por_fila, figsize=(7.2, 4.6))
    for r, nombre in enumerate(("Only LoRA correct", "Only frozen correct", "Both wrong")):
        sub = d[d.des == nombre]
        sel = pd.concat([sub[sub.real.isin(veg)].head(por_fila // 2), sub[~sub.real.isin(veg)]]).head(por_fila)
        for c in range(por_fila):
            ax = axes[r, c]; ax.axis("off")
            if c < len(sel):
                f = sel.iloc[c]
                ax.imshow(Image.open(os.path.join(raiz, f.archivo)))
                ax.set_title(f"true: {CLASES[f.real]}\nfrozen: {CLASES[f.pb]}\nLoRA: {CLASES[f.pl]}", fontsize=5.5, color=TXT)
        axes[r, 0].text(-0.12, 0.5, nombre, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ejemplos.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = cargar_corridas(CORR)
    split = json.load(open("splits/split_geo.json", encoding="utf-8"))
    mapa_particion(pd.read_csv("splits/eurosat_coords.csv"), split)
    puntos_por_conjunto(tabla_semillas(df))
    f1_clase(f1_por_clase_geo(CORR, CLASES))
    confusion()
    ejemplos(encontrar_raiz_clases("data"))
    print("Figuras en", OUT, sorted(os.listdir(OUT)))

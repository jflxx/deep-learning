"""analisis.py — Tablas de resultados del E3 y veredicto de A4.

Convenciones:
- Todas las tablas de salida expresan macro-F1 y diferencias en PUNTOS porcentuales
  (0–100). Los umbrales de la regla de A4 (DESIGN_LOG, sección 2) están en puntos.
- La comparación principal (A2 en la partición interna, A4 en la geográfica) usa solo
  las corridas de esos experimentos (LoRA r=8, Q+V, bloques 11–12). Las ablaciones
  (A3, A5, A6, A3g, A5g, A6g) se analizan aparte en `tabla_ablaciones` y nunca se
  mezclan con la comparación principal.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from eurosat_lora.metricas import mcnemar_exacto

PTS = 100.0
EXP_PRINCIPALES = {"interna": "A2", "geo": "A4"}
CONJUNTOS_TEST = {"interna": ["test"], "geo": ["test_dentro", "test_fuera"]}
SEMILLAS = [42, 0, 1]


def cargar_corridas(salida: str) -> pd.DataFrame:
    """Una fila por corrida y conjunto evaluado (val y tests). macro_f1 en [0, 1]."""
    filas = []
    if not os.path.exists(salida):
        return pd.DataFrame()
    for nombre in sorted(os.listdir(salida)):
        ruta_m = os.path.join(salida, nombre, "metricas.json")
        if not os.path.exists(ruta_m):
            continue
        with open(ruta_m, encoding="utf-8") as f:
            d = json.load(f)
        for conjunto, m_dict in d.get("metricas", {}).items():
            filas.append({
                "id": d["id"],
                "experimento": d["experimento"],
                "particion": d["particion"],
                "condicion": d["condicion"],
                "semilla": d["semilla"],
                "rango": d["rango"],
                "bloques": "-".join(str(b + 1) for b in d["bloques"]),  # 1-indexados, como en el cuaderno
                "proyecciones": "+".join(p.upper() for p in d["proyecciones"]),
                "conjunto": conjunto,
                "accuracy": m_dict.get("accuracy"),
                "macro_f1": m_dict.get("macro_f1"),
            })
    return pd.DataFrame(filas)


def _principales(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for particion, exp in EXP_PRINCIPALES.items():
        mask |= (df["particion"] == particion) & (df["experimento"] == exp)
    return df[mask]


def _f1(df: pd.DataFrame, particion: str, condicion: str, semilla: int, conjunto: str) -> float:
    sub = df[(df["particion"] == particion) & (df["condicion"] == condicion)
             & (df["semilla"] == semilla) & (df["conjunto"] == conjunto)]
    if len(sub) != 1:
        raise ValueError(f"Se esperaba 1 corrida para {particion}/{condicion}/s{semilla}/{conjunto}, hay {len(sub)}")
    return float(sub["macro_f1"].iloc[0]) * PTS


def tabla_semillas(df: pd.DataFrame) -> pd.DataFrame:
    """C2: macro-F1 de base y LoRA r8 por partición, conjunto y semilla, Δ = LoRA − base.

    Agrega filas 'media' y 'desv' (desviación estándar muestral, ddof=1) por partición y conjunto.
    """
    p = _principales(df)
    filas = []
    for particion, conjuntos in CONJUNTOS_TEST.items():
        for conjunto in conjuntos:
            bloque = []
            for s in SEMILLAS:
                base = _f1(p, particion, "base", s, conjunto)
                lora = _f1(p, particion, "lora", s, conjunto)
                bloque.append({"particion": particion, "conjunto": conjunto, "semilla": str(s),
                               "f1_base": base, "f1_lora": lora, "delta": lora - base})
            t = pd.DataFrame(bloque)
            for nombre, fn in (("media", lambda c: c.mean()), ("desv", lambda c: c.std(ddof=1))):
                bloque.append({"particion": particion, "conjunto": conjunto, "semilla": nombre,
                               **{c: float(fn(t[c])) for c in ("f1_base", "f1_lora", "delta")}})
            filas += bloque
    return pd.DataFrame(filas)


def tabla_mcnemar(salida: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """C3: McNemar exacto base vs LoRA r8 por semilla, en cada conjunto de test.

    b = aciertos solo de la base; c = aciertos solo de LoRA.
    """
    filas = []
    for particion, conjuntos in CONJUNTOS_TEST.items():
        for semilla in SEMILLAS:
            for conjunto in conjuntos:
                id_base = f"{particion}_base_r8_b10-11_q+v_s{semilla}"
                id_lora = f"{particion}_lora_r8_b10-11_q+v_s{semilla}"
                df_b = pd.read_csv(os.path.join(salida, id_base, f"predicciones_{conjunto}.csv"))
                df_l = pd.read_csv(os.path.join(salida, id_lora, f"predicciones_{conjunto}.csv"))
                if not df_b["archivo"].equals(df_l["archivo"]):
                    raise ValueError(f"Predicciones no pareadas en {particion}/s{semilla}/{conjunto}")
                res = mcnemar_exacto(df_b["real"].values, df_b["predicho"].values, df_l["predicho"].values)
                filas.append({"particion": particion, "semilla": semilla, "conjunto": conjunto,
                              "solo_base_acierta": res["b"], "solo_lora_acierta": res["c"],
                              "n_discordantes": res["n_discordantes"], "p_valor": res["p_valor"]})
    return pd.DataFrame(filas)


def tabla_fase_c(df: pd.DataFrame) -> pd.DataFrame:
    """C4: por semilla y condición, macro-F1 (pts) en test dentro y fuera y caída = dentro − fuera.

    Solo usa las corridas de A4 (LoRA r8, Q+V, bloques 11–12).
    """
    p = _principales(df)
    filas = []
    for s in SEMILLAS:
        for condicion in ("base", "lora"):
            dentro = _f1(p, "geo", condicion, s, "test_dentro")
            fuera = _f1(p, "geo", condicion, s, "test_fuera")
            filas.append({"semilla": s, "condicion": condicion, "f1_dentro": dentro,
                          "f1_fuera": fuera, "caida": dentro - fuera})
    return pd.DataFrame(filas)


def veredicto_a4(tabla_fc: pd.DataFrame) -> dict:
    """Regla de decisión de A4 (DESIGN_LOG, fijada el 24/11/2026). Valores en puntos."""
    base = tabla_fc[tabla_fc["condicion"] == "base"].set_index("semilla")
    lora = tabla_fc[tabla_fc["condicion"] == "lora"].set_index("semilla")
    semillas = sorted(set(base.index) & set(lora.index))
    if not semillas:
        return {"veredicto": "intermedio", "error": "datos insuficientes"}

    ventaja_fuera = np.array([lora.loc[s, "f1_fuera"] - base.loc[s, "f1_fuera"] for s in semillas])
    ventaja_dentro = np.array([lora.loc[s, "f1_dentro"] - base.loc[s, "f1_dentro"] for s in semillas])
    caida_base = float(base.loc[semillas, "caida"].mean())
    caida_lora = float(lora.loc[semillas, "caida"].mean())
    v_fuera = float(ventaja_fuera.mean())

    if 3.0 <= caida_base <= 10.0 and 3.0 <= caida_lora <= 10.0 and v_fuera <= 0.5:
        veredicto = "se_cumple"
    elif caida_base < 1.0 or caida_lora < 1.0 or v_fuera >= 1.3:
        veredicto = "se_rechaza"
    else:
        veredicto = "intermedio"

    signo_consistente = len(set(np.sign(ventaja_fuera))) == 1
    return {
        "veredicto": veredicto,
        "caida_media_base": caida_base,
        "caida_media_lora": caida_lora,
        "caida_desv_base": float(base.loc[semillas, "caida"].std(ddof=1)) if len(semillas) > 1 else 0.0,
        "caida_desv_lora": float(lora.loc[semillas, "caida"].std(ddof=1)) if len(semillas) > 1 else 0.0,
        "ventaja_media_fuera": v_fuera,
        "ventaja_desv_fuera": float(ventaja_fuera.std(ddof=1)) if len(semillas) > 1 else 0.0,
        "ventaja_media_dentro": float(ventaja_dentro.mean()),
        "ventaja_fuera_por_semilla": dict(zip([int(s) for s in semillas], ventaja_fuera.tolist())),
        "signo_consistente": signo_consistente,
        "lectura_signo": ("LoRA mejor fuera" if (ventaja_fuera > 0).all() else
                          "LoRA peor fuera" if (ventaja_fuera < 0).all() else
                          "sin diferencia consistente"),
    }


def _leer_pred(salida: str, id_corrida: str, conjunto: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(salida, id_corrida, f"predicciones_{conjunto}.csv"))


def f1_por_clase_geo(salida: str, clases: list[str]) -> pd.DataFrame:
    """C5: F1 por clase (pts, media de las 3 semillas) de base y LoRA r8 en test dentro y fuera."""
    filas = []
    for i, clase in enumerate(clases):
        fila = {"clase": clase}
        for condicion in ("base", "lora"):
            for conjunto in ("test_dentro", "test_fuera"):
                vals = []
                for s in SEMILLAS:
                    d = _leer_pred(salida, f"geo_{condicion}_r8_b10-11_q+v_s{s}", conjunto)
                    vals.append(f1_score(d["real"], d["predicho"], labels=[i], average=None)[0] * PTS)
                fila[f"{condicion}_{conjunto.replace('test_', '')}"] = float(np.mean(vals))
        fila["caida_base"] = fila["base_dentro"] - fila["base_fuera"]
        fila["caida_lora"] = fila["lora_dentro"] - fila["lora_fuera"]
        fila["ventaja_lora_fuera"] = fila["lora_fuera"] - fila["base_fuera"]
        filas.append(fila)
    return pd.DataFrame(filas)


def f1_por_region(salida: str, region_por_archivo: dict) -> pd.DataFrame:
    """C6: macro-F1 (pts, media de 3 semillas) por región de test fuera, base vs LoRA r8."""
    filas = []
    ref = _leer_pred(salida, "geo_base_r8_b10-11_q+v_s42", "test_fuera")
    regiones = sorted({region_por_archivo[a] for a in ref["archivo"]})
    for r in regiones:
        fila = {"region": r}
        for condicion in ("base", "lora"):
            vals = []
            for s in SEMILLAS:
                d = _leer_pred(salida, f"geo_{condicion}_r8_b10-11_q+v_s{s}", "test_fuera")
                d = d[d["archivo"].map(region_por_archivo) == r]
                vals.append(f1_score(d["real"], d["predicho"], average="macro") * PTS)
                fila["n_imagenes"] = len(d)
                fila["n_clases"] = d["real"].nunique()
            fila[f"f1_{condicion}"] = float(np.mean(vals))
        fila["ventaja_lora"] = fila["f1_lora"] - fila["f1_base"]
        filas.append(fila)
    return pd.DataFrame(filas)


ABLACIONES = {"A3": "interna", "A5": "interna", "A6": "interna",
              "A3g": "geo", "A5g": "geo", "A6g": "geo"}


def tabla_ablaciones(df: pd.DataFrame) -> pd.DataFrame:
    """C7: cada ablación (semilla 42) frente a LoRA r8 Q+V bloques 11–12 (semilla 42) en la misma partición.

    En la partición geográfica incluye la caída (dentro − fuera) de la variante y de r8.
    """
    ref = df[(df["condicion"] == "lora") & (df["semilla"] == 42)
             & (df["experimento"].isin(EXP_PRINCIPALES.values()))]
    abl = df[df["experimento"].isin(ABLACIONES)]
    filas = []
    for id_corrida, g in abl.groupby("id", sort=True):
        exp = g["experimento"].iloc[0]
        particion = ABLACIONES[exp]
        r = ref[ref["particion"] == particion]
        fila = {"experimento": exp, "particion": particion, "id": id_corrida,
                "variante": f"r={g['rango'].iloc[0]}, bloques {g['bloques'].iloc[0]}, {g['proyecciones'].iloc[0]}"}
        for conjunto in CONJUNTOS_TEST[particion]:
            f_var = float(g.loc[g["conjunto"] == conjunto, "macro_f1"].iloc[0]) * PTS
            f_ref = float(r.loc[r["conjunto"] == conjunto, "macro_f1"].iloc[0]) * PTS
            sufijo = conjunto.replace("test_", "") if particion == "geo" else "test"
            fila[f"f1_{sufijo}"] = f_var
            fila[f"f1_r8_{sufijo}"] = f_ref
            fila[f"delta_{sufijo}"] = f_var - f_ref
        if particion == "geo":
            fila["caida"] = fila["f1_dentro"] - fila["f1_fuera"]
            fila["caida_r8"] = fila["f1_r8_dentro"] - fila["f1_r8_fuera"]
            fila["delta_caida"] = fila["caida"] - fila["caida_r8"]
        filas.append(fila)
    orden = {e: i for i, e in enumerate(ABLACIONES)}
    return pd.DataFrame(filas).sort_values(["experimento", "id"], key=lambda c: c.map(orden) if c.name == "experimento" else c).reset_index(drop=True)


def comparacion_interna_vs_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Dato secundario (enfoque "a" del diseño): test interno de los modelos internos vs test fuera de los geográficos."""
    t = tabla_semillas(df)
    t = t[t["semilla"] == "media"]
    filas = []
    for condicion in ("base", "lora"):
        interno = float(t[(t["particion"] == "interna")][f"f1_{condicion}"].iloc[0])
        fuera = float(t[(t["particion"] == "geo") & (t["conjunto"] == "test_fuera")][f"f1_{condicion}"].iloc[0])
        filas.append({"condicion": condicion, "f1_test_interno": interno, "f1_test_fuera_geo": fuera,
                      "diferencia": interno - fuera})
    return pd.DataFrame(filas)

"""tests/test_analisis.py — Tests de veredicto A4 y norma de tokens."""

import numpy as np
import pandas as pd
import pytest

from eurosat_lora.analisis import veredicto_a4
from eurosat_lora.diagnosticos import norma_tokens


def _tabla(semillas, f1_base_dentro, f1_base_fuera, f1_lora_dentro, f1_lora_fuera):
    filas = []
    for s, bd, bf, ld, lf in zip(semillas, f1_base_dentro, f1_base_fuera, f1_lora_dentro, f1_lora_fuera):
        filas.append({"semilla": s, "condicion": "base", "f1_dentro": bd, "f1_fuera": bf, "caida": bd - bf})
        filas.append({"semilla": s, "condicion": "lora", "f1_dentro": ld, "f1_fuera": lf, "caida": ld - lf})
    return pd.DataFrame(filas)


def test_a4_se_cumple():
    t = _tabla([42, 0, 1], [80, 81, 79], [75, 75, 76], [81, 82, 80], [75.5, 75.3, 76.0])
    res = veredicto_a4(t)
    assert res["veredicto"] == "se_cumple"


def test_a4_rechazo_sin_caida():
    # caída media base 0.5
    t = _tabla([42, 0, 1], [80, 80, 80], [79.5, 79.7, 79.8], [81, 81, 81], [80, 80, 80])
    res = veredicto_a4(t)
    assert res["veredicto"] == "se_rechaza"


def test_a4_rechazo_ventaja():
    # ventaja LoRA fuera [1.5, 1.4, 1.3]
    t = _tabla([42, 0, 1], [80, 80, 80], [70, 70, 70], [82, 82, 82], [71.5, 71.4, 71.3])
    res = veredicto_a4(t)
    assert res["veredicto"] == "se_rechaza"


def test_a4_intermedio():
    # caídas [2, 2, 2] — entre 1 y 3, ventaja fuera pequeña
    t = _tabla([42, 0, 1], [80, 80, 80], [78, 78, 78], [82, 82, 82], [78.3, 78.3, 78.3])
    res = veredicto_a4(t)
    assert res["veredicto"] == "intermedio"


def test_a4_signo_inconsistente():
    # ventaja fuera [0.4, -0.3, 0.2]
    t = _tabla([42, 0, 1], [82, 82, 82], [76, 76, 76], [83, 81, 82.5], [76.4, 75.7, 76.2])
    res = veredicto_a4(t)
    assert res["signo_consistente"] is False


def _escribir_corrida(raiz, experimento, particion, condicion, semilla, f1s, rango=8, bloques=(10, 11), proy=("q", "v")):
    import json, os
    id_c = f"{particion}_{condicion}_r{rango}_b{'-'.join(map(str, bloques))}_{'+'.join(proy)}_s{semilla}"
    os.makedirs(raiz / id_c)
    m = {k: {"accuracy": v, "macro_f1": v} for k, v in f1s.items()}
    (raiz / id_c / "metricas.json").write_text(json.dumps({
        "id": id_c, "experimento": experimento, "particion": particion, "condicion": condicion,
        "semilla": semilla, "rango": rango, "bloques": list(bloques), "proyecciones": list(proy), "metricas": m}))


@pytest.fixture
def corridas_falsas(tmp_path):
    for s, (bd, bf, ld, lf) in {42: (0.96, 0.89, 0.97, 0.91), 0: (0.96, 0.88, 0.97, 0.92), 1: (0.97, 0.90, 0.97, 0.92)}.items():
        _escribir_corrida(tmp_path, "A4", "geo", "base", s, {"val": .9, "test_dentro": bd, "test_fuera": bf})
        _escribir_corrida(tmp_path, "A4", "geo", "lora", s, {"val": .9, "test_dentro": ld, "test_fuera": lf})
        _escribir_corrida(tmp_path, "A2", "interna", "base", s, {"val": .9, "test": 0.958})
        _escribir_corrida(tmp_path, "A2", "interna", "lora", s, {"val": .9, "test": 0.970})
    # Ablaciones con la misma semilla 42: NO deben entrar en C2/C4
    _escribir_corrida(tmp_path, "A6g", "geo", "lora", 42, {"val": .9, "test_dentro": 0.50, "test_fuera": 0.40}, proy=("q",))
    _escribir_corrida(tmp_path, "A3", "interna", "lora", 42, {"val": .9, "test": 0.50}, rango=4)
    return tmp_path


def test_fase_c_en_puntos_y_solo_a4(corridas_falsas):
    from eurosat_lora.analisis import cargar_corridas, tabla_fase_c
    t = tabla_fase_c(cargar_corridas(str(corridas_falsas)))
    fila = t[(t.semilla == 42) & (t.condicion == "lora")].iloc[0]
    assert fila.f1_dentro == pytest.approx(97.0)
    assert fila.f1_fuera == pytest.approx(91.0)
    assert fila.caida == pytest.approx(6.0)


def test_veredicto_desde_corridas(corridas_falsas):
    from eurosat_lora.analisis import cargar_corridas, tabla_fase_c
    res = veredicto_a4(tabla_fase_c(cargar_corridas(str(corridas_falsas))))
    # ventaja fuera = [2, 4, 2] pts → media 2.67 ≥ 1.3 → se rechaza; caídas en [3, 10]
    assert res["ventaja_media_fuera"] == pytest.approx(8 / 3)
    assert res["veredicto"] == "se_rechaza"
    assert res["lectura_signo"] == "LoRA mejor fuera"


def test_semillas_excluye_ablaciones(corridas_falsas):
    from eurosat_lora.analisis import cargar_corridas, tabla_semillas
    t = tabla_semillas(cargar_corridas(str(corridas_falsas)))
    media = t[(t.particion == "interna") & (t.semilla == "media")].iloc[0]
    assert media.f1_lora == pytest.approx(97.0)
    assert media.delta == pytest.approx(1.2)


def test_ablaciones_contra_r8(corridas_falsas):
    from eurosat_lora.analisis import cargar_corridas, tabla_ablaciones
    t = tabla_ablaciones(cargar_corridas(str(corridas_falsas)))
    a6g = t[t.experimento == "A6g"].iloc[0]
    assert a6g.delta_fuera == pytest.approx(40 - 91)
    assert a6g.caida == pytest.approx(10.0)
    assert a6g.caida_r8 == pytest.approx(6.0)
    a3 = t[t.experimento == "A3"].iloc[0]
    assert a3.delta_test == pytest.approx(50 - 97)


def test_norma_tokens_forma(modelo_chico):
    x = torch.randn(2, 3, 28, 28)
    normas = norma_tokens(modelo_chico, x)
    # 28x28 con patch 14 → (28/14)^2 = 4 patches + 1 CLS = 5 tokens
    assert normas.shape == (2, 5)


import torch

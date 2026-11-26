"""experimentos.py — Matriz de experimentos y corridas idempotentes."""

import csv
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Literal
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from eurosat_lora.config import Config
from eurosat_lora.metricas import metricas, mcnemar_exacto
from eurosat_lora.modelo import aplicar_lora, cargar_dinov2, contar_entrenables, fijar_semilla
from eurosat_lora.datos import indices_de, cargador
from eurosat_lora.entrenamiento import (
    extraer_features,
    entrenar_base,
    predecir_base,
    entrenar_lora,
    predecir_lora,
    cargador_desde_subset,
)


@dataclass(frozen=True)
class Corrida:
    id: str
    experimento: str  # A2, A4, A3, A5, A6, A3g, A5g, A6g
    particion: Literal["interna", "geo"]
    condicion: Literal["base", "lora"]
    semilla: int
    rango: int = 8
    bloques: tuple = (10, 11)
    proyecciones: tuple = ("q", "v")

    @staticmethod
    def crear(
        experimento: str,
        particion: str,
        condicion: str,
        semilla: int,
        rango: int = 8,
        bloques: tuple = (10, 11),
        proyecciones: tuple = ("q", "v"),
    ) -> "Corrida":
        proy_str = "+".join(proyecciones)
        bloques_str = "-".join(str(b) for b in bloques)
        id_corrida = (
            f"{particion}_{condicion}_r{rango}_b{bloques_str}_{proy_str}_s{semilla}"
        ).replace(" ", "")
        return Corrida(
            id=id_corrida,
            experimento=experimento,
            particion=particion,
            condicion=condicion,
            semilla=semilla,
            rango=rango,
            bloques=bloques,
            proyecciones=proyecciones,
        )


def _hacer_matriz() -> list[Corrida]:
    corridas = []
    # A2: interna, base + LoRA r8 Q+V b11-12 × 3 semillas
    for semilla in [42, 0, 1]:
        corridas.append(Corrida.crear("A2", "interna", "base", semilla))
        corridas.append(Corrida.crear("A2", "interna", "lora", semilla))

    # A4/Fase C: geográfica, base + LoRA r8 Q+V b11-12 × 3 semillas
    for semilla in [42, 0, 1]:
        corridas.append(Corrida.crear("A4", "geo", "base", semilla))
        corridas.append(Corrida.crear("A4", "geo", "lora", semilla))

    # A3: interna, LoRA r4 y r16, semilla 42
    corridas.append(Corrida.crear("A3", "interna", "lora", 42, rango=4))
    corridas.append(Corrida.crear("A3", "interna", "lora", 42, rango=16, bloques=(10, 11)))

    # A5: interna, LoRA r8 Q+V en bloques 8-9-10-11 (4 últimos), semilla 42
    corridas.append(Corrida.crear("A5", "interna", "lora", 42, bloques=(8, 9, 10, 11)))

    # A6: interna, LoRA r8 solo Q en b11-12, semilla 42
    corridas.append(Corrida.crear("A6", "interna", "lora", 42, proyecciones=("q",)))

    # A3g, A5g, A6g: geográfica
    corridas.append(Corrida.crear("A3g", "geo", "lora", 42, rango=4))
    corridas.append(Corrida.crear("A3g", "geo", "lora", 42, rango=16))
    corridas.append(Corrida.crear("A5g", "geo", "lora", 42, bloques=(8, 9, 10, 11)))
    corridas.append(Corrida.crear("A6g", "geo", "lora", 42, proyecciones=("q",)))

    return corridas


MATRIZ_E3: list[Corrida] = _hacer_matriz()

PREDICCION_REQUERIDA = {"A5", "A6", "A3g", "A5g", "A6g"}


def verificar_predicciones(ids: set, ruta_log: str = "DESIGN_LOG.md") -> None:
    """Verifica que cada ID tenga una fila | ID | en el cuaderno de decisiones."""
    if not os.path.exists(ruta_log):
        raise RuntimeError(f"No se encuentra el cuaderno en '{ruta_log}'")

    with open(ruta_log, "r", encoding="utf-8") as f:
        contenido = f.read()

    faltantes = []
    for id_pred in sorted(ids):
        if f"| {id_pred} |" not in contenido:
            faltantes.append(id_pred)

    if faltantes:
        raise RuntimeError(
            f"Predicciones faltantes en {ruta_log}: {faltantes}. "
            "El equipo debe registrar una predicción con '| ID |' antes de continuar."
        )


@dataclass
class Contexto:
    dataset: object
    clases: list
    particiones: dict
    cfg: Config
    device: str
    amp: bool
    cache_features: dict = field(default_factory=dict)


def _guardar_csv(ruta: str, filas: list[dict]) -> None:
    if not filas:
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def ejecutar_corrida(c: Corrida, ctx: Contexto, salida: str) -> dict:
    """Ejecuta (o carga si ya existe) una corrida. Idempotente."""
    carpeta = os.path.join(salida, c.id)
    ruta_metricas = os.path.join(carpeta, "metricas.json")

    # Si ya existe, cargar y devolver
    if os.path.exists(ruta_metricas):
        with open(ruta_metricas, "r", encoding="utf-8") as f:
            return json.load(f)

    # Limpiar carpeta parcial
    if os.path.exists(carpeta):
        shutil.rmtree(carpeta)
    os.makedirs(carpeta, exist_ok=True)

    t0 = time.time()
    # Como en el E2: fijar la semilla ANTES de crear LoRA, para que la
    # inicialización de A dependa solo de la semilla de la corrida.
    fijar_semilla(c.semilla)
    particion = ctx.particiones[c.particion]
    dataset = ctx.dataset
    cfg = ctx.cfg
    cfg_corrida = Config(
        modelo_id=cfg.modelo_id,
        img_size=cfg.img_size,
        batch=cfg.batch,
        epocas=cfg.epocas,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        lora_rango=c.rango,
        lora_alpha=float(c.rango),
        lora_dropout=0.0,
        lora_bloques=c.bloques,
        lora_proyecciones=c.proyecciones,
        max_por_clase=cfg.max_por_clase,
        num_workers=cfg.num_workers,
    )

    idx_train = indices_de(dataset, particion["train"])
    idx_val = indices_de(dataset, particion["val"])
    n_clases = len(ctx.clases)

    resultado_metricas = {}
    conjuntos_test = {}

    if c.condicion == "base":
        # Cache de features
        clave_cache_tr = f"{c.particion}_train"
        clave_cache_va = f"{c.particion}_val"
        if clave_cache_tr not in ctx.cache_features:
            m_base = cargar_dinov2(cfg.modelo_id, ctx.device)
            dl_tr = cargador(dataset, idx_train, batch=cfg.batch, num_workers=cfg.num_workers)
            dl_va = cargador(dataset, idx_val, batch=cfg.batch, num_workers=cfg.num_workers)
            ctx.cache_features[clave_cache_tr] = extraer_features(m_base, dl_tr, ctx.device, ctx.amp)
            ctx.cache_features[clave_cache_va] = extraer_features(m_base, dl_va, ctx.device, ctx.amp)
            del m_base

        cache_local = {
            "train": ctx.cache_features[clave_cache_tr],
            "val": ctx.cache_features[clave_cache_va],
        }
        cabeza, hist = entrenar_base(cache_local, n_clases, cfg_corrida, c.semilla, ctx.device)
        params_entrenables = sum(p.numel() for p in cabeza.parameters())
        hist.to_csv(os.path.join(carpeta, "historial.csv"), index=False)
        torch.save(cabeza.state_dict(), os.path.join(carpeta, "cabeza.pt"))

        X_va, y_va = ctx.cache_features[clave_cache_va]
        preds_va, probs_va = predecir_base(cabeza, X_va, ctx.device)
        resultado_metricas["val"] = metricas(y_va.numpy(), preds_va)

        # Test(s)
        for nombre_test in _nombres_test(c.particion):
            if nombre_test not in particion:
                continue
            idx_te = indices_de(dataset, particion[nombre_test])
            clave_te = f"{c.particion}_{nombre_test}"
            if clave_te not in ctx.cache_features:
                m_base2 = cargar_dinov2(cfg.modelo_id, ctx.device)
                dl_te = cargador(dataset, idx_te, batch=cfg.batch, num_workers=cfg.num_workers)
                ctx.cache_features[clave_te] = extraer_features(m_base2, dl_te, ctx.device, ctx.amp)
                del m_base2
            X_te, y_te = ctx.cache_features[clave_te]
            preds_te, probs_te = predecir_base(cabeza, X_te, ctx.device)
            resultado_metricas[nombre_test] = metricas(y_te.numpy(), preds_te)
            conjuntos_test[nombre_test] = (y_te.numpy(), preds_te, probs_te, particion[nombre_test])

    else:  # lora
        m = cargar_dinov2(cfg.modelo_id, ctx.device)
        aplicar_lora(m, bloques=c.bloques, proyecciones=c.proyecciones,
                     r=c.rango, alpha=float(c.rango), dropout=0.0)
        params_lora = contar_entrenables(m)

        dl_tr = cargador(dataset, idx_train, batch=cfg.batch, shuffle=True,
                         seed=c.semilla, num_workers=cfg.num_workers)
        dl_va = cargador(dataset, idx_val, batch=cfg.batch, num_workers=cfg.num_workers)
        cabeza, hist, pesos = entrenar_lora(
            m, dl_tr, dl_va, n_clases, cfg_corrida, c.semilla, ctx.device, ctx.amp
        )
        hist.to_csv(os.path.join(carpeta, "historial.csv"), index=False)
        torch.save(pesos, os.path.join(carpeta, "lora_y_cabeza.pt"))
        params_entrenables = params_lora + sum(p.numel() for p in cabeza.parameters())

        primer_bloque = min(c.bloques)
        preds_va, probs_va = predecir_lora(m, cabeza, dl_va, primer_bloque, ctx.device, ctx.amp)
        y_va = np.array([dataset.targets[i] for i in idx_val])
        resultado_metricas["val"] = metricas(y_va, preds_va)

        for nombre_test in _nombres_test(c.particion):
            if nombre_test not in particion:
                continue
            idx_te = indices_de(dataset, particion[nombre_test])
            dl_te = cargador(dataset, idx_te, batch=cfg.batch, num_workers=cfg.num_workers)
            preds_te, probs_te = predecir_lora(m, cabeza, dl_te, primer_bloque, ctx.device, ctx.amp)
            y_te = np.array([dataset.targets[i] for i in idx_te])
            resultado_metricas[nombre_test] = metricas(y_te, preds_te)
            conjuntos_test[nombre_test] = (y_te, preds_te, probs_te, particion[nombre_test])

    # Guardar predicciones
    for nombre_test, (y_te, preds_te, probs_te, archivos_te) in conjuntos_test.items():
        filas = [
            {"archivo": a, "real": int(y), "predicho": int(p), "prob": float(pr)}
            for a, y, p, pr in zip(archivos_te, y_te, preds_te, probs_te)
        ]
        _guardar_csv(os.path.join(carpeta, f"predicciones_{nombre_test}.csv"), filas)

    t1 = time.time()
    salida_dict = {
        "id": c.id,
        "experimento": c.experimento,
        "particion": c.particion,
        "condicion": c.condicion,
        "semilla": c.semilla,
        "rango": c.rango,
        "bloques": list(c.bloques),
        "proyecciones": list(c.proyecciones),
        "segundos": round(t1 - t0, 1),
        "parametros_entrenables": int(params_entrenables),
        "config": {
            "epocas": cfg.epocas, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
            "batch": cfg.batch, "img_size": cfg.img_size, "amp": bool(ctx.amp),
            "grad_scaler": bool(ctx.amp), "semilla_antes_de_lora": True,
        },
        "entorno": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "metricas": resultado_metricas,
    }

    # Escritura atómica
    ruta_tmp = ruta_metricas + ".tmp"
    with open(ruta_tmp, "w", encoding="utf-8") as f:
        json.dump(salida_dict, f, indent=2)
    os.replace(ruta_tmp, ruta_metricas)

    return salida_dict


def _nombres_test(particion: str) -> list[str]:
    if particion == "interna":
        return ["test"]
    else:
        return ["test_dentro", "test_fuera"]

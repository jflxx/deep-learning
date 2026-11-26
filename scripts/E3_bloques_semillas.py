"""scripts/E3_bloques_semillas.py  (Colab con GPU T4, ~45 min)

1) Replica la ablación A5/A5g (LoRA r=8, Q+V, bloques 9–12) con las semillas 0 y 1 en la
   partición interna y en la geográfica (4 corridas, código corregido). Predicción registrada
   antes de correr: fila A5s de DESIGN_LOG.md.
2) Re-mide la anomalía de latencia fp16 con batch 1 del E2 (results/C10_latencia_fp16_b1.csv).

Uso (desde la raíz del repo):
    python scripts/E3_bloques_semillas.py --salida /content/drive/MyDrive/E3_bloques
Al final imprime las tablas; copiar esa salida. Las corridas quedan en --salida (Drive) y
también se comprimen en <salida>.zip para descargarlas.
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch

from eurosat_lora.config import Config
from eurosat_lora.datos import archivos_de, construir_dataset, particion_estratificada, preparar_eurosat_rgb
from eurosat_lora.diagnosticos import remedir_fp16_b1
from eurosat_lora.experimentos import Contexto, Corrida, ejecutar_corrida, verificar_predicciones
from eurosat_lora.modelo import cargar_dinov2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="outputs/E3_bloques")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "Hace falta GPU (Colab T4), como en las corridas originales."
    verificar_predicciones({"A5s"}, "DESIGN_LOG.md")

    raiz = preparar_eurosat_rgb("data")
    dataset = construir_dataset(raiz, img_size=224)
    y = np.array(dataset.targets)
    tr, va, te = particion_estratificada(y, frac_val=0.15, frac_test=0.15, seed=42)
    assert (len(tr), len(va), len(te)) == (18900, 4050, 4050)
    interna = {"train": archivos_de(dataset, tr), "val": archivos_de(dataset, va), "test": archivos_de(dataset, te)}
    geo = json.load(open("splits/split_geo.json", encoding="utf-8"))
    ctx = Contexto(dataset=dataset, clases=dataset.classes, particiones={"interna": interna, "geo": geo},
                   cfg=Config(), device="cuda", amp=True)

    filas = []
    for particion, exp in (("interna", "A5"), ("geo", "A5g")):
        for semilla in (0, 1):
            c = Corrida.crear(exp, particion, "lora", semilla, bloques=(8, 9, 10, 11))
            print(f"Corrida {c.id} ...", flush=True)
            m = ejecutar_corrida(c, ctx, salida=args.salida)
            fila = {"particion": particion, "semilla": semilla, "segundos": m["segundos"]}
            fila.update({k: round(v["macro_f1"] * 100, 4) for k, v in m["metricas"].items()})
            filas.append(fila)
    t = pd.DataFrame(filas)
    print("\n=== Bloques 9–12, semillas 0 y 1 (macro-F1 %) ===")
    print(t.to_string(index=False))

    print("\n=== Latencia batch 1 (Tesla T4) ===")
    lat = remedir_fp16_b1(cargar_dinov2("facebook/dinov2-small", "cuda"), "cuda")
    lat["gpu"] = torch.cuda.get_device_name(0)
    print(lat[["precision", "warmup", "latencia_mediana_ms", "latencia_p10_ms", "latencia_p90_ms"]].round(2).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    t.to_csv("results/C12_bloques_semillas_resumen.csv", index=False)
    lat.to_csv("results/C10_latencia_fp16_b1.csv", index=False)
    shutil.make_archive(args.salida.rstrip("/"), "zip", args.salida)
    print(f"\nCorridas en {args.salida} y {args.salida.rstrip('/')}.zip")


if __name__ == "__main__":
    main()

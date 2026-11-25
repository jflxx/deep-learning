"""scripts/E3_control_correccion.py  (Colab con GPU T4, ~10 min)

Corrida de control: repite UNA corrida del E3 (LoRA r=8, Q+V, bloques 11–12, partición
geográfica, semilla 42) con el código corregido (GradScaler en fp16 y semilla fijada antes
de crear LoRA) y la compara con la corrida original de results/E3_corridas/.

Además verifica que la partición geográfica regenerada sea idéntica a la usada en la
corrida original (mismos archivos en test fuera).

Uso (desde la raíz del repo):
    python scripts/E3_control_correccion.py --salida /content/drive/MyDrive/E3_control
Salida: results/C11_control_correccion.csv
"""

import argparse
import json
import os

import pandas as pd
import torch

from eurosat_lora.config import Config
from eurosat_lora.datos import construir_dataset, preparar_eurosat_rgb
from eurosat_lora.experimentos import Contexto, Corrida, ejecutar_corrida

ID = "geo_lora_r8_b10-11_q+v_s42"
ID_BASE = "geo_base_r8_b10-11_q+v_s42"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="outputs/E3_control")
    ap.add_argument("--originales", default="results/E3_corridas")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Esta corrida debe hacerse con GPU (Colab T4), como las originales."
    split = json.load(open("splits/split_geo.json", encoding="utf-8"))

    # 1) La partición debe ser la misma que la de la corrida original
    orig_fuera = pd.read_csv(os.path.join(args.originales, ID, "predicciones_test_fuera.csv"))["archivo"]
    if sorted(orig_fuera) != sorted(split["test_fuera"]):
        raise SystemExit("La partición geográfica NO coincide con la de la corrida original. No se continúa.")
    print("Partición geográfica idéntica a la original: OK")

    raiz = preparar_eurosat_rgb("data")
    dataset = construir_dataset(raiz, img_size=224)
    ctx = Contexto(dataset=dataset, clases=dataset.classes, particiones={"geo": split},
                   cfg=Config(), device="cuda", amp=True)

    corrida = Corrida.crear("A4", "geo", "lora", 42)
    assert corrida.id == ID
    nueva = ejecutar_corrida(corrida, ctx, salida=args.salida)

    original = json.load(open(os.path.join(args.originales, ID, "metricas.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(args.originales, ID_BASE, "metricas.json"), encoding="utf-8"))
    filas = []
    for conjunto in ("val", "test_dentro", "test_fuera"):
        o = original["metricas"][conjunto]["macro_f1"] * 100
        n = nueva["metricas"][conjunto]["macro_f1"] * 100
        b = base["metricas"][conjunto]["macro_f1"] * 100
        filas.append({"conjunto": conjunto, "lora_original": o, "lora_corregida": n, "diferencia": n - o,
                      "base": b, "ventaja_original": o - b, "ventaja_corregida": n - b})
    t = pd.DataFrame(filas)
    os.makedirs("results", exist_ok=True)
    t.to_csv("results/C11_control_correccion.csv", index=False, float_format="%.4f")
    print(t.round(2).to_string(index=False))
    print("\nGuardado en results/C11_control_correccion.csv")
    print(f"Carpeta de la corrida corregida: {os.path.join(args.salida, ID)}")


if __name__ == "__main__":
    main()

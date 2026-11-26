"""app/demo_nucleo.py — Funciones núcleo para la demo Streamlit."""

import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from eurosat_lora.modelo import cargar_dinov2, aplicar_lora, adelante_lora, representacion

RUTA_CKPT = "checkpoints/geo_s42"

CLASES_EUROSAT = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def checkpoints_disponibles(ruta: str = RUTA_CKPT) -> bool:
    if not os.path.exists(ruta):
        return False
    tiene_lora = os.path.exists(os.path.join(ruta, "lora_y_cabeza.pt"))
    tiene_base = (
        os.path.exists(os.path.join(ruta, "cabeza_base.pt"))
        or os.path.exists(os.path.join(ruta, "cabeza.pt"))
    )
    return tiene_lora and tiene_base


def cargar_modelos(ruta: str = RUTA_CKPT, device: str = "cpu", modelo_id: str = "facebook/dinov2-small") -> dict:
    if not checkpoints_disponibles(ruta):
        raise FileNotFoundError(f"Checkpoints incompletos en '{ruta}'")

    m_base = cargar_dinov2(modelo_id, device=device)
    cabeza_base_path = (
        os.path.join(ruta, "cabeza_base.pt")
        if os.path.exists(os.path.join(ruta, "cabeza_base.pt"))
        else os.path.join(ruta, "cabeza.pt")
    )
    in_dim = m_base.config.hidden_size * 2
    cabeza_base = nn.Linear(in_dim, len(CLASES_EUROSAT)).to(device)
    cabeza_base.load_state_dict(torch.load(cabeza_base_path, map_location=device, weights_only=True))
    cabeza_base.eval()

    m_lora = cargar_dinov2(modelo_id, device=device)
    # Por defecto bloques 10-11 para r8
    primer_bloque = 10 if len(m_lora.encoder.layer) >= 12 else 2
    bloques = (10, 11) if len(m_lora.encoder.layer) >= 12 else (2, 3)
    aplicar_lora(m_lora, bloques=bloques, proyecciones=("q", "v"), r=8, alpha=8.0)

    cabeza_lora = nn.Linear(in_dim, len(CLASES_EUROSAT)).to(device)
    lora_ckpt = torch.load(os.path.join(ruta, "lora_y_cabeza.pt"), map_location=device, weights_only=True)
    cabeza_lora.load_state_dict(lora_ckpt["cabeza"])
    m_lora.load_state_dict(lora_ckpt["lora"], strict=False)
    cabeza_lora.eval()
    m_lora.eval()

    return {
        "base": m_base,
        "cabeza_base": cabeza_base,
        "lora": m_lora,
        "cabeza_lora": cabeza_lora,
        "primer_bloque": primer_bloque,
        "clases": CLASES_EUROSAT,
    }


def predecir_imagen(modelos: dict, img: Image.Image, device: str = "cpu", img_size: int | None = None) -> dict:
    if img_size is None:
        # Misma resolución que en el entrenamiento (224 px). OJO: config.image_size
        # de facebook/dinov2-small es 518 y NO es la que se usó para entrenar.
        img_size = 224 if modelos["base"].config.hidden_size == 384 else modelos["base"].config.image_size

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    clases = modelos["clases"]

    # Base
    with torch.no_grad():
        out_base = modelos["base"](x).last_hidden_state
        rep_base = representacion(out_base)
        logits_base = modelos["cabeza_base"](rep_base)
        probs_base = torch.softmax(logits_base, dim=-1).squeeze(0).cpu().numpy()

    # LoRA
    primer_bloque = modelos.get("primer_bloque", 10)
    with torch.no_grad():
        rep_lora = adelante_lora(modelos["lora"], x, primer_bloque=primer_bloque)
        logits_lora = modelos["cabeza_lora"](rep_lora)
        probs_lora = torch.softmax(logits_lora, dim=-1).squeeze(0).cpu().numpy()

    top3_base_idx = np.argsort(probs_base)[::-1][:3]
    top3_lora_idx = np.argsort(probs_lora)[::-1][:3]

    return {
        "base": [(clases[i], float(probs_base[i])) for i in top3_base_idx],
        "lora": [(clases[i], float(probs_lora[i])) for i in top3_lora_idx],
    }

"""diagnosticos.py — Norma de tokens y latencia fp16."""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def norma_tokens(m: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Calcula norma L2 por token del último bloque. Devuelve (B, N)."""
    m.eval()
    with torch.no_grad():
        out = m(x)
        # last_hidden_state: (B, N+1, D) — incluye CLS
        h = out.last_hidden_state  # (B, N+1, D)
        normas = h.norm(dim=-1)    # (B, N+1)
    return normas.cpu().numpy()


def medir_latencia(
    m: nn.Module,
    batch: int,
    img_size: int,
    device: str,
    reps: int = 20,
    warmup: int = 5,
    precision: str = "fp32",
) -> dict:
    """Latencia de una pasada hacia adelante, como en el E2: calentamiento, sincronización
    CUDA antes y después de cada medición y MEDIANA de `reps` repeticiones.

    precision: "fp32", "autocast" (fp16 con autocast, como en el E2) o "half" (modelo y
    entrada en fp16, sin autocast).
    """
    import copy
    m = (copy.deepcopy(m).half() if precision == "half" else m).eval().to(device)
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    if precision == "half":
        x = x.half()
    cuda = device == "cuda" and torch.cuda.is_available()

    def pasada():
        if precision == "autocast" and cuda:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                m(x)
        else:
            m(x)

    tiempos = []
    with torch.no_grad():
        for i in range(warmup + reps):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pasada()
            if cuda:
                torch.cuda.synchronize()
            if i >= warmup:
                tiempos.append((time.perf_counter() - t0) * 1000)

    return {"batch": batch, "img_size": img_size, "device": device, "precision": precision,
            "warmup": warmup, "reps": reps, "latencia_mediana_ms": float(np.median(tiempos)),
            "latencia_p10_ms": float(np.percentile(tiempos, 10)),
            "latencia_p90_ms": float(np.percentile(tiempos, 90))}


def remedir_fp16_b1(m: nn.Module, device: str) -> pd.DataFrame:
    """Anomalía del E2: fp16 con batch 1 tardó 39.5 ms (más que fp32, 8.9 ms).

    Compara fp32, fp16 con autocast y fp16 puro (.half()) con batch 1, y con poco o mucho
    calentamiento, para separar el costo de los casteos de autocast del calentamiento.
    """
    filas = []
    for warmup in (5, 50):
        for precision in ("fp32", "autocast", "half"):
            filas.append(medir_latencia(m, batch=1, img_size=224, device=device,
                                        reps=50, warmup=warmup, precision=precision))
    return pd.DataFrame(filas)

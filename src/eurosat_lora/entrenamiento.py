"""entrenamiento.py — extraer_features, entrenar_base, entrenar_lora, predecir."""
# (helper cargador_desde_subset exportado para experimentos.py)

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eurosat_lora.config import Config
from eurosat_lora.metricas import metricas
from eurosat_lora.modelo import representacion, adelante_lora


def extraer_features(
    m: nn.Module,
    dl: DataLoader,
    device: str = "cpu",
    amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    m.eval()
    m.to(device)
    feats_list = []
    targets_list = []
    use_amp = amp and device == "cuda" and torch.cuda.is_available()

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = m(x).last_hidden_state
                    rep = representacion(out)
            else:
                out = m(x).last_hidden_state
                rep = representacion(out)
            feats_list.append(rep.cpu())
            targets_list.append(y.clone().cpu())

    return torch.cat(feats_list, dim=0), torch.cat(targets_list, dim=0)


def cargador_desde_subset(dataset, indices, batch=64, shuffle=False, seed=42, num_workers=0):
    """Helper para experimentos.py — devuelve DataLoader."""
    from torch.utils.data import Subset
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        pin_memory=False,
    )


# Alias para compatibilidad con otros módulos
cargador = cargador_desde_subset


def entrenar_base(
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    n_clases: int,
    cfg: Config,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[nn.Linear, pd.DataFrame]:
    torch.manual_seed(seed)
    X_tr, y_tr = cache["train"]
    X_va, y_va = cache["val"]

    in_dim = X_tr.shape[1]
    cabeza = nn.Linear(in_dim, n_clases).to(device)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.AdamW(cabeza.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ds_tr = TensorDataset(X_tr, y_tr)
    g = torch.Generator()
    g.manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True, generator=g)

    historial = []
    mejor_f1 = -1.0
    mejor_estado = None

    for epoca in range(1, cfg.epocas + 1):
        cabeza.train()
        total_loss = 0.0
        n_batches = 0
        for x_b, y_b in dl_tr:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizador.zero_grad()
            logits = cabeza(x_b)
            loss = criterio(logits, y_b)
            loss.backward()
            optimizador.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        preds_va, _ = predecir_base(cabeza, X_va, device=device)
        m_va = metricas(y_va.numpy(), preds_va)

        historial.append({
            "epoca": epoca,
            "loss_train": avg_loss,
            "val_accuracy": m_va["accuracy"],
            "val_macro_f1": m_va["macro_f1"],
        })

        if m_va["macro_f1"] > mejor_f1:
            mejor_f1 = m_va["macro_f1"]
            mejor_estado = copy.deepcopy(cabeza.state_dict())

    if mejor_estado is not None:
        cabeza.load_state_dict(mejor_estado)

    return cabeza, pd.DataFrame(historial)


def predecir_base(
    cabeza: nn.Linear,
    X: torch.Tensor,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    cabeza.eval()
    cabeza.to(device)
    with torch.no_grad():
        logits = cabeza(X.to(device))
        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)
    return preds.cpu().numpy(), max_probs.cpu().numpy()


def entrenar_lora(
    m: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    n_clases: int,
    cfg: Config,
    seed: int = 42,
    device: str = "cpu",
    amp: bool = False,
) -> tuple[nn.Linear, pd.DataFrame, dict]:
    torch.manual_seed(seed)
    m.to(device)
    primer_bloque = min(cfg.lora_bloques)
    in_dim = m.config.hidden_size * 2
    cabeza = nn.Linear(in_dim, n_clases).to(device)

    params_entrenables = [p for p in m.parameters() if p.requires_grad] + list(cabeza.parameters())
    optimizador = torch.optim.AdamW(params_entrenables, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterio = nn.CrossEntropyLoss()
    use_amp = amp and device == "cuda" and torch.cuda.is_available()
    # Como en el E2: escalar la pérdida en fp16 para que los gradientes pequeños
    # (B arranca en 0) no se redondeen a cero.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    historial = []
    mejor_f1 = -1.0
    mejor_cabeza_estado = None
    mejor_lora_estado = None

    for epoca in range(1, cfg.epocas + 1):
        m.train()
        cabeza.train()
        total_loss = 0.0
        n_batches = 0

        for x_b, y_b in dl_train:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizador.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    rep = adelante_lora(m, x_b, primer_bloque=primer_bloque)
                    logits = cabeza(rep)
                    loss = criterio(logits, y_b)
            else:
                rep = adelante_lora(m, x_b, primer_bloque=primer_bloque)
                logits = cabeza(rep)
                loss = criterio(logits, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizador)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        preds_va, _ = predecir_lora(m, cabeza, dl_val, primer_bloque, device, amp)
        # Recolectar etiquetas de validación de forma estable
        y_val_arr = []
        for _, y_b in dl_val:
            y_val_arr.extend(y_b.numpy().tolist())
        y_va = np.array(y_val_arr)
        m_va = metricas(y_va, preds_va)

        historial.append({
            "epoca": epoca,
            "loss_train": avg_loss,
            "val_accuracy": m_va["accuracy"],
            "val_macro_f1": m_va["macro_f1"],
        })

        if m_va["macro_f1"] > mejor_f1:
            mejor_f1 = m_va["macro_f1"]
            mejor_cabeza_estado = copy.deepcopy(cabeza.state_dict())
            mejor_lora_estado = {
                k: v.cpu().clone()
                for k, v in m.state_dict().items()
                if k.split(".")[-1] in ("A", "B")
            }

    if mejor_cabeza_estado is not None:
        cabeza.load_state_dict(mejor_cabeza_estado)
    if mejor_lora_estado is not None:
        m.load_state_dict(mejor_lora_estado, strict=False)

    pesos = {
        "cabeza": cabeza.state_dict(),
        "lora": mejor_lora_estado if mejor_lora_estado is not None else {},
    }
    return cabeza, pd.DataFrame(historial), pesos


def predecir_lora(
    m: nn.Module,
    cabeza: nn.Linear,
    dl: DataLoader,
    primer_bloque: int,
    device: str = "cpu",
    amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    m.eval()
    cabeza.eval()
    m.to(device)
    cabeza.to(device)
    use_amp = amp and device == "cuda" and torch.cuda.is_available()

    preds_list = []
    probs_list = []

    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    rep = adelante_lora(m, x, primer_bloque=primer_bloque)
                    logits = cabeza(rep)
            else:
                rep = adelante_lora(m, x, primer_bloque=primer_bloque)
                logits = cabeza(rep)
            probs = torch.softmax(logits, dim=-1)
            max_p, p_cls = probs.max(dim=-1)
            preds_list.append(p_cls.cpu().numpy())
            probs_list.append(max_p.cpu().numpy())

    if not preds_list:
        return np.array([]), np.array([])
    return np.concatenate(preds_list), np.concatenate(probs_list)

from __future__ import annotations

import os
import torch


def resolve_torch_device(require_directml: bool = False) -> tuple[torch.device, str]:
    directml_error: str | None = None
    try:
        import torch_directml  # type: ignore

        adapter_index_raw = os.getenv("TORCH_DIRECTML_DEVICE_INDEX")
        if adapter_index_raw is not None and adapter_index_raw.strip() != "":
            adapter_index = int(adapter_index_raw)
            dml_device = torch_directml.device(adapter_index)
            backend_label = f"directml:{adapter_index}"
        else:
            dml_device = torch_directml.device()
            backend_label = "directml"
        _ = torch.zeros(1, dtype=torch.float32, device=dml_device)
        return dml_device, backend_label
    except Exception as exc:
        directml_error = str(exc)

    if require_directml:
        raise RuntimeError(
            "DirectML/Arc GPU nicht verfügbar. Installiere 'torch-directml' und setze optional "
            "TORCH_DIRECTML_DEVICE_INDEX (z. B. 0 oder 1)."
            + (f" Letzter Fehler: {directml_error}" if directml_error else "")
        )

    return torch.device("cpu"), "cpu"

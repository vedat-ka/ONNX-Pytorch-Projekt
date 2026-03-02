from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LogAutoencoder(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int):
		super().__init__()
		bottleneck_dim = max(8, hidden_dim // 2)
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, bottleneck_dim),
			nn.ReLU(),
		)
		self.decoder = nn.Sequential(
			nn.Linear(bottleneck_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, input_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		latent = self.encoder(x)
		return self.decoder(latent)


@dataclass
class TrainArtifacts:
	model: LogAutoencoder
	losses: list[float]
	threshold: float


def train_autoencoder(
	features: np.ndarray,
	hidden_dim: int,
	epochs: int,
	batch_size: int,
	learning_rate: float,
	threshold_quantile: float,
	progress_prefix: str = "",
	device: torch.device | None = None,
) -> TrainArtifacts:
	if features.ndim != 2:
		raise ValueError("Features müssen 2-dimensional sein.")

	input_dim = features.shape[1]
	model = LogAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
	if device is not None:
		model = model.to(device)
	criterion = nn.MSELoss()
	is_directml = device is not None and device.type == "privateuseone"
	if is_directml:
		print("[DEVICE] DirectML erkannt: Verwende SGD-Optimizer statt Adam.", flush=True)
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=False)

	tensor_data = torch.from_numpy(features.astype(np.float32))
	data_loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True)

	losses: list[float] = []
	model.train()
	for epoch_index in range(epochs):
		running = 0.0
		batches = 0
		for (batch_x,) in data_loader:
			if device is not None:
				batch_x = batch_x.to(device)
			optimizer.zero_grad()
			reconstructed = model(batch_x)
			loss = criterion(reconstructed, batch_x)
			loss.backward()
			optimizer.step()
			running += float(loss.item())
			batches += 1
		epoch_loss = running / max(1, batches)
		losses.append(epoch_loss)
		prefix = f"{progress_prefix} " if progress_prefix else ""
		print(f"[TRAIN] {prefix}Epoch {epoch_index + 1}/{epochs} - loss={epoch_loss:.6f}", flush=True)

	scores = reconstruction_errors(model, features, device=device)
	threshold = float(np.quantile(scores, threshold_quantile))
	return TrainArtifacts(model=model, losses=losses, threshold=threshold)


def reconstruction_errors(
	model: LogAutoencoder,
	features: np.ndarray,
	device: torch.device | None = None,
) -> np.ndarray:
	if device is None:
		device = next(model.parameters()).device
	model = model.to(device)
	model.eval()
	with torch.no_grad():
		x = torch.from_numpy(features.astype(np.float32)).to(device)
		reconstructed = model(x)
		mse = torch.mean((reconstructed - x) ** 2, dim=1)
	return mse.cpu().numpy()


def export_onnx(model: LogAutoencoder, input_dim: int, onnx_path: str | Path) -> None:
	model.eval()
	onnx_output_path = Path(onnx_path)
	dummy_input = torch.randn(1, input_dim, dtype=torch.float32)
	try:
		torch.onnx.export(
			model,
			dummy_input,
			str(onnx_output_path),
			input_names=["input"],
			output_names=["reconstruction"],
			dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
			opset_version=18,
			dynamo=False,
		)

		model_proto = onnx.load_model(str(onnx_output_path), load_external_data=True)
		onnx.save_model(model_proto, str(onnx_output_path), save_as_external_data=False)

		external_data_path = onnx_output_path.with_name(f"{onnx_output_path.name}.data")
		if external_data_path.exists():
			external_data_path.unlink()
	except ModuleNotFoundError as exc:
		if exc.name == "onnxscript":
			raise RuntimeError(
				"ONNX-Export fehlgeschlagen: Paket 'onnxscript' fehlt. "
				"Bitte 'pip install -r requirements.txt' ausführen."
			) from exc
		raise

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.encoder_input_projection = nn.Linear(1, d_model)
        self.decoder_input_projection = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[None, torch.Tensor]:
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()
        tgt_mask = nopeak_mask.unsqueeze(1)
        return None, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if (not torch.jit.is_tracing()) and (src.size(1) > self.max_seq_length or tgt.size(1) > self.max_seq_length):
            raise ValueError("Input sequence length exceeds configured max_seq_length.")

        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_input_projection(src.unsqueeze(-1))))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_input_projection(tgt.unsqueeze(-1))))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output).squeeze(-1)
        return output

    def reconstruct(self, src: torch.Tensor) -> torch.Tensor:
        tgt = torch.zeros_like(src)
        tgt[:, 1:] = src[:, :-1]
        return self.forward(src, tgt)


@dataclass
class TrainArtifacts:
    model: Transformer
    losses: list[float]
    threshold: float


def train_transformer_autoencoder(
    features: np.ndarray,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threshold_quantile: float,
    progress_prefix: str = "",
    device: torch.device | None = None,
) -> TrainArtifacts:
    if features.ndim != 2:
        raise ValueError("Features müssen 2-dimensional sein.")

    seq_length = features.shape[1]
    model = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=seq_length,
        dropout=dropout,
    )
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
            reconstructed = model.reconstruct(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
        epoch_loss = running / max(1, batches)
        losses.append(epoch_loss)
        prefix = f"{progress_prefix} " if progress_prefix else ""
        print(f"[TRAIN] {prefix}Epoch {epoch_index + 1}/{epochs} - loss={epoch_loss:.6f}", flush=True)

    scores = transformer_reconstruction_errors(model, features, device=device, batch_size=batch_size)
    threshold = float(np.quantile(scores, threshold_quantile))
    return TrainArtifacts(model=model, losses=losses, threshold=threshold)


def transformer_reconstruction_errors(
    model: Transformer,
    features: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()
    scores_chunks: list[np.ndarray] = []
    effective_batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, features.shape[0], effective_batch_size):
            end = min(start + effective_batch_size, features.shape[0])
            x = torch.from_numpy(features[start:end].astype(np.float32)).to(device)
            reconstructed = model.reconstruct(x)
            mse = torch.mean((reconstructed - x) ** 2, dim=1)
            scores_chunks.append(mse.cpu().numpy())
    if not scores_chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(scores_chunks, axis=0)


class _TransformerOnnxWrapper(nn.Module):
    def __init__(self, model: Transformer):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.reconstruct(x)


def export_transformer_onnx(model: Transformer, input_dim: int, onnx_path: str | Path) -> None:
    model.eval()
    onnx_output_path = Path(onnx_path)
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)
    wrapper = _TransformerOnnxWrapper(model)
    wrapper.eval()

    try:
        torch.onnx.export(
            wrapper,
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

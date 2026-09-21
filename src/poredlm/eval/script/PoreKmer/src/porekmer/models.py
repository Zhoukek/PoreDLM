"""Small readouts for cached, frozen V003 event features.

Every input has five ordered slots: events ``e-2, e-1, e, e+1, e+2``.
The context readout masks slots outside the chosen centered window, keeping
its parameter count identical for the one-, three-, and five-event models.
No waveform encoder, move labels, or base sequence is an input to this module.
The features themselves may already contain V003's full-chunk context.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .bank import KmerHiddenBank


_INTEGER_DTYPES = {
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
}


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _position_encoding(dimension: int) -> torch.Tensor:
    """Fixed signed positions distinguish upstream from downstream events."""
    positions = torch.arange(-2, 3, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.empty(5, dimension)
    encoding[:, 0::2] = torch.sin(positions * frequency)
    encoding[:, 1::2] = torch.cos(positions * frequency[: dimension // 2])
    return encoding


class EventContextClassifier(nn.Module):
    """Classify the center event from an ordered, optional neighborhood.

    ``context`` uses a shared projection, fixed positional encoding, one
    dropout-free Transformer layer, and a learned cosine class bank. Its
    trainable parameter count is independent of ``window_size``. ``linear``
    and ``cosine`` are center-only baselines and require ``window_size=1``.

    ``encode`` returns the center readout embedding before cosine
    normalization (or the unchanged center input for the two baselines).
    Features are cast to the model's floating dtype, but must already reside
    on the same device as the model. Cached half-precision inputs are accepted.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 768,
        num_classes: int = 1024,
        window_size: int = 1,
        head: str = "context",
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        self.num_classes = _positive_integer(num_classes, "num_classes", minimum=2)
        self.projection_dim = _positive_integer(
            projection_dim, "projection_dim", minimum=2
        )
        self.window_size = _positive_integer(window_size, "window_size")
        if self.window_size not in (1, 3, 5):
            raise ValueError("window_size must be 1, 3, or 5")
        if head not in ("context", "cosine", "linear"):
            raise ValueError("head must be 'context', 'cosine', or 'linear'")
        if head != "context" and self.window_size != 1:
            raise ValueError("cosine and linear heads require window_size=1")
        self.head = head

        if self.head == "context":
            self.projection = nn.Linear(self.hidden_dim, self.projection_dim)
            self.register_buffer(
                "position_encoding", _position_encoding(self.projection_dim)
            )
            # Do not copy the window selection when transferring otherwise
            # identical weights between ablations via load_state_dict().
            inactive = torch.arange(-2, 3).abs() > self.window_size // 2
            self.register_buffer("inactive_positions", inactive, persistent=False)
            self.context_layer = nn.TransformerEncoderLayer(
                d_model=self.projection_dim,
                nhead=4 if self.projection_dim % 4 == 0 else 1,
                dim_feedforward=2 * self.projection_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.output_norm = nn.LayerNorm(self.projection_dim)
            self.classifier = KmerHiddenBank(
                num_classes=self.num_classes, hidden_dim=self.projection_dim
            )
        elif self.head == "cosine":
            self.classifier = KmerHiddenBank(
                num_classes=self.num_classes, hidden_dim=self.hidden_dim
            )
        else:
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def architecture_config(self) -> dict[str, int | str]:
        """Return JSON-serializable arguments that reconstruct this model."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "window_size": self.window_size,
            "head": self.head,
            "projection_dim": self.projection_dim,
        }

    def parameter_count(self) -> int:
        """Number of trainable scalar parameters (buffers excluded)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters(self) -> int:
        return self.parameter_count()

    def _validate_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 3
            or inputs.shape[0] == 0
            or tuple(inputs.shape[1:]) != (5, self.hidden_dim)
        ):
            raise ValueError(
                f"inputs must have nonempty shape [N, 5, {self.hidden_dim}]"
            )
        if not inputs.is_floating_point():
            raise ValueError("inputs must be floating-point hidden features")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must contain only finite values")
        parameter = next(self.parameters())
        if inputs.device != parameter.device:
            raise ValueError("inputs and model must be on the same device")
        return inputs.to(dtype=parameter.dtype)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one embedding per center event; never consume labels."""
        inputs = self._validate_inputs(inputs)
        if self.head != "context":
            return inputs[:, 2, :]
        # Explicit zeroing also prevents inactive extreme finite features from
        # overflowing in projection before the attention key mask is applied.
        inputs = inputs.masked_fill(self.inactive_positions[None, :, None], 0.0)
        projected = self.projection(inputs) + self.position_encoding[None, :, :]
        mask = self.inactive_positions[None, :].expand(inputs.shape[0], -1)
        encoded = self.context_layer(projected, src_key_padding_mask=mask)
        center = self.output_norm(encoded[:, 2, :])
        if not bool(torch.isfinite(center).all()):
            raise ValueError("encoded features became non-finite")
        return center

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits with shape ``[N, num_classes]``."""
        logits = self.classifier(self.encode(inputs))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("class logits became non-finite")
        return logits


def read_balanced_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    read_indices: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average event CE within each read, then average the represented reads.

    Events, not frames, are the units, so no dwell weighting is applied.
    Optional class weights are normalized *within each read*, retaining an
    equal outer weight for every represented read. Every read must have a
    positive total class weight. Read IDs may be non-contiguous integers.
    All tensor arguments must reside on the logits' device. Invalid targets
    are rejected rather than silently ignored: filtering belongs upstream.
    """
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[0] == 0
        or logits.shape[1] < 2
        or not logits.is_floating_point()
    ):
        raise ValueError("logits must have nonempty floating shape [N, C] with C >= 2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must contain only finite values")
    for name, values in (("labels", labels), ("read_indices", read_indices)):
        if (
            not isinstance(values, torch.Tensor)
            or tuple(values.shape) != (logits.shape[0],)
            or values.dtype not in _INTEGER_DTYPES
        ):
            raise ValueError(f"{name} must have integer shape [N]")
        if values.device != logits.device:
            raise ValueError(f"{name} and logits must be on the same device")
    if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
        raise ValueError("labels must be in [0, num_classes)")
    if bool((read_indices < 0).any()):
        raise ValueError("read_indices must be non-negative")

    targets = labels.to(dtype=torch.long)
    stable_logits = (
        logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    )
    event_losses = F.cross_entropy(stable_logits, targets, reduction="none")
    event_weights = torch.ones_like(event_losses)
    if class_weights is not None:
        if (
            not isinstance(class_weights, torch.Tensor)
            or tuple(class_weights.shape) != (logits.shape[1],)
            or not class_weights.is_floating_point()
        ):
            raise ValueError("class_weights must have floating shape [num_classes]")
        if class_weights.device != logits.device:
            raise ValueError("class_weights and logits must be on the same device")
        if not bool(torch.isfinite(class_weights).all()) or bool((class_weights < 0).any()):
            raise ValueError("class_weights must be finite and non-negative")
        event_weights = class_weights.to(dtype=event_losses.dtype)[targets]

    unique_reads, inverse = torch.unique(
        read_indices.to(dtype=torch.long), sorted=True, return_inverse=True
    )
    read_losses = event_losses.new_zeros(unique_reads.numel()).scatter_add(
        0, inverse, event_losses * event_weights
    )
    read_weights = event_losses.new_zeros(unique_reads.numel()).scatter_add(
        0, inverse, event_weights
    )
    if not bool(torch.isfinite(read_losses).all()) or not bool(torch.isfinite(read_weights).all()):
        raise ValueError("weighted read loss became non-finite")
    if bool((read_weights <= 0).any()):
        raise ValueError("every represented read must have positive total class weight")
    return (read_losses / read_weights).mean()

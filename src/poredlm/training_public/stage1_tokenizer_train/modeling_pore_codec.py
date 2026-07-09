import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
from einops import reduce

# =========================================================================
# 🌍 Global Constants Config Management
# =========================================================================
MODEL_TYPE = "pore_codec_rsqf42c12a"

###
class PoreCNNModel(nn.Module):
    """
    Static Nanopore Electrical Signal CNN Encoding and Decoding Network.
    (High-performance original cnn_type==2 architecture)
    """

    # --- Model Configuration Constants ---
    stride = 4           # Signal downsampling factor (input_len / output_len)
    out_channels = 512   # Dimensionality of the latent feature space
    RF = 33              # Total receptive field size (number of input samples per feature)
    type = 12            # Unique identifier for the architecture registry


    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 512, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(512),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 1, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


###

class PoreRSQCodecConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        fsq_levels: str = "5 5 5 5",
        codebook_size: int = 625,
        codebook_nqtz: int = 2,
        cnn_type: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fsq_levels = fsq_levels
        self.codebook_size = codebook_size
        self.codebook_nqtz = codebook_nqtz
        self.cnn_type = cnn_type


# --- Core Straight-Through Estimator (STE) Operator ---
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

def round_ste(x):
    return RoundSTE.apply(x)

class PoreResidualFSQ(nn.Module):
    """
    PoreResidualFSQ: Production-optimized Residual Finite Scalar Quantization.
    All codebook data are pre-stored within the safetensors weight file.
    """
    def __init__(
        self,
        levels: List[int],
        num_quantizers: int,
        dim: int,
        quantize_dropout: bool = False,
        quantize_dropout_cutoff_index: int = 0
    ):
        super().__init__()
        self.levels = list(levels)
        self.num_quantizers = num_quantizers
        self.codebook_dim = len(levels)
        self.dim = dim
        self.codebook_size = math.prod(self.levels)

        # Core Projections: Map input dim to FSQ codebook dimensionality
        requires_projection = self.codebook_dim != self.dim
        self.project_in = nn.Linear(self.dim, self.codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(self.codebook_dim, self.dim) if requires_projection else nn.Identity()

        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.register_buffer('codebooks', torch.randn(num_quantizers, self.codebook_size, self.codebook_dim) * 0.1, persistent=True)

    def _get_constants(self, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates and caches exhaustive FSQ scaling factors as pure static float32 tensors.
        """
        levels_tensor = torch.tensor(self.levels, dtype=torch.float32, device=device)
        basis = torch.cumprod(torch.tensor([1] + self.levels[:-1], dtype=torch.long, device=device), dim=0)
        scales = torch.stack([levels_tensor ** -ind for ind in range(self.num_quantizers)])
        
        # Explicitly avoid float-based floor division to ensure compatibility across PyTorch versions
        floor_levels = torch.tensor([lvl // 2 for lvl in self.levels], dtype=torch.float32, device=device)
        return levels_tensor, basis, scales, floor_levels

    def _generate_full_codebook(self, device=torch.device('cpu')) -> torch.Tensor:
        """
        Generates the mathematically exhaustive FSQ codebook mapped across all residual layers.
        Guarantees perfect 1:1 row-index alignment with `get_codes_from_indices` via basis inversion.
        """
        levels_tensor, basis, scales, floor_levels = self._get_constants(device)

        # 1. Allocate a flat, contiguous sequence of absolute row indices
        indices = torch.arange(self.codebook_size, device=device, dtype=torch.long)

        # 2. Reconstruct spatial grid coordinates from linear indices
        remainder = indices.unsqueeze(-1)
        coords = torch.zeros(self.codebook_size, self.codebook_dim, device=device, dtype=torch.float32)

        for i in range(self.codebook_dim - 1, -1, -1):
            mul = basis[i]
            coords[:, i] = (remainder // mul).squeeze(-1).to(torch.float32)
            remainder = remainder % mul

        # 3. Shift and bound coordinate spaces to symmetric range
        half_l = (levels_tensor - 1) / 2
        z_quantized = coords - half_l

        # 4. Normalize grid locations relative to individual integer step-sizes
        single_layer_codebook = z_quantized / floor_levels

        # 5. Broadcast layer-specific residual quantization scales
        all_layers = []
        for ind in range(self.num_quantizers):
            scale = scales[ind]
            all_layers.append(single_layer_codebook * scale)

        return torch.stack(all_layers, dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        levels_tensor, basis, scales, floor_levels = self._get_constants(device)
        x_proj = self.project_in(x)

        quantized_out = 0.
        residual = x_proj
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout and torch.is_grad_enabled()
        null_indices = torch.full(x.shape[:2], -1, device=device, dtype=torch.long) if should_quantize_dropout else None

        # Pre-define constants for coordinate boundary enforcement
        levels_tensor = torch.tensor(self.levels, device=device, dtype=torch.long)
        min_val = torch.tensor(0, device=device, dtype=torch.long)
        max_val = levels_tensor - 1

        for ind in range(self.num_quantizers):
            # Apply structural dropout by skipping layers during training
            if should_quantize_dropout and ind > torch.randint(self.quantize_dropout_cutoff_index, self.num_quantizers, ()).item():
                all_indices.append(null_indices)
                continue

            scale = scales[ind]
            z = residual / scale
            half_l = (levels_tensor - 1) / 2
            clamp_val = 1.0 + (1.0 / (levels_tensor - 1))
            
            # Apply tanh-based soft-clamping for stable gradient propagation
            z_bounded = (z / clamp_val).tanh() * clamp_val
            z_quantized = round_ste(z_bounded * half_l)
            
            # Convert to integer coordinates with rounding to ensure numerical stability
            coords = (z_quantized + half_l).round().to(torch.long)
            
            # Strict boundary enforcement to prevent index overflow
            coords = torch.clamp(coords, min=min_val, max=max_val)

            indices = (coords * basis).sum(dim=-1)
            
            # Critical assertion for index range integrity
            if indices.max() >= self.codebook_size:
                print(f"[DEBUG CRITICAL] Out-of-bounds index detected! Max Index: {indices.max().item()}, Codebook Size: {self.codebook_size}")
                assert False, "Index calculation overflow!"
            all_indices.append(indices)

            quantized = (z_quantized / floor_levels) * scale
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

        return self.project_out(quantized_out), torch.stack(all_indices, dim=-1)

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Pure O(1) table-lookup decoding"""
        mask = indices == -1
        safe_indices = indices.masked_fill(mask, 0)

        q_indices = torch.arange(self.num_quantizers, device=indices.device).view(1, 1, -1)
        safe_indices = torch.clamp(safe_indices, 0, self.codebook_size - 1)
        codes = self.codebooks[q_indices, safe_indices]

        # Nullify masked residual contributions
        codes = codes.masked_fill(mask.unsqueeze(-1), 0.)
        return codes.permute(2, 0, 1, 3)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.get_codes_from_indices(indices)
        return self.project_out(reduce(codes, 'q b n d -> b n d', 'sum'))

class PoreRSQCodec(PreTrainedModel):
    """
    PoreRSQCodec: Standardized and efficient codec for nanopore signals.
    All codebook weights are automatically loaded via the safetensors weight file.
    """
    config_class = PoreRSQCodecConfig
    _no_split_modules = ["PoreCNNModel", "PoreResidualFSQ"]

    def __init__(self, config: PoreRSQCodecConfig):
        # 🟢 Defensive Patch for transformers v4.40+ Compatibility:
        # Explicitly initialize weight-tying properties before super().__init__ to prevent
        # internal hook failures triggered by child module registrations during setup.
        self.all_tied_weights_keys = {}
        self._tied_weights_keys = set()

        super().__init__(config)

        # 1. Backbone Network Architecture
        self.cnn_model = PoreCNNModel()
        cnn_output_dim = self.cnn_model.out_channels
        self.cnn_stride = self.cnn_model.stride

        # 2. Configuration Parsing
        self.fsq_levels = [int(x) for x in config.fsq_levels.split()] if isinstance(config.fsq_levels, str) else config.fsq_levels
        num_levels = len(self.fsq_levels)
        self.num_quantizers = int(config.codebook_nqtz)

        # 3. Projection Layers
        self.project_in = nn.Linear(cnn_output_dim, num_levels)
        self.project_out = nn.Linear(num_levels, cnn_output_dim)

        # 4. Residual Quantization Operator
        self.vq = PoreResidualFSQ(
            levels=self.fsq_levels,
            num_quantizers=self.num_quantizers,
            dim=num_levels
        )

    def forward(
        self, 
        signal: torch.Tensor 
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs the forward pass of the PoreRSQCodec, mapping continuous signals to 
        discrete residual quantized representations and back.

        The pipeline executes a four-stage process:
        1. Feature Extraction: Processes raw electrical signals through a 1D-CNN backbone.
        2. Projection: Maps extracted latent features into the quantization codebook space.
        3. Residual Quantization: Performs multi-layer vector quantization to obtain 
           discrete level indices.
        4. Reconstruction: Projects quantized representations back to continuous space 
           and decodes them via the CNN backbone.

        Args:
            signal (torch.Tensor): Input continuous signal tensor. 
                Expected shape: [B, C, T], where B=Batch size, C=1, T=Sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - recon (torch.Tensor): The reconstructed signal after quantization, 
                  shape [B, C, T].
                - level_indices (torch.Tensor): The discrete quantization indices, 
                  shape [B, N, K], where N is the latent sequence length and 
                  K is the number of quantizers.

        Note:
            The implementation includes automated spatial alignment to compensate for 
            strided convolution operations, ensuring the output length matches the input.
        """    
        # Feature extraction via deep 1D convolutional layers
        z_cnn = self.cnn_model.encode(signal)   # [B, C, N]
        z_permuted = z_cnn.permute(0, 2, 1)          # [B, N, C]

        # Vector projection mapping to match codebook dimensions
        z_projected = self.project_in(z_permuted)     # [B, N, num_levels]
        z_quantized_projected, level_indices = self.vq(z_projected)

        # Ensure long-integer format compliance for cross-entropy downstream models
        level_indices = level_indices.to(device=signal.device, dtype=torch.long)

        # Inverse projection to continuous latent channel size
        z_quantized_permuted = self.project_out(z_quantized_projected)
        z_quantized = z_quantized_permuted.permute(0, 2, 1)
        recon = self.cnn_model.decode(z_quantized)

        # Spatial boundary sequence alignment via explicit truncation or padding margins
        target_len = signal.shape[-1]
        current_len = recon.shape[-1]
        if current_len > target_len:
            recon = recon[..., :target_len]
        elif current_len < target_len:
            recon = F.pad(recon, (0, target_len - current_len))
        return recon, level_indices

    def tokenize_indices(self, level_indices: torch.Tensor, layer: int = 0) -> torch.Tensor:
        if level_indices.dim() != 3:
            raise ValueError("level_indices must be a 3D tensor of shape [B, N, K]")

        B, N, K = level_indices.shape
        use_layers = K if layer == 0 else layer
        if use_layers > K or use_layers < 1:
            raise ValueError(f"Invalid layer requested: available layers K={K}, requested layer={use_layers}")

        selected_indices = level_indices[:, :, :use_layers].to(torch.long)
        base_codebook_size = math.prod(self.fsq_levels)
        device = level_indices.device

        exponents = torch.arange(use_layers - 1, -1, -1, device=device, dtype=torch.long)
        base_tensor = torch.tensor(base_codebook_size, dtype=torch.long, device=device)
        multipliers = torch.pow(base_tensor, exponents)

        weighted = selected_indices * multipliers.view(1, 1, -1)
        uni_indices = torch.sum(weighted, dim=-1)
        return uni_indices


    def decode_indices(self, level_indices: torch.Tensor, layer: int = 0) -> torch.Tensor:
        B, N, K = level_indices.shape
        device = level_indices.device

        # -------------------------------------------------------------
        # 🚨 Mathematical Fix: Overwrite unactivated tracks with null_index instead of 0
        # -------------------------------------------------------------
        if layer == 0:
            selected_indices = level_indices.to(torch.long)
        else:
            selected_indices = level_indices.clone().to(torch.long)
            if layer < K:
                base_codebook_size = math.prod(self.fsq_levels)
                null_index = (base_codebook_size - 1) // 2
                # Overwrite unselected layers directly with the target device safe null_index
                selected_indices[:, :, layer:] = null_index

        # Pass the robust [B, N, K] tensor into the multi-layer VQ lookup space
        z_quantized_projected = self.vq.get_output_from_indices(selected_indices)

        # -------------------------------------------------------------
        # 🧱 Standard Feature Projection and CNN Decoding Pipeline
        # -------------------------------------------------------------
        z_quantized_permuted = self.project_out(z_quantized_projected)
        z_quantized = z_quantized_permuted.permute(0, 2, 1)

        recon = self.cnn_model.decode(z_quantized)

        # Dynamic spatial sequence alignment due to stride padding margins
        target_len = N * self.cnn_stride
        current_len = recon.shape[-1]
        if current_len > target_len:
            recon = recon[..., :target_len]
        elif current_len < target_len:
            recon = F.pad(recon, (0, target_len - current_len))

        return recon

    def encode_signal(self, signal: torch.Tensor, layer: int = 0) -> torch.Tensor:
        """
        Encodes preprocessed signal tensors into discrete token identifiers.

        Crucial: This interface expects structured, preprocessed tensors (signal) 
        directly from the feature extraction pipeline. Raw numpy arrays or unaligned 
        sequences will trigger an assertion failure.

        Args:
            signal (torch.Tensor): Preprocessed continuous signal tensor.
                Expected shape: [B, C, T] where B is batch size, C=1, and T is sequence length.
            layer (int): Hierarchical lookup tracking target layer depth.
                0 returns full residual combined streams, >=1 caps at specific backbone depth.

        Returns:
            torch.Tensor: Quantized discrete token id matrix. Shape: [B, N].
        """
        self.eval()
        
        # Enforce strict input tensor protocol invariants
        assert isinstance(signal, torch.Tensor), (
            f"Input validation failed: Expected torch.Tensor, received '{type(signal).__name__}'."
        )
        assert signal.dim() == 3, (
            f"Dimensional invariant broken: Expected a 3D tensor [B, C, T], received shape {list(signal.shape)}."
        )

        device = next(self.parameters()).device
        tensor_device = signal.device
        
        # Ensure identical device placement context
        if tensor_device != device:
            signal = signal.to(device)

        with torch.no_grad():
            _, level_indices = self.forward(signal)
            
            # Compress multi-stage lookup streams into unified numerical matrix tokens
            token_ids = self.tokenize_indices(level_indices, layer=layer)
            
        return token_ids

    def decode_token(self, token_ids: torch.Tensor, layer: int = 0) -> torch.Tensor:
        self.eval()
        device = token_ids.device
        B, N = token_ids.shape
        K = self.num_quantizers

        extract_layers = K if layer == 0 else layer

        base_codebook_size = math.prod(self.fsq_levels)
        exponents = torch.arange(extract_layers - 1, -1, -1, device=device, dtype=torch.long)
        base_tensor = torch.tensor(base_codebook_size, dtype=torch.long, device=device)
        multipliers = torch.pow(base_tensor, exponents)

        remainder = token_ids.clone()

        null_index = (base_codebook_size - 1) // 2
        reconstructed_level_indices = torch.full((B, N, K), null_index, dtype=torch.long, device=device)

        for i in range(extract_layers):
            mul = multipliers[i]
            reconstructed_level_indices[:, :, i] = remainder // mul
            remainder = remainder % mul

        with torch.no_grad():
            recon_signal = self.decode_indices(reconstructed_level_indices, layer=layer)
        return recon_signal

# --- Auto-Class Static Registration Mechanism ---
AutoConfig.register(MODEL_TYPE, PoreRSQCodecConfig)
AutoModel.register(PoreRSQCodecConfig, PoreRSQCodec)

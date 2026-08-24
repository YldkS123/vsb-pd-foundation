"""Full VSB pipeline model: encoder -> aggregator -> cyclic -> classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from .cyclic import noisy_or_probs


class VSBPipeline(nn.Module):
    """Full VSB partial discharge detection pipeline.

    Assembly:
        windows(B,3,K,8192) + features(B,3,K,58)
        -> WindowEncoder -> (B,3,K,128)  -- per-window representations
        -> MILAggregator -> (B,3,128)     -- per-phase aggregation
        -> CyclicPhaseModule -> (B,3,128) -- phase interaction
        -> PhaseClassifier -> (B,3,1)     -- per-phase logits
        -> sigmoid -> phase probs
        -> noisy_or -> measurement prob
    """

    def __init__(
        self,
        encoder: nn.Module,
        aggregator: nn.Module,
        cyclic: nn.Module,
        classifier: nn.Module,
        max_encode_chunk: int | None = None,
        checkpoint_chunks: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.cyclic = cyclic
        self.classifier = classifier
        self.max_encode_chunk = max_encode_chunk
        self.checkpoint_chunks = checkpoint_chunks

    def forward(
        self,
        windows: torch.Tensor,
        features: torch.Tensor,
        phase_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # windows: (B, num_phases, K, window_length) = (B, 3, K, 8192)
        # features: (B, num_phases, K, feature_dim) = (B, 3, K, 58)
        B, num_phases, K, L = windows.shape

        # Encode each phase independently
        # Reshape to (B * num_phases, K, 8192) and (B * num_phases, K, 58)
        windows_flat = windows.reshape(B * num_phases, K, L)
        features_flat = features.reshape(B * num_phases, K, -1)

        # WindowEncoder: (B*P, K, 8192/58) -> (B*P, K, 128)
        # Encode in bounded chunks to keep activation memory flat on small GPUs.
        # Gradients flow through the full encoder exactly as without chunking.
        n_flat = windows_flat.shape[0]
        if self.max_encode_chunk is None or self.max_encode_chunk >= n_flat:
            encoded = self.encoder(windows_flat, features_flat)
        else:
            chunks = []
            for s in range(0, n_flat, self.max_encode_chunk):
                if self.checkpoint_chunks:
                    e = checkpoint(
                        lambda w, f: self.encoder(w, f),
                        windows_flat[s : s + self.max_encode_chunk],
                        features_flat[s : s + self.max_encode_chunk],
                        use_reentrant=False,
                    )
                else:
                    e = self.encoder(windows_flat[s : s + self.max_encode_chunk],
                                     features_flat[s : s + self.max_encode_chunk])
                chunks.append(e)
            encoded = torch.cat(chunks, dim=0)

        # MILAggregator: (B*P, K, 128) -> (B*P, 128)
        aggregated = self.aggregator(encoded)

        # Reshape back to (B, num_phases, 128)
        aggregated = aggregated.reshape(B, num_phases, -1)

        # CyclicPhaseModule: (B, 3, 128) -> (B, 3, 128)
        if phase_mask is not None:
            interacted = self.cyclic(aggregated, mask=phase_mask)
            interacted = interacted * phase_mask.unsqueeze(-1).float()
        else:
            interacted = self.cyclic(aggregated)

        # PhaseClassifier: per-phase logits
        # Flatten to (B*3, 128), classify, then reshape
        interacted_flat = interacted.reshape(B * num_phases, -1)
        logits_flat = self.classifier(interacted_flat)  # (B*3, 1)
        phase_logits = logits_flat.reshape(B, num_phases)  # (B, 3)

        # Apply phase mask to zero out missing phase logits
        if phase_mask is not None:
            phase_logits = phase_logits * phase_mask.float()

        # Phase probabilities and measurement probability via noisy-OR
        phase_probs = torch.sigmoid(phase_logits)
        measurement_prob = noisy_or_probs(phase_probs, phase_mask)

        return phase_logits, measurement_prob

    def configure_optimizers(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> torch.optim.AdamW:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

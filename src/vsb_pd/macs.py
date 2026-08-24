"""Manual MACs/FLOPs counting for Conv1d/Linear modules."""

from __future__ import annotations

import torch
import torch.nn as nn


def count_macs(model: nn.Module, *inputs: torch.Tensor) -> float:
    """Count multiply-accumulate operations for one forward pass.

    Counts only nn.Conv1d and nn.Linear. Norms, activations, pooling, and
    the robust-normalization quantiles are excluded (negligible MACs by
    comparison); FLOPs = 2 * MACs.
    """
    macs = 0.0

    def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv1d):
            batch = output.shape[0]
            positions = output.shape[-1]
            per_output = (module.in_channels // module.groups) * module.kernel_size[0]
            macs += batch * positions * module.out_channels * per_output
        elif isinstance(module, nn.Linear):
            num_rows = input[0].shape[:-1].numel()
            macs += num_rows * module.in_features * module.out_features

    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return macs


def count_flops(model: nn.Module, *inputs: torch.Tensor) -> float:
    """Return FLOPs estimated as 2 x MACs for the same forward pass."""
    return 2.0 * count_macs(model, *inputs)

# -*- coding: utf-8 -*-
"""Smoke tests for DL baseline encoders."""
import pytest
import torch

from vsb_pd.dl_encoders import ResNet1DEncoder, TCNEncoder, InceptionTimeEncoder


@pytest.mark.parametrize("cls", [ResNet1DEncoder, TCNEncoder, InceptionTimeEncoder])
def test_dl_encoder_forward_shape(cls):
    enc = cls(8192, 128)
    x = torch.randn(2, 3, 8192)  # (B, K, L)
    out = enc(x)
    assert out.shape == (2, 3, 128)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cls", [ResNet1DEncoder, TCNEncoder, InceptionTimeEncoder])
def test_dl_encoder_parameter_budget(cls):
    enc = cls(8192, 128)
    n = sum(p.numel() for p in enc.parameters())
    assert 10_000 < n < 1_000_000, f"unexpected parameter count: {n}"

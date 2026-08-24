"""Tests for dual-branch window encoder."""

import torch
import pytest
from vsb_pd.encoder import WindowEncoder
from vsb_pd.encoder import RobustNormalize
from vsb_pd.dl_encoders import TimWindowEncoder


def test_robust_normalize_matches_window_encoder():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8192) * torch.tensor([1.0, 10.0, 100.0]).view(1, 3, 1)
    rn = RobustNormalize(8192)
    we = WindowEncoder(8192, 58, 128, branch="cnn")
    expected = we._robust_normalize(x.reshape(-1, 8192)).reshape(2, 3, 8192)
    assert torch.allclose(rn(x), expected, atol=1e-6)


def test_window_encoder_default_behavior_unchanged():
    torch.manual_seed(0)
    w = torch.randn(2, 3, 8192)
    f = torch.randn(2, 3, 58)
    we = WindowEncoder(8192, 58, 128, branch="cnn")
    out = we(w, f)
    assert out.shape == (2, 3, 128)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["cnn", "simple_cnn", "resnet1d", "inceptiontime"])
def test_tim_encoder_forward_shape(name):
    enc = TimWindowEncoder(name)
    w = torch.randn(2, 3, 8192)
    f = torch.randn(2, 3, 58)
    out = enc(w, f)
    assert out.shape == (2, 3, 128)
    assert torch.isfinite(out).all()


def test_tim_cnn_equals_window_encoder_default():
    torch.manual_seed(0)
    w = torch.randn(2, 3, 8192)
    f = torch.randn(2, 3, 58)
    torch.manual_seed(123)  # same init stream for both models
    tim = TimWindowEncoder("cnn")
    torch.manual_seed(123)
    we = WindowEncoder(8192, 58, 128, branch="cnn")
    assert torch.allclose(tim(w, f), we(w, f), atol=1e-6)


@pytest.mark.parametrize("name", ["cnn", "simple_cnn", "resnet1d", "inceptiontime"])
def test_tim_encoder_parameter_budget(name):
    enc = TimWindowEncoder(name)
    n = sum(p.numel() for p in enc.parameters())
    assert 10_000 < n < 1_000_000, f"unexpected parameter count: {n}"


def test_tim_encoder_rejects_unknown_name():
    with pytest.raises(ValueError):
        TimWindowEncoder("tcn")


def test_encoder_output_shape():
    encoder = WindowEncoder(window_length=8192, feature_dim=58, hidden_dim=128)
    B, K = 4, 8
    windows = torch.randn(B, K, 8192)
    features = torch.randn(B, K, 58)
    out = encoder(windows, features)
    assert out.shape == (B, K, 128)


def test_encoder_parameter_count_under_300k():
    encoder = WindowEncoder(window_length=8192, feature_dim=58, hidden_dim=128)
    count = sum(p.numel() for p in encoder.parameters())
    assert count < 300_000


def test_encoder_is_deterministic():
    torch.manual_seed(42)
    enc1 = WindowEncoder(window_length=8192, feature_dim=58, hidden_dim=128)
    enc1.eval()
    x = torch.randn(2, 4, 8192)
    f = torch.randn(2, 4, 58)
    out1 = enc1(x, f)
    torch.manual_seed(42)
    enc2 = WindowEncoder(window_length=8192, feature_dim=58, hidden_dim=128)
    enc2.eval()
    out2 = enc2(x, f)
    assert torch.allclose(out1, out2)


def test_encoder_handles_single_window():
    encoder = WindowEncoder(window_length=8192, feature_dim=58, hidden_dim=128)
    windows = torch.randn(1, 1, 8192)
    features = torch.randn(1, 1, 58)
    out = encoder(windows, features)
    assert out.shape == (1, 1, 128)
    assert not torch.isnan(out).any()


def test_encoder_gradient_flows():
    encoder = WindowEncoder(window_length=8192, feature_dim=58, hidden_dim=128)
    windows = torch.randn(2, 4, 8192, requires_grad=False)
    features = torch.randn(2, 4, 58, requires_grad=True)
    out = encoder(windows, features)
    loss = out.sum()
    loss.backward()
    assert features.grad is not None
    assert not torch.isnan(features.grad).any()


def test_encoder_internal_normalization_handles_outliers():
    encoder = WindowEncoder(window_length=8192, feature_dim=58, hidden_dim=128)
    windows = torch.randn(2, 4, 8192)
    windows[:, 0, :] = windows[:, 0, :] * 100 + 50  # large offset and scale
    features = torch.randn(2, 4, 58)
    out = encoder(windows, features)
    assert not torch.isnan(out).any()
    assert torch.isfinite(out).all()

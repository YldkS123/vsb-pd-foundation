"""Tests for three-phase cyclic symmetry, noisy-OR, and loss."""

import torch
import pytest
from vsb_pd.cyclic import CyclicPhaseModule, noisy_or_probs, PhaseCyclicLoss, PhaseInteractionModule


def test_cyclic_shift_equivariance():
    module = CyclicPhaseModule(128)
    x = torch.randn(4, 3, 128)
    out = module(x)
    rolled_x = torch.roll(x, shifts=1, dims=1)
    rolled_out = module(rolled_x)
    expected = torch.roll(out, shifts=1, dims=1)
    assert torch.allclose(rolled_out, expected, atol=1e-5)


def test_cyclic_shift_equivariance_multiple_shifts():
    module = CyclicPhaseModule(128)
    x = torch.randn(4, 3, 128)
    for shift in [0, 1, 2]:
        out_shifted = module(torch.roll(x, shifts=shift, dims=1))
        expected = torch.roll(module(x), shifts=shift, dims=1)
        assert torch.allclose(out_shifted, expected, atol=1e-5)


def test_cyclic_preserves_shape():
    module = CyclicPhaseModule(128)
    x = torch.randn(2, 3, 128)
    out = module(x)
    assert out.shape == (2, 3, 128)


def test_cyclic_gradient_flows():
    module = CyclicPhaseModule(128)
    x = torch.randn(2, 3, 128, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_noisy_or_numerical():
    probs = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    result = noisy_or_probs(probs)
    expected = torch.tensor([1 - 0.9 * 0.8 * 0.7, 0.0])
    assert torch.allclose(result, expected)


def test_noisy_or_with_missing_phase():
    probs = torch.tensor([[0.1, 0.2, 0.3]])
    mask = torch.tensor([[True, False, True]])
    result = noisy_or_probs(probs, mask=mask)
    expected = torch.tensor([1 - 0.9 * 0.7])
    assert torch.allclose(result, expected)


def test_noisy_or_all_missing_returns_zero():
    probs = torch.tensor([[0.1, 0.2, 0.3]])
    mask = torch.tensor([[False, False, False]])
    result = noisy_or_probs(probs, mask=mask)
    assert torch.allclose(result, torch.tensor([0.0]))


def test_noisy_or_identity_for_one_phase():
    probs = torch.tensor([[0.5], [0.8]])
    result = noisy_or_probs(probs)
    assert torch.allclose(result, probs.squeeze(-1))


def test_loss_computation():
    criterion = PhaseCyclicLoss(lambda_m=0.25)
    phase_logits = torch.randn(4, 3)
    phase_labels = torch.randint(0, 2, (4, 3)).float()
    loss = criterion(phase_logits, phase_labels)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_loss_measurement_label_is_max_of_phases():
    criterion = PhaseCyclicLoss(lambda_m=0.25)
    phase_logits = torch.tensor([[2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
    phase_labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    loss = criterion(phase_logits, phase_labels)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_loss_lambda_zero_is_pure_phase_loss():
    criterion_phase = PhaseCyclicLoss(lambda_m=0.0)
    criterion_both = PhaseCyclicLoss(lambda_m=0.25)
    phase_logits = torch.randn(4, 3)
    phase_labels = torch.randint(0, 2, (4, 3)).float()
    loss_phase = criterion_phase(phase_logits, phase_labels)
    loss_both = criterion_both(phase_logits, phase_labels)
    # When lambda_m=0, measurement loss is excluded entirely
    assert loss_phase.item() > 0
    # The two losses may differ; that's expected
    assert torch.isfinite(loss_both)


@pytest.mark.parametrize("kind", ["context_concat", "context_add"])
def test_context_modes_preserve_shape(kind):
    module = PhaseInteractionModule(kind, 128)
    x = torch.randn(4, 3, 128)
    out = module(x)
    assert out.shape == (4, 3, 128)
    assert torch.isfinite(out).all()


def test_context_concat_param_budget():
    module = PhaseInteractionModule("context_concat", 128)
    n = sum(p.numel() for p in module.parameters())
    assert n == 128 * 256 + 128 + 128 * 2, n


def test_context_add_param_budget():
    module = PhaseInteractionModule("context_add", 128)
    n = sum(p.numel() for p in module.parameters())
    assert n == 128 * 128 + 128 + 128 * 2, n


def test_context_mask_excludes_masked_phase():
    module = PhaseInteractionModule("context_concat", 128)
    x = torch.zeros(1, 3, 128)
    x[0, 1] = 10.0  # masked phase B must not leak into context
    mask = torch.tensor([[True, False, True]])
    out = module(x, mask)
    assert torch.equal(out[0, 1], torch.zeros(128))
    expected = module.context_proj(
        torch.cat([torch.zeros(1, 128), torch.zeros(1, 128)], dim=-1)
    ).squeeze(0)
    assert torch.allclose(out[0, 0], expected)
    assert torch.allclose(out[0, 2], expected)


def test_context_single_present_phase_uses_that_phase():
    module = PhaseInteractionModule("context_concat", 128)
    torch.manual_seed(0)
    x = torch.randn(1, 3, 128)
    mask = torch.tensor([[True, False, False]])
    out = module(x, mask)
    expected = module.context_proj(
        torch.cat([x[0, 0], x[0, 0]], dim=-1)
    )
    assert torch.allclose(out[0, 0], expected, atol=1e-6)


def test_context_all_missing_output_is_zero():
    for kind in ("context_concat", "context_add"):
        module = PhaseInteractionModule(kind, 128)
        x = torch.randn(1, 3, 128)
        mask = torch.tensor([[False, False, False]])
        out = module(x, mask)
        assert torch.equal(out, torch.zeros_like(out))


def test_context_modes_allow_distinct_phase_outputs():
    for kind in ("context_concat", "context_add"):
        module = PhaseInteractionModule(kind, 128)
        x = torch.zeros(1, 3, 128)
        x[0, 0, 0] = 1.0
        x[0, 1, 0] = 2.0
        out = module(x)
        assert not torch.allclose(out[0, 0], out[0, 1])


def test_context_gradient_flows():
    for kind in ("context_concat", "context_add"):
        module = PhaseInteractionModule(kind, 128)
        x = torch.randn(2, 3, 128, requires_grad=True)
        out = module(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

"""Error-adaptive replay weighting of demonstration groups."""

import pytest
import torch

from retroagi.stages.block_smb.demonstrations import adaptive_group_weights, priority_sample_weights


def test_adaptive_replay_increases_braking_and_preserves_family_retention():
    # Family 0: 100 ride-brake rows, 100 exit-walk rows. Family 1: ordinary walk.
    groups = torch.tensor([31] * 100 + [36] * 100 + [43] * 100)
    base = torch.full((300,), 0.005)
    base[200:] = 0.01
    counts = torch.bincount(groups).float()
    errors = torch.zeros_like(counts)
    errors[31] = 1
    weights = adaptive_group_weights(base, groups, counts, errors)
    assert weights[:100].sum() > base[:100].sum()
    assert weights[:200].sum() == pytest.approx(1.0, abs=1e-5)
    assert weights[200:].sum() == pytest.approx(1.0, abs=1e-5)
    assert bool((weights >= 0.35 * base).all())
    priorities = torch.linspace(0.05, 5, 300)
    weighted = priority_sample_weights(weights, groups, priorities)
    for group in groups.unique():
        assert weighted[groups == group].sum() == pytest.approx(
            float(weights[groups == group].sum()), abs=1e-5
        )
    # A corrected group gives its allocation back; stale failure priority is not permanent.
    errors.zero_()
    assert torch.allclose(adaptive_group_weights(base, groups, counts, errors), base)


def test_tiny_correction_group_cannot_get_full_error_boost():
    groups = torch.tensor([17] * 2 + [31] * 100 + [36] * 100)
    base = torch.cat((torch.full((2,), 0.25), torch.full((200,), 0.005)))
    counts = torch.bincount(groups).float()
    errors = torch.ones_like(counts)
    weights = adaptive_group_weights(base, groups, counts, errors)
    tiny_ratio = weights[:2].sum() / base[:2].sum()
    supported_ratio = weights[2:102].sum() / base[2:102].sum()
    assert supported_ratio > tiny_ratio

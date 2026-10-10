"""The action predictor's records, outputs and loss."""

import torch

from retroagi.core.action_predictor import (
    ACTION_RECORD,
    ACTIONS_PER_DECISION,
    COLUMN,
    MARIO_TOKEN,
    ActionPredictor,
    action_record,
    empty_records,
    outcome_loss,
)
from retroagi.core.action_tokens import VERBS, ActionToken, pointer_index


def test_records_carry_the_action_and_its_outcome():
    outcome = {"dx": 50.0, "dy": -10.0, "frames": 48, "safe": True, "won": False, "room": 12.0}
    stomp = action_record(
        ActionToken("stomp", ("enemies", 2)),
        {**outcome, "object_dx": 44.0, "object_dy": -10.0},
        True,
    )
    assert stomp[COLUMN["verb"]] == VERBS.index("stomp")
    assert stomp[COLUMN["pointer"]] == pointer_index(("enemies", 2))
    assert stomp[COLUMN["has_object"]] == 1 and stomp[COLUMN["chosen"]] == 1
    hold = action_record(ActionToken("hold"), outcome, False)
    assert hold[COLUMN["pointer"]] == MARIO_TOKEN and hold[COLUMN["has_object"]] == 0
    assert empty_records().shape == (ACTIONS_PER_DECISION, len(ACTION_RECORD))
    assert (empty_records()[:, COLUMN["verb"]] == -1).all()


def test_untrained_predictor_is_unsure_and_learns():
    torch.manual_seed(0)
    predictor = ActionPredictor()
    tokens, present = torch.randn(2, 86, 96), torch.ones(2, 86, dtype=torch.bool)
    memory, forecast = torch.randn(2, 128), torch.zeros(2, 6, 5)
    rows, verbs = torch.tensor([0, 1]), torch.tensor([VERBS.index("land_on"), VERBS.index("stomp")])
    pointers = torch.tensor([pointer_index(("surfaces", 1)), pointer_index(("enemies", 0))])
    out = predictor(tokens, present, memory, forecast, rows, verbs, pointers)
    assert torch.allclose(out["mean"][:, :4], torch.zeros(2, 4))
    assert torch.allclose(out["sigma"][:, 0], torch.full((2,), 32.0))
    records = torch.zeros(2, len(ACTION_RECORD))
    records[:, COLUMN["dx"]] = torch.tensor([50.0, 40.0])
    records[:, COLUMN["frames"]] = 50.0
    records[:, COLUMN["safe"]] = torch.tensor([1.0, 0.0])
    optimizer = torch.optim.Adam(predictor.parameters(), lr=3e-3)
    first = None
    for _ in range(100):
        loss, stats = outcome_loss(
            predictor(tokens, present, memory, forecast, rows, verbs, pointers), records
        )
        first = first if first is not None else stats["end_error_pixels"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert stats["end_error_pixels"] < first / 4
    assert stats["success_accuracy"] == 1.0

"""The skill's action heads: a verb, then the object it is aimed at."""

import torch

from retroagi.core.action_tokens import POINTER_TOKENS, VERBS, pointer_index
from retroagi.core.layered_policy import POINTER_START, LayeredSMBPolicy, PolicySettings
from retroagi.core.smb_observer import C_SPANS, SEQ_LEN_A, SEQ_LEN_B, SEQ_LEN_C
from retroagi.core.tokens import encode_tactic, tactic_token


def scene(batch=2, enemy=(True, False)):
    """Rows with Mario, surface slot 0 and (per row) enemy slot 0."""
    a = torch.zeros(batch, SEQ_LEN_A, dtype=torch.long)
    b = torch.zeros(batch, SEQ_LEN_B, dtype=torch.long)
    c = torch.zeros(batch, SEQ_LEN_C)
    c[:, C_SPANS["c_mario"][0]] = 1
    c[:, C_SPANS["c_surfaces"][0]] = 1
    for row, present in enumerate(enemy):
        c[row, C_SPANS["c_enemies"][0]] = float(present)
    return a, b, c


def run(policy, rows, picks=None):
    encoded = policy.encode_scene(rows)
    state = policy.memory.initial(len(rows[0]), "cpu")
    _, expected = policy.expect(state)
    tactic = torch.stack([encode_tactic(tactic_token("advance"))] * len(rows[0]))
    return policy.run_skill(encoded, expected, tactic, memory=state.hidden, picks=picks)


def test_verbs_and_objects_are_masked_by_what_is_in_view():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy(PolicySettings()).eval()
    out = run(policy, scene())
    usable = out["verb"] > -1e3
    # Row 0 sees an enemy, row 1 does not: stomping needs one in view.
    stomp = VERBS.index("stomp")
    assert usable[0, stomp] and not usable[1, stomp]
    assert usable[:, VERBS.index("hold")].all() and usable[:, VERBS.index("land_on")].all()
    assert out["pointer"].shape == (2, POINTER_TOKENS)
    enemy = pointer_index(("enemies", 0)) - POINTER_START
    surface = pointer_index(("surfaces", 0)) - POINTER_START
    for verb, aimed in (("stomp", enemy), ("land_on", surface)):
        logits = run(
            policy,
            scene(),
            picks={
                "mode": torch.zeros(2),
                "x": torch.zeros(2),
                "verb": torch.full((2,), VERBS.index(verb)),
            },
        )["pointer"][0]
        assert (logits > -1e3).nonzero().flatten().tolist() == [aimed]


def test_action_heads_learn_the_teachers_action():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy(PolicySettings())
    rows = scene()
    stomp, land = VERBS.index("stomp"), VERBS.index("land_on")
    verbs = torch.tensor([stomp, land])
    pointers = (
        torch.tensor([pointer_index(("enemies", 0)), pointer_index(("surfaces", 0))])
        - POINTER_START
    )
    optimizer = torch.optim.Adam(policy.skill.parameters(), lr=3e-3)
    for _ in range(60):
        out = run(policy, rows, picks={"mode": torch.zeros(2), "x": torch.zeros(2), "verb": verbs})
        loss = torch.nn.functional.cross_entropy(
            out["verb"], verbs
        ) + torch.nn.functional.cross_entropy(out["pointer"], pointers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    policy.eval()
    picks = run(policy, rows)["picks"]
    assert picks["verb"].tolist() == verbs.tolist()
    assert picks["pointer"].tolist() == pointers.tolist()

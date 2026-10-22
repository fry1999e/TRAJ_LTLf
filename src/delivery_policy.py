"""The released workflow never substitutes automatic delivery for review."""


def resolve_delivery_policy(policy, review_mode, experiment_mode):
    if policy not in {None, "strict"}:
        raise ValueError("Only strict delivery is supported by the human-review release")
    if experiment_mode != "human":
        raise ValueError("Only human experiment mode is supported")
    return "strict"

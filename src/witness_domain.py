"""Query-count-independent semantic witness-domain planning."""

from __future__ import annotations

from dataclasses import dataclass

from contract_versions import WITNESS_DOMAIN_POLICY_VERSION
from ir_witness_solver import HorizonPolicy
from semantic_node_contracts import discriminator_horizon, structural_horizon


@dataclass(frozen=True)
class WitnessDomainSpec:
    minimum_evaluation_horizon: int
    minimum_positive_witness_horizon: int
    minimum_negative_witness_horizon: int
    discriminator_horizon: int
    minimum_length: int
    maximum_length: int
    exact_length: int | None
    lengths: tuple[int, ...]
    basis: tuple[str, ...]
    policy_version: str = WITNESS_DOMAIN_POLICY_VERSION


def plan_witness_domain(
    ir: object,
    *,
    exact_length: int | None = None,
    maximum_extension: int = 1,
) -> WitnessDomainSpec:
    if maximum_extension < 0:
        raise ValueError("witness domain maximum_extension must be non-negative")
    structural = max(1, structural_horizon(ir))
    root = getattr(ir, "node_map")[getattr(ir, "root_node_id")]
    discriminator = max(1, discriminator_horizon(root))
    minimum = max(structural, discriminator)
    if exact_length is not None:
        if exact_length < minimum:
            raise ValueError(
                "explicit witness length is below the IR semantic horizon"
            )
        maximum = exact_length
        minimum = exact_length
    else:
        maximum = minimum + maximum_extension
    lengths = (
        (exact_length,)
        if exact_length is not None
        else tuple(range(minimum, maximum + 1))
    )
    return WitnessDomainSpec(
        minimum_evaluation_horizon=structural,
        minimum_positive_witness_horizon=structural,
        minimum_negative_witness_horizon=structural,
        discriminator_horizon=discriminator,
        minimum_length=minimum,
        maximum_length=maximum,
        exact_length=exact_length,
        lengths=lengths,
        basis=(
            "semantic_node_contract_structural_horizon",
            "root_discriminator_horizon",
            *(
                ("explicit_exact_trace_length",)
                if exact_length is not None
                else ("bounded_semantic_extension",)
            ),
        ),
    )


def horizon_policy_for_domain(spec: WitnessDomainSpec) -> HorizonPolicy:
    return HorizonPolicy(
        minimum_length=spec.minimum_length,
        maximum_length=spec.maximum_length,
        exact_length=spec.exact_length,
    )


__all__ = [
    "WitnessDomainSpec",
    "horizon_policy_for_domain",
    "plan_witness_domain",
]

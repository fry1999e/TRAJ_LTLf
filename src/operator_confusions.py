"""General, answer-free operator-confusion catalog for runtime.

The catalog describes *how* two semantic shapes differ and what a minimal
discriminating trace must exercise.  It deliberately does not encode which
alternative is correct for any requirement; that choice remains bound to
approved natural-language semantics and independent trace annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Iterator


OPERATOR_CONFUSION_CATALOG_VERSION = "operator-confusions-v2"

TRACE_DISCRIMINATION = "trace_discrimination"
STRUCTURAL_AUDIT = "structural_audit"
_EVIDENCE_MODES = frozenset({TRACE_DISCRIMINATION, STRUCTURAL_AUDIT})


def _normalize_key(value: object) -> str:
    normalized = str(value).strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def _stable_strings(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class OperatorConfusion:
    """One reusable ambiguity family, independent of a requirement answer."""

    confusion_id: str
    left_shape: str
    right_shape: str
    semantic_difference: str
    construction_constraints: tuple[str, ...]
    minimum_horizon: int
    ir_node_kinds: tuple[str, ...]
    ambiguity_axes: tuple[str, ...]
    expected_distinguishing_outcome: str = (
        "the alternatives evaluate differently; approved NL review supplies the label"
    )
    evidence_mode: str = TRACE_DISCRIMINATION
    required_trace_role: str = "boundary"
    catalog_version: str = OPERATOR_CONFUSION_CATALOG_VERSION

    def __post_init__(self) -> None:
        confusion_id = _normalize_key(self.confusion_id)
        if not confusion_id:
            raise ValueError("confusion_id must not be empty")
        object.__setattr__(self, "confusion_id", confusion_id)
        object.__setattr__(self, "left_shape", self.left_shape.strip())
        object.__setattr__(self, "right_shape", self.right_shape.strip())
        object.__setattr__(
            self,
            "semantic_difference",
            self.semantic_difference.strip(),
        )
        object.__setattr__(
            self,
            "construction_constraints",
            _stable_strings(self.construction_constraints),
        )
        object.__setattr__(
            self,
            "ir_node_kinds",
            tuple(
                dict.fromkeys(
                    _normalize_key(value) for value in self.ir_node_kinds if str(value).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "ambiguity_axes",
            tuple(
                dict.fromkeys(
                    _normalize_key(value) for value in self.ambiguity_axes if str(value).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "expected_distinguishing_outcome",
            self.expected_distinguishing_outcome.strip(),
        )
        object.__setattr__(self, "evidence_mode", _normalize_key(self.evidence_mode))
        object.__setattr__(
            self,
            "required_trace_role",
            _normalize_key(self.required_trace_role),
        )
        object.__setattr__(self, "catalog_version", self.catalog_version.strip())

        if not self.left_shape or not self.right_shape:
            raise ValueError("both confusion shapes must be non-empty")
        if not self.semantic_difference:
            raise ValueError("semantic_difference must not be empty")
        if not self.construction_constraints:
            raise ValueError("construction_constraints must not be empty")
        if self.minimum_horizon < 1:
            raise ValueError("minimum_horizon must be positive")
        if not self.ir_node_kinds:
            raise ValueError("ir_node_kinds must not be empty")
        if not self.ambiguity_axes:
            raise ValueError("ambiguity_axes must not be empty")
        if self.evidence_mode not in _EVIDENCE_MODES:
            raise ValueError("unsupported evidence_mode: " + self.evidence_mode)
        if not self.expected_distinguishing_outcome:
            raise ValueError("expected_distinguishing_outcome must not be empty")
        if not self.catalog_version:
            raise ValueError("catalog_version must not be empty")

    @property
    def confusion_pair(self) -> str:
        return f"{self.left_shape} <> {self.right_shape}"

    @property
    def trace_distinguishable(self) -> bool:
        return self.evidence_mode == TRACE_DISCRIMINATION

    def matches_ir_node_kind(self, node_kind: object) -> bool:
        return _normalize_key(_read_node_kind(node_kind)) in self.ir_node_kinds

    def matches_ambiguity_axis(self, ambiguity_axis: object) -> bool:
        return _normalize_key(ambiguity_axis) in self.ambiguity_axes

    def to_trace_obligation(
        self,
        target_ir_node_id: str,
        *,
        obligation_id: str = "",
        ambiguity_axis: str = "",
        minimum_horizon: int | None = None,
        semantic_parameters: Iterable[tuple[str, Any]] = (),
    ):
        """Adapt this entry to the existing ``schemas.TraceObligation`` shape.

        Structural-only catalog entries retain ``expected_relation`` as
        ``structural_audit`` so current trace binders cannot accidentally claim
        that a finite trace distinguished a truth-functionally equivalent pair.
        """

        from schemas import TraceObligation

        node_id = str(target_ir_node_id).strip()
        if not node_id:
            raise ValueError("target_ir_node_id must not be empty")
        axis = _normalize_key(ambiguity_axis) if ambiguity_axis else self.ambiguity_axes[0]
        if axis not in self.ambiguity_axes:
            raise ValueError(
                f"ambiguity axis {axis!r} is not declared by {self.confusion_id}"
            )
        horizon = self.minimum_horizon if minimum_horizon is None else int(minimum_horizon)
        if horizon < self.minimum_horizon:
            raise ValueError(
                f"minimum_horizon cannot weaken catalog floor {self.minimum_horizon}"
            )
        relation = "distinguish" if self.trace_distinguishable else STRUCTURAL_AUDIT
        role = self.required_trace_role if self.trace_distinguishable else "structural_review"
        parameters = (
            ("confusion_id", self.confusion_id),
            ("confusion_pair", self.confusion_pair),
            ("target_ambiguity_axis", axis),
            ("evidence_mode", self.evidence_mode),
            *tuple(semantic_parameters),
        )
        return TraceObligation(
            obligation_id=(
                str(obligation_id).strip()
                or f"obl-{node_id}-{self.confusion_id}"
            ),
            ir_node_ids=(node_id,),
            required_trace_role=role,
            construction_constraints=(
                f"target ambiguity axis {axis}",
                f"confusion pair {self.confusion_pair}",
                *self.construction_constraints,
                "candidate formulas must not determine the adjudicated label",
            ),
            expected_relation=relation,
            minimum_horizon=horizon,
            boundary_case=axis,
            semantic_parameters=parameters,
            ambiguity_axis_ids=(self.confusion_id, axis),
            evidence_mode=(
                "explicit_discriminator"
                if self.trace_distinguishable
                else "diagnostic_only"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible catalog record."""

        return {
            "confusion_id": self.confusion_id,
            "confusion_pair": self.confusion_pair,
            "left_shape": self.left_shape,
            "right_shape": self.right_shape,
            "semantic_difference": self.semantic_difference,
            "construction_constraints": list(self.construction_constraints),
            "minimum_horizon": self.minimum_horizon,
            "ir_node_kinds": list(self.ir_node_kinds),
            "ambiguity_axes": list(self.ambiguity_axes),
            "expected_distinguishing_outcome": self.expected_distinguishing_outcome,
            "evidence_mode": self.evidence_mode,
            "required_trace_role": self.required_trace_role,
            "catalog_version": self.catalog_version,
        }


class OperatorConfusionCatalog:
    """Immutable, deterministic index of operator-confusion specifications."""

    def __init__(self, entries: Iterable[OperatorConfusion] = ()) -> None:
        normalized = tuple(entries)
        if any(not isinstance(entry, OperatorConfusion) for entry in normalized):
            raise TypeError("catalog entries must be OperatorConfusion instances")
        identifiers = [entry.confusion_id for entry in normalized]
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        if duplicates:
            raise ValueError(
                "duplicate operator confusion ids: " + ", ".join(duplicates)
            )
        self._entries = normalized
        self._by_id = {entry.confusion_id: entry for entry in normalized}

    def __iter__(self) -> Iterator[OperatorConfusion]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> tuple[OperatorConfusion, ...]:
        return self._entries

    def get(self, confusion_id: str) -> OperatorConfusion:
        normalized = _normalize_key(confusion_id)
        try:
            return self._by_id[normalized]
        except KeyError as exc:
            raise KeyError(f"unknown operator confusion: {confusion_id}") from exc

    def for_ir_node_kind(self, node_kind: object) -> tuple[OperatorConfusion, ...]:
        normalized = _normalize_key(_read_node_kind(node_kind))
        return tuple(
            entry for entry in self._entries if normalized in entry.ir_node_kinds
        )

    def for_ambiguity_axis(self, ambiguity_axis: object) -> tuple[OperatorConfusion, ...]:
        normalized = _normalize_key(ambiguity_axis)
        return tuple(
            entry for entry in self._entries if normalized in entry.ambiguity_axes
        )

    def query(
        self,
        *,
        ir_node_kind: object | None = None,
        ambiguity_axis: object | None = None,
    ) -> tuple[OperatorConfusion, ...]:
        """Return the stable intersection of optional node/axis filters."""

        node_key = (
            _normalize_key(_read_node_kind(ir_node_kind))
            if ir_node_kind is not None
            else ""
        )
        axis_key = _normalize_key(ambiguity_axis) if ambiguity_axis is not None else ""
        return tuple(
            entry
            for entry in self._entries
            if (not node_key or node_key in entry.ir_node_kinds)
            and (not axis_key or axis_key in entry.ambiguity_axes)
        )

    def extend(
        self,
        entries: Iterable[OperatorConfusion],
    ) -> "OperatorConfusionCatalog":
        """Return a new catalog; the current stable catalog is never mutated."""

        return OperatorConfusionCatalog((*self._entries, *tuple(entries)))

    def trace_obligations_for(
        self,
        target_ir_node_id: str,
        *,
        ir_node_kind: object | None = None,
        ambiguity_axis: object | None = None,
    ) -> tuple[Any, ...]:
        return tuple(
            entry.to_trace_obligation(
                target_ir_node_id,
                ambiguity_axis=(
                    str(ambiguity_axis) if ambiguity_axis is not None else ""
                ),
            )
            for entry in self.query(
                ir_node_kind=ir_node_kind,
                ambiguity_axis=ambiguity_axis,
            )
        )


def _read_node_kind(value: object) -> str:
    return str(getattr(value, "node_type", value) or "")


_DEFAULT_CONFUSIONS = (
    OperatorConfusion(
        confusion_id="current_vs_global_scope",
        left_shape="phi",
        right_shape="G phi",
        semantic_difference=(
            "The left shape constrains the current position only; the right shape "
            "constrains every position of the finite trace."
        ),
        construction_constraints=(
            "use at least two positions",
            "make phi agree at the initial position",
            "flip phi at a later position so only the all-position scope observes it",
        ),
        minimum_horizon=2,
        ir_node_kinds=("ap_condition", "global_constraint", "persist_to_end"),
        ambiguity_axes=(
            "current_vs_global_scope",
            "local_vs_global_scope",
            "phi_vs_globally_phi",
        ),
    ),
    OperatorConfusion(
        confusion_id="eventually_vs_next",
        left_shape="F phi",
        right_shape="X phi",
        semantic_difference=(
            "Eventually permits a witness at the current or any later position; "
            "strong next inspects exactly the immediate successor."
        ),
        construction_constraints=(
            "use a nonempty finite suffix after the current position",
            "place the only phi witness either now or later than the immediate successor",
            "keep phi false at the immediate successor in the delayed-witness variant",
        ),
        minimum_horizon=3,
        ir_node_kinds=("eventually", "next_step", "eventually_strict", "strict_future", "current_or_future", "bounded_window"),
        ambiguity_axes=("eventually_vs_next", "f_vs_x", "eventually_vs_strict_next"),
    ),
    OperatorConfusion(
        confusion_id="strict_eventually_vs_next",
        left_shape="X F phi",
        right_shape="X phi",
        semantic_difference=(
            "Strict eventuality permits a witness at any later position, while "
            "exact next requires it at the immediate successor."
        ),
        construction_constraints=(
            "use at least three positions",
            "keep phi false at the immediate successor",
            "place the only phi witness later than the immediate successor",
        ),
        minimum_horizon=3,
        ir_node_kinds=("eventually_strict", "next_step", "strict_future"),
        ambiguity_axes=("strict_eventually_vs_next", "future_window_vs_exact_next"),
    ),
    OperatorConfusion(
        confusion_id="next_one_vs_next_two",
        left_shape="X phi",
        right_shape="X X phi",
        semantic_difference=(
            "The alternatives inspect phi at offsets one and two respectively, "
            "including different finite-suffix requirements."
        ),
        construction_constraints=(
            "use at least three positions",
            "assign different phi values at offsets one and two",
            "include both directions as sibling boundary queries when labels are available",
        ),
        minimum_horizon=3,
        ir_node_kinds=("next_step",),
        ambiguity_axes=("next_depth", "next_one_vs_next_two", "temporal_offset"),
    ),
    OperatorConfusion(
        confusion_id="next_depth_neighbor",
        left_shape="X^k phi",
        right_shape="X^(k-1) phi / X^(k+1) phi",
        semantic_difference=(
            "The alternatives inspect adjacent temporal offsets around the declared "
            "depth k; neither neighboring offset is privileged by the catalog."
        ),
        construction_constraints=(
            "bind k from the reviewed IR rather than from a candidate formula",
            "set trace length to at least k+2",
            "vary phi independently at offsets k-1, k, and k+1",
            "generate early, exact, and late sibling queries under one annotation lock",
        ),
        minimum_horizon=3,
        ir_node_kinds=("next_step", "fixed_duration", "ordered_sequence"),
        ambiguity_axes=(
            "next_depth",
            "temporal_offset_too_early",
            "temporal_offset_too_late",
        ),
    ),
    OperatorConfusion(
        confusion_id="next_exact_vs_early",
        left_shape="X^k phi",
        right_shape="X^(k-1) phi",
        semantic_difference="The neighbour checks one position earlier than the reviewed exact offset.",
        construction_constraints=(
            "bind k from the reviewed IR",
            "assign different phi values at offsets k-1 and k",
        ),
        minimum_horizon=2,
        ir_node_kinds=("next_step",),
        ambiguity_axes=("temporal_offset_too_early", "relative_offset"),
    ),
    OperatorConfusion(
        confusion_id="next_exact_vs_late",
        left_shape="X^k phi",
        right_shape="X^(k+1) phi",
        semantic_difference="The neighbour checks one position later than the reviewed exact offset.",
        construction_constraints=(
            "bind k from the reviewed IR",
            "assign different phi values at offsets k and k+1",
        ),
        minimum_horizon=3,
        ir_node_kinds=("next_step",),
        ambiguity_axes=("temporal_offset_too_late", "relative_offset"),
    ),
    OperatorConfusion(
        confusion_id="strong_until_vs_weak_until",
        left_shape="phi U psi",
        right_shape="weak_until(phi, psi)",
        semantic_difference=(
            "Strong until requires a psi witness; weak until also permits phi to "
            "hold through the finite suffix without any psi witness."
        ),
        construction_constraints=(
            "keep phi true at every position",
            "omit psi through the end of the trace",
            "add a companion trace with a delayed psi witness to control left-side persistence",
        ),
        minimum_horizon=2,
        ir_node_kinds=("until", "release"),
        ambiguity_axes=(
            "strong_until_vs_weak_until",
            "strong_until",
            "until_witness",
        ),
    ),
    OperatorConfusion(
        confusion_id="until_current_vs_strict_future",
        left_shape="until witness may occur now",
        right_shape="until witness must occur strictly later",
        semantic_difference=(
            "The current-or-future reading accepts a right witness at the activation "
            "state; the strict-future reading requires at least one successor."
        ),
        construction_constraints=(
            "make the right witness true at activation",
            "omit the right witness at every later position",
            "use at least two positions",
        ),
        minimum_horizon=2,
        ir_node_kinds=("until",),
        ambiguity_axes=("until_witness_timing", "current_vs_strict_future"),
    ),
    OperatorConfusion(
        confusion_id="until_left_at_witness",
        left_shape="left remains required at the witness",
        right_shape="left may stop when the witness occurs",
        semantic_difference=(
            "The alternatives differ when the right witness occurs at a state where "
            "the left condition is already false."
        ),
        construction_constraints=(
            "keep the left condition true before the witness",
            "make it false exactly at the right-witness position",
        ),
        minimum_horizon=2,
        ir_node_kinds=("until",),
        ambiguity_axes=("until_left_at_witness", "until_boundary_inclusion"),
    ),
    OperatorConfusion(
        confusion_id="current_vs_strict_future",
        left_shape="current response",
        right_shape="strict_future(response)",
        semantic_difference=(
            "The current-response reading may be witnessed now; the strict-future "
            "reading requires a witness after the current position."
        ),
        construction_constraints=(
            "make response true at the current position",
            "keep response false at every strict future position",
            "use at least one successor to avoid relying only on an empty suffix",
        ),
        minimum_horizon=2,
        ir_node_kinds=("current_or_future", "strict_future", "eventually_strict", "next_step", "eventually"),
        ambiguity_axes=(
            "current_vs_strict_future",
            "simultaneous_vs_strict_future",
            "immediate_witness_vs_strict_future",
        ),
    ),
    OperatorConfusion(
        confusion_id="bounded_window_vs_unbounded_eventually",
        left_shape="F[lower,upper] phi",
        right_shape="F phi",
        semantic_difference=(
            "The bounded reading rejects a witness after its upper bound; "
            "unbounded eventuality still accepts it."
        ),
        construction_constraints=(
            "bind lower and upper from the reviewed IR",
            "keep phi false throughout the bounded window",
            "place the only phi witness after the upper bound",
        ),
        minimum_horizon=3,
        ir_node_kinds=("bounded_window",),
        ambiguity_axes=("bounded_window_vs_unbounded_eventually", "window_upper_bound"),
    ),
    OperatorConfusion(
        confusion_id="precedence_required_vs_optional",
        left_shape="required prerequisite before constrained",
        right_shape="optional if constrained never occurs",
        semantic_difference=(
            "A required-witness reading rejects an execution with no constrained "
            "event; the conditional reading accepts that execution."
        ),
        construction_constraints=(
            "omit the constrained event for the entire trace",
            "also omit an accidental prerequisite witness",
        ),
        minimum_horizon=2,
        ir_node_kinds=("precedence", "before"),
        ambiguity_axes=("precedence_required_vs_optional", "precedence_witness_policy"),
    ),
    OperatorConfusion(
        confusion_id="precedence_strict_vs_non_strict",
        left_shape="prerequisite strictly before event",
        right_shape="prerequisite may coincide with event",
        semantic_difference=(
            "The strict reading rejects a same-state prerequisite and event; the "
            "non-strict reading accepts that state as the witness."
        ),
        construction_constraints=(
            "place prerequisite and constrained event together",
            "omit any earlier prerequisite",
        ),
        minimum_horizon=1,
        ir_node_kinds=("precedence", "before"),
        ambiguity_axes=("precedence_ordering", "same_step_policy"),
    ),
    OperatorConfusion(
        confusion_id="periodicity_terminal_policy",
        left_shape="finite periodic prefix may end before first schedule",
        right_shape="first scheduled witness is required",
        semantic_difference=(
            "The finite-prefix reading accepts a run ending before the first scheduled "
            "position; the required-first-witness reading rejects it."
        ),
        construction_constraints=(
            "target a fixed-phase periodicity node",
            "end the trace before its first scheduled phase",
            "omit off-schedule occurrences",
        ),
        minimum_horizon=1,
        ir_node_kinds=("periodicity",),
        ambiguity_axes=("periodicity_terminal", "terminal_policy"),
    ),
    OperatorConfusion(
        confusion_id="cardinality_threshold_neighbor",
        left_shape="count relation k",
        right_shape="the same relation at k+1",
        semantic_difference="The readings differ exactly at a neighbouring count threshold.",
        construction_constraints=(
            "bind mode and k from the reviewed IR",
            "construct a trace whose occurrence count lies on the neighbouring threshold",
        ),
        minimum_horizon=2,
        ir_node_kinds=("cardinality",),
        ambiguity_axes=("cardinality_threshold_neighbor", "count_threshold"),
    ),
    OperatorConfusion(
        confusion_id="local_implication_vs_global_response",
        left_shape="p -> q",
        right_shape="G(p -> q)",
        semantic_difference=(
            "The local implication is checked only at the current position; the "
            "global response checks every later occurrence of p as well."
        ),
        construction_constraints=(
            "make both alternatives agree at the initial position",
            "introduce p at a later position while q is false there",
            "include a late-trigger companion where q is true to control trigger placement",
        ),
        minimum_horizon=2,
        ir_node_kinds=("implication", "global_constraint"),
        ambiguity_axes=(
            "local_implication_vs_global_response",
            "local_vs_global_scope",
            "missing_global_scope",
            "excess_global_scope",
        ),
    ),
    OperatorConfusion(
        confusion_id="implication_disjunction_grouping",
        left_shape="(p -> q) | r",
        right_shape="p -> (q | r)",
        semantic_difference=(
            "Under the current truth-functional Boolean semantics these two shapes "
            "are equivalent (both are !p | q | r); the risk is AST parent/provenance "
            "drift, not a trace-distinguishable semantic choice."
        ),
        construction_constraints=(
            "do not claim a finite valuation distinguishes the two displayed shapes",
            "audit implication/disjunction parent ids and source-span ownership",
            "if an operand has a distinct temporal scope, catalog that scoped pair separately",
        ),
        minimum_horizon=1,
        ir_node_kinds=("implication", "disjunction"),
        ambiguity_axes=("boolean_grouping_error", "scope_parent_mismatch"),
        expected_distinguishing_outcome=(
            "no trace difference for the displayed Boolean equivalence; structural audit must preserve the reviewed AST parent"
        ),
        evidence_mode=STRUCTURAL_AUDIT,
        required_trace_role="structural_review",
    ),
    OperatorConfusion(
        confusion_id="simultaneous_vs_independent_eventuality",
        left_shape="F(a & b)",
        right_shape="F a & F b",
        semantic_difference=(
            "The simultaneous reading requires one shared witness position; the "
            "independent reading permits separate witness positions for a and b."
        ),
        construction_constraints=(
            "use at least two positions",
            "place a and b at different positions",
            "ensure no position contains both a and b",
            "add a companion with a shared witness to control mere occurrence",
        ),
        minimum_horizon=2,
        ir_node_kinds=("eventually", "conjunction"),
        ambiguity_axes=(
            "simultaneous_vs_independent_eventuality",
            "simultaneous_vs_independent",
            "boolean_grouping_error",
        ),
    ),
)


OPERATOR_CONFUSION_CATALOG = OperatorConfusionCatalog(_DEFAULT_CONFUSIONS)
# Compatibility name only; both references point to the same immutable
# registry, so there is no second active rule set.
DEFAULT_OPERATOR_CONFUSION_CATALOG = OPERATOR_CONFUSION_CATALOG


def confusions_for_ir_node_kind(node_kind: object) -> tuple[OperatorConfusion, ...]:
    return DEFAULT_OPERATOR_CONFUSION_CATALOG.for_ir_node_kind(node_kind)


def confusions_for_ambiguity_axis(ambiguity_axis: object) -> tuple[OperatorConfusion, ...]:
    return DEFAULT_OPERATOR_CONFUSION_CATALOG.for_ambiguity_axis(ambiguity_axis)


def query_operator_confusions(
    *,
    ir_node_kind: object | None = None,
    ambiguity_axis: object | None = None,
) -> tuple[OperatorConfusion, ...]:
    return DEFAULT_OPERATOR_CONFUSION_CATALOG.query(
        ir_node_kind=ir_node_kind,
        ambiguity_axis=ambiguity_axis,
    )


__all__ = [
    "DEFAULT_OPERATOR_CONFUSION_CATALOG",
    "OPERATOR_CONFUSION_CATALOG",
    "OPERATOR_CONFUSION_CATALOG_VERSION",
    "STRUCTURAL_AUDIT",
    "TRACE_DISCRIMINATION",
    "OperatorConfusion",
    "OperatorConfusionCatalog",
    "confusions_for_ambiguity_axis",
    "confusions_for_ir_node_kind",
    "query_operator_confusions",
]

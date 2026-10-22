"""Policy-driven, AST-level closure checks for complete LTLf formulas.

The parser deliberately accepts a wider language than the production output
policy.  This module keeps those two concerns separate: parsing establishes a
``Formula`` tree, while :func:`validate_formula_closure` decides whether that
tree is legal for one approved semantic-IR version.

The implementation is intentionally side-effect free so the same validation
can be used at completion, compiler, selection, approval, checkpoint, and
learning-writeback boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Iterator, Mapping

from ltlf_parser import Formula


FORMULA_CLOSURE_POLICY_VERSION = "formula-closure-v1"

DEFAULT_TEMPORAL_OPERATORS = ("F", "G", "U", "X")
DEFAULT_BOOLEAN_OPERATORS = ("and", "implies", "not", "or")
BOOLEAN_CONSTANTS = frozenset({"true", "false"})

_PLACEHOLDER_EXACT = frozenset(
    {
        "epsilon",
        "fallback",
        "fallback_ap",
        "hole",
        "placeholder",
        "unknown",
        "unknown_ap",
    }
)
_PLACEHOLDER_PREFIX = re.compile(
    r"^(?:h_|hole(?:_|$)|placeholder(?:_|$)|fallback(?:_|$)|unknown(?:_|$))",
    flags=re.IGNORECASE,
)


def _normalized_names(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(value).strip()
                for value in values
                if str(value).strip()
            }
        )
    )


def _normalized_boolean_constants(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip().lower() for value in values if str(value).strip()}))


@dataclass(frozen=True)
class FormulaClosurePolicy:
    """The complete-formula vocabulary approved for one semantic IR.

    ``required_aps`` and ``optional_aps`` must be explicit subsets of
    ``allowed_aps``.  The strict policy builder treats unannotated APs as
    optional: whether an AP must occur is a semantic fact and must not be
    guessed merely because the AP was declared.
    """

    allowed_aps: tuple[str, ...]
    required_aps: tuple[str, ...] = ()
    optional_aps: tuple[str, ...] = ()
    allowed_boolean_constants: tuple[str, ...] = ()
    allowed_temporal_operators: tuple[str, ...] = DEFAULT_TEMPORAL_OPERATORS
    allowed_boolean_operators: tuple[str, ...] = DEFAULT_BOOLEAN_OPERATORS
    allow_placeholders: bool = False
    approved_ir_version: str = ""
    policy_version: str = FORMULA_CLOSURE_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_aps", _normalized_names(self.allowed_aps))
        object.__setattr__(self, "required_aps", _normalized_names(self.required_aps))
        object.__setattr__(self, "optional_aps", _normalized_names(self.optional_aps))
        object.__setattr__(
            self,
            "allowed_boolean_constants",
            _normalized_boolean_constants(self.allowed_boolean_constants),
        )
        object.__setattr__(
            self,
            "allowed_temporal_operators",
            _normalized_names(self.allowed_temporal_operators),
        )
        object.__setattr__(
            self,
            "allowed_boolean_operators",
            tuple(
                sorted(
                    {
                        str(value).strip().lower()
                        for value in self.allowed_boolean_operators
                        if str(value).strip()
                    }
                )
            ),
        )
        object.__setattr__(self, "approved_ir_version", self.approved_ir_version.strip())
        object.__setattr__(self, "policy_version", self.policy_version.strip())

        allowed = set(self.allowed_aps)
        required = set(self.required_aps)
        optional = set(self.optional_aps)
        if required - allowed:
            raise ValueError(
                "required_aps must be a subset of allowed_aps: "
                + ", ".join(sorted(required - allowed))
            )
        if optional - allowed:
            raise ValueError(
                "optional_aps must be a subset of allowed_aps: "
                + ", ".join(sorted(optional - allowed))
            )
        if required & optional:
            raise ValueError(
                "required_aps and optional_aps must be disjoint: "
                + ", ".join(sorted(required & optional))
            )
        invalid_constants = set(self.allowed_boolean_constants) - BOOLEAN_CONSTANTS
        if invalid_constants:
            raise ValueError(
                "allowed_boolean_constants only accepts true/false: "
                + ", ".join(sorted(invalid_constants))
            )
        if not self.policy_version:
            raise ValueError("policy_version must not be empty")


@dataclass(frozen=True)
class FormulaClosureReport:
    """Deterministic result of checking one formula against one policy."""

    status: str
    formula_hash: str
    ir_version: str
    observed_aps: tuple[str, ...] = ()
    missing_required_aps: tuple[str, ...] = ()
    extra_aps: tuple[str, ...] = ()
    boolean_constants: tuple[str, ...] = ()
    placeholders: tuple[str, ...] = ()
    unsupported_operators: tuple[str, ...] = ()
    violating_node_paths: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True)
class FormulaASTNode:
    """One node yielded by the stable pre-order AST visitor."""

    path: str
    op: str
    name: str = ""


def visit_formula_ast(formula: Formula) -> Iterator[FormulaASTNode]:
    """Yield every node in pre-order with a stable ``root.args[N]`` path."""

    if not isinstance(formula, Formula):
        raise TypeError("formula must be an ltlf_parser.Formula")

    def visit(node: Formula, path: str) -> Iterator[FormulaASTNode]:
        if not isinstance(node, Formula):
            raise TypeError(f"formula child at {path} must be an ltlf_parser.Formula")
        yield FormulaASTNode(path=path, op=str(node.op), name=str(node.name))
        for index, child in enumerate(node.args):
            yield from visit(child, f"{path}.args[{index}]")

    yield from visit(formula, "root")


def stable_formula_hash(formula: Formula) -> str:
    """Return a stable SHA-256 hash of the exact AST, independent of rendering."""

    if not isinstance(formula, Formula):
        raise TypeError("formula must be an ltlf_parser.Formula")

    def payload(node: Formula) -> list[Any]:
        if not isinstance(node, Formula):
            raise TypeError("formula children must be ltlf_parser.Formula instances")
        return [str(node.op), str(node.name), [payload(child) for child in node.args]]

    canonical = json.dumps(
        payload(formula),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_formula_placeholder(value: object) -> bool:
    """Return whether an AP name or synthetic node operator is a placeholder."""

    normalized = str(value).strip().lower()
    return bool(
        normalized in _PLACEHOLDER_EXACT
        or _PLACEHOLDER_PREFIX.match(normalized)
    )


def validate_formula_closure(
    formula: Formula,
    policy: FormulaClosurePolicy,
    *,
    ir_version: str = "",
) -> FormulaClosureReport:
    """Validate ``formula`` against ``policy`` without mutating external state.

    ``ir_version`` is the version to which the candidate artifact is bound.  If
    the policy names an approved version, an omitted or different binding is a
    ``stale_ir_binding`` failure rather than an implicit match.
    """

    if not isinstance(policy, FormulaClosurePolicy):
        raise TypeError("policy must be a FormulaClosurePolicy")

    allowed_aps = set(policy.allowed_aps)
    required_aps = set(policy.required_aps)
    allowed_constants = set(policy.allowed_boolean_constants)
    allowed_temporal = set(policy.allowed_temporal_operators)
    allowed_boolean = set(policy.allowed_boolean_operators)

    observed_aps: set[str] = set()
    constants: set[str] = set()
    placeholders: set[str] = set()
    unsupported: set[str] = set()
    violation_paths: set[str] = set()
    reasons: list[str] = []

    for node in visit_formula_ast(formula):
        op = node.op
        normalized_op = op.lower()
        if op == "ap":
            if is_formula_placeholder(node.name):
                placeholders.add(node.name)
                if not policy.allow_placeholders:
                    violation_paths.add(node.path)
                    reasons.append(
                        f"unresolved_placeholder:{node.name}@{node.path}"
                    )
                continue
            observed_aps.add(node.name)
            if node.name not in allowed_aps:
                violation_paths.add(node.path)
                reasons.append(f"extra_ap:{node.name}@{node.path}")
            continue

        if normalized_op in BOOLEAN_CONSTANTS:
            constants.add(normalized_op)
            if normalized_op not in allowed_constants:
                violation_paths.add(node.path)
                reasons.append(
                    f"forbidden_boolean_constant:{normalized_op}@{node.path}"
                )
            continue

        if is_formula_placeholder(op):
            placeholders.add(op)
            if not policy.allow_placeholders:
                violation_paths.add(node.path)
                reasons.append(f"unresolved_placeholder:{op}@{node.path}")
            continue

        if op in allowed_temporal or normalized_op in allowed_boolean:
            continue
        unsupported.add(op)
        violation_paths.add(node.path)
        reasons.append(f"unsupported_operator:{op}@{node.path}")

    missing_required = required_aps - observed_aps
    extra_aps = observed_aps - allowed_aps
    for ap_name in sorted(missing_required):
        violation_paths.add(f"missing_required_ap[{ap_name}]")
        reasons.append(f"missing_required_ap:{ap_name}")

    bound_ir_version = str(ir_version).strip()
    if (
        policy.approved_ir_version
        and bound_ir_version != policy.approved_ir_version
    ):
        violation_paths.add("ir_version")
        reasons.append(
            "stale_ir_binding:expected="
            + policy.approved_ir_version
            + ":actual="
            + (bound_ir_version or "<missing>")
        )

    unique_reasons = tuple(dict.fromkeys(reasons))
    return FormulaClosureReport(
        status="passed" if not unique_reasons else "failed",
        formula_hash=stable_formula_hash(formula),
        ir_version=bound_ir_version,
        observed_aps=tuple(sorted(observed_aps)),
        missing_required_aps=tuple(sorted(missing_required)),
        extra_aps=tuple(sorted(extra_aps)),
        boolean_constants=tuple(sorted(constants)),
        placeholders=tuple(sorted(placeholders)),
        unsupported_operators=tuple(sorted(unsupported)),
        violating_node_paths=tuple(sorted(violation_paths)),
        reasons=unique_reasons,
    )


def build_strict_formula_closure_policy(
    faithful_ir: object | None = None,
    approved_semantic_ir: object | None = None,
    *,
    required_aps: Iterable[str] | None = None,
    optional_aps: Iterable[str] | None = None,
    allowed_temporal_operators: Iterable[str] = DEFAULT_TEMPORAL_OPERATORS,
    allowed_boolean_operators: Iterable[str] = DEFAULT_BOOLEAN_OPERATORS,
    policy_version: str = FORMULA_CLOSURE_POLICY_VERSION,
) -> FormulaClosurePolicy:
    """Build the default production policy from approved semantic artifacts.

    Approved AP mappings take precedence over the compatibility IR AP list.  A
    mismatch between two supplied artifacts is rejected instead of silently
    widening the alphabet.  Required/optional annotations are read from
    prospective IR metadata using deliberately explicit names:

    * AP or ``ap_condition`` attributes/parameters ``closure_role``,
      ``formula_presence``, or ``ap_usage`` with value ``required``/``optional``;
    * boolean ``required_in_formula`` / ``optional_in_formula``;
    * aggregate node parameters ``required_aps`` / ``optional_aps``.

    Unannotated APs are optional.  This conservative compatibility behavior
    avoids inventing a required-AP rule from declaration alone.
    """

    ir_aps = _normalized_names(getattr(faithful_ir, "ap_names", ()) or ())
    approved_mapping = getattr(approved_semantic_ir, "approved_ap_mapping", ()) or ()
    approved_aps = _normalized_names(
        item[1]
        for item in approved_mapping
        if isinstance(item, (tuple, list)) and len(item) >= 2
    )
    if approved_aps and ir_aps and set(approved_aps) != set(ir_aps):
        raise ValueError(
            "ApprovedSemanticIR AP mapping does not match FaithfulSemanticIR: "
            f"approved={approved_aps}, ir={ir_aps}"
        )
    allowed_aps = approved_aps or ir_aps
    if not allowed_aps:
        raise ValueError(
            "strict formula closure policy requires ApprovedSemanticIR or "
            "FaithfulSemanticIR AP declarations"
        )

    approved_version = str(
        getattr(approved_semantic_ir, "ir_version", "") or ""
    ).strip()
    faithful_version = str(getattr(faithful_ir, "ir_version", "") or "").strip()
    if approved_version and faithful_version and approved_version != faithful_version:
        raise ValueError(
            "ApprovedSemanticIR ir_version does not match FaithfulSemanticIR: "
            f"approved={approved_version}, ir={faithful_version}"
        )

    annotated_required, annotated_optional = _ir_ap_annotations(faithful_ir)
    selected_required = set(
        _normalized_names(required_aps)
        if required_aps is not None
        else annotated_required
    )
    if optional_aps is not None:
        selected_optional = set(_normalized_names(optional_aps))
    else:
        selected_optional = set(allowed_aps) - selected_required
        selected_optional.update(annotated_optional)
    conflict = selected_required & selected_optional
    if conflict:
        raise ValueError(
            "strict policy marks APs as both required and optional: "
            + ", ".join(sorted(conflict))
        )

    return FormulaClosurePolicy(
        allowed_aps=allowed_aps,
        required_aps=tuple(selected_required),
        optional_aps=tuple(selected_optional),
        allowed_boolean_constants=("true", "false") if any(
            n.node_type in {"start_boundary", "boolean_constant", "short_trace_boundary"}
            for n in getattr(faithful_ir, "nodes", ())) else (),
        allowed_temporal_operators=tuple(allowed_temporal_operators),
        allowed_boolean_operators=tuple(allowed_boolean_operators),
        allow_placeholders=False,
        approved_ir_version=approved_version or faithful_version,
        policy_version=policy_version,
    )


def strict_formula_closure_policy(*args: Any, **kwargs: Any) -> FormulaClosurePolicy:
    """Backward-friendly alias for :func:`build_strict_formula_closure_policy`."""

    return build_strict_formula_closure_policy(*args, **kwargs)


def _ir_ap_annotations(faithful_ir: object | None) -> tuple[set[str], set[str]]:
    required: set[str] = set()
    optional: set[str] = set()
    if faithful_ir is None:
        return required, optional

    for ap in getattr(faithful_ir, "atomic_propositions", ()) or ():
        ap_name = str(getattr(ap, "name", "") or "").strip()
        if not ap_name:
            continue
        role = _first_nonempty_attribute(
            ap,
            ("closure_role", "formula_presence", "ap_usage"),
        )
        _add_annotation_role(ap_name, role, required, optional)
        if bool(getattr(ap, "required_in_formula", False)):
            required.add(ap_name)
        if bool(getattr(ap, "optional_in_formula", False)):
            optional.add(ap_name)

    for node in getattr(faithful_ir, "nodes", ()) or ():
        raw_parameters = getattr(node, "parameters", ()) or ()
        try:
            parameters: Mapping[str, object] = dict(raw_parameters)
        except (TypeError, ValueError):
            parameters = {}
        required.update(_names_from_annotation(parameters.get("required_aps")))
        optional.update(_names_from_annotation(parameters.get("optional_aps")))
        if str(getattr(node, "node_type", "")) != "ap_condition":
            continue
        ap_name = str(parameters.get("ap_name", "") or "").strip()
        if not ap_name:
            continue
        role = next(
            (
                str(parameters[key]).strip()
                for key in ("closure_role", "formula_presence", "ap_usage")
                if str(parameters.get(key, "")).strip()
            ),
            "",
        )
        _add_annotation_role(ap_name, role, required, optional)
        if _truthy_annotation(parameters.get("required_in_formula")):
            required.add(ap_name)
        if _truthy_annotation(parameters.get("optional_in_formula")):
            optional.add(ap_name)

    conflict = required & optional
    if conflict:
        raise ValueError(
            "IR marks APs as both required and optional: "
            + ", ".join(sorted(conflict))
        )
    declared = set(getattr(faithful_ir, "ap_names", ()) or ())
    unknown = (required | optional) - declared
    if unknown:
        raise ValueError(
            "IR formula-closure annotations reference undeclared APs: "
            + ", ".join(sorted(unknown))
        )
    return required, optional


def _first_nonempty_attribute(value: object, names: tuple[str, ...]) -> str:
    for name in names:
        item = str(getattr(value, name, "") or "").strip()
        if item:
            return item
    return ""


def _add_annotation_role(
    ap_name: str,
    role: object,
    required: set[str],
    optional: set[str],
) -> None:
    normalized = str(role).strip().lower()
    if normalized in {"required", "must_appear", "mandatory"}:
        required.add(ap_name)
    elif normalized in {"optional", "may_appear", "background"}:
        optional.add(ap_name)


def _truthy_annotation(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _names_from_annotation(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {
            item
            for item in re.split(r"[\s,;]+", value.strip())
            if item
        }
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()} if str(value).strip() else set()


__all__ = [
    "BOOLEAN_CONSTANTS",
    "DEFAULT_BOOLEAN_OPERATORS",
    "DEFAULT_TEMPORAL_OPERATORS",
    "FORMULA_CLOSURE_POLICY_VERSION",
    "FormulaASTNode",
    "FormulaClosurePolicy",
    "FormulaClosureReport",
    "build_strict_formula_closure_policy",
    "is_formula_placeholder",
    "stable_formula_hash",
    "strict_formula_closure_policy",
    "validate_formula_closure",
    "visit_formula_ast",
]

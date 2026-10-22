"""Stable-ID repair contract for Plain Reading artifacts.

The LLM proposes domain intent only.  Array positions, JSON Pointer ordering,
root status, and the final candidate are backend-owned.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from contract_versions import (
    PLAIN_READING_REPAIR_COMPILER_VERSION,
    PLAIN_READING_REPAIR_PLAN_CONTRACT_VERSION,
)
from validated_generation import (
    PatchValidationError,
    ValidationIssue,
    artifact_hash,
)


ACTION_REMOVE_AXIS = "remove_axis"
ACTION_REPLACE_OPTION_TEXT = "replace_option_text"
ACTION_REPLACE_OPTION_PAIR = "replace_option_pair"


@dataclass(frozen=True)
class PlainReadingRepairTask:
    task_id: str
    artifact_hash: str
    ambiguity_id: str
    target_kind: str
    option_id: str = ""
    issue_codes: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    failed_grounding_checks: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()

    @property
    def target_key(self) -> tuple[str, str, str]:
        return self.ambiguity_id, self.option_id, self.target_kind


@dataclass(frozen=True)
class PlainReadingRepairAction:
    action: str
    ambiguity_id: str = ""
    option_id: str = ""
    plain_interpretation: str = ""
    options: tuple[dict[str, Any], ...] = ()
    reason_code: str = ""


@dataclass(frozen=True)
class PlainReadingRepairPlan:
    artifact_hash: str
    actions: tuple[PlainReadingRepairAction, ...]
    contract_version: str = PLAIN_READING_REPAIR_PLAN_CONTRACT_VERSION


@dataclass(frozen=True)
class CandidateCompilation:
    candidate: dict[str, Any]
    changed_targets: tuple[str, ...]
    plan_fingerprint: str


def normalize_plain_reading_issues(
    current: Mapping[str, Any],
    issues: Sequence[ValidationIssue],
    *,
    forced_generation: bool = False,
) -> tuple[PlainReadingRepairTask, ...]:
    """Collapse many findings into one authorized task per stable target."""

    current_hash = artifact_hash(current)
    axes = current.get("ambiguities", ())
    if not isinstance(axes, (list, tuple)):
        axes = ()
    axis_ids = [
        str(item.get("ambiguity_id", ""))
        if isinstance(item, Mapping)
        else ""
        for item in axes
    ]
    grouped: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}
    axis_dominates: set[str] = set()

    for issue in issues:
        axis_id, option_id = _issue_target(issue, axes, axis_ids)
        target_kind = _target_kind(issue, axis_id, option_id)
        if target_kind == "axis":
            axis_dominates.add(axis_id)
        key = (axis_id, option_id, target_kind)
        entry = grouped.setdefault(
            key,
            {
                "codes": [],
                "messages": [],
                "failed": [],
            },
        )
        entry["codes"].append(issue.code)
        if issue.message:
            entry["messages"].append(issue.message)
        failed = issue.context.get("failed_checks", ())
        if isinstance(failed, (list, tuple)):
            entry["failed"].extend(str(item) for item in failed)

    for key in tuple(grouped):
        axis_id, _option_id, kind = key
        if axis_id in axis_dominates and kind != "axis":
            del grouped[key]

    tasks: list[PlainReadingRepairTask] = []
    for (axis_id, option_id, target_kind), entry in sorted(grouped.items()):
        allowed = _allowed_actions(
            target_kind,
            axis_id=axis_id,
            axis_count=len(axes),
            forced_generation=forced_generation,
        )
        if not allowed:
            continue
        task_payload = {
            "artifact_hash": current_hash,
            "ambiguity_id": axis_id,
            "option_id": option_id,
            "target_kind": target_kind,
            "issue_codes": sorted(set(entry["codes"])),
        }
        tasks.append(
            PlainReadingRepairTask(
                task_id="prt-" + _digest(task_payload, 16),
                artifact_hash=current_hash,
                ambiguity_id=axis_id,
                option_id=option_id,
                target_kind=target_kind,
                issue_codes=tuple(dict.fromkeys(entry["codes"])),
                messages=tuple(dict.fromkeys(entry["messages"])),
                failed_grounding_checks=tuple(
                    dict.fromkeys(entry["failed"])
                ),
                allowed_actions=allowed,
            )
        )
    return tuple(tasks)


def repair_plan_response_schema(
    tasks: Sequence[PlainReadingRepairTask],
) -> dict[str, Any]:
    """Return the actual structured-output schema for repair intent."""

    axis_ids = sorted(
        {
            task.ambiguity_id
            for task in tasks
            if task.ambiguity_id
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["artifact_hash", "actions"],
        "properties": {
            "artifact_hash": {
                "type": "string",
                "enum": sorted({task.artifact_hash for task in tasks}),
            },
            "actions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": sorted(
                                {
                                    action
                                    for task in tasks
                                    for action in task.allowed_actions
                                }
                            ),
                        },
                        "ambiguity_id": {
                            "type": "string",
                            "enum": axis_ids or [""],
                        },
                        "option_id": {
                            "type": "string",
                            "enum": ["", "A", "B"],
                        },
                        "plain_interpretation": {"type": "string"},
                        "options": {"type": "array"},
                        "reason_code": {"type": "string"},
                    },
                },
            },
        },
    }


def parse_plain_reading_repair_plan(
    raw: object,
    tasks: Sequence[PlainReadingRepairTask],
) -> PlainReadingRepairPlan:
    if not isinstance(raw, Mapping):
        raise PatchValidationError("plain-reading repair plan must be an object")
    if "operations" in raw:
        raise PatchValidationError(
            "plain-reading repair no longer accepts JSON Patch operations"
        )
    current_hashes = {task.artifact_hash for task in tasks}
    plan_hash = str(raw.get("artifact_hash", ""))
    if len(current_hashes) != 1 or plan_hash not in current_hashes:
        raise PatchValidationError("plain-reading repair plan is stale")
    raw_actions = raw.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise PatchValidationError("plain-reading repair plan requires actions")
    actions: list[PlainReadingRepairAction] = []
    for raw_action in raw_actions:
        if not isinstance(raw_action, Mapping):
            raise PatchValidationError("plain-reading repair action must be an object")
        unknown = set(raw_action) - {
            "action",
            "ambiguity_id",
            "option_id",
            "plain_interpretation",
            "options",
            "reason_code",
        }
        if unknown:
            raise PatchValidationError(
                "unknown plain-reading repair fields: "
                + ", ".join(sorted(unknown))
            )
        options = raw_action.get("options", ())
        actions.append(
            PlainReadingRepairAction(
                action=str(raw_action.get("action", "")),
                ambiguity_id=str(raw_action.get("ambiguity_id", "")),
                option_id=str(raw_action.get("option_id", "")),
                plain_interpretation=str(
                    raw_action.get("plain_interpretation", "")
                ),
                options=tuple(
                    dict(item)
                    for item in options
                    if isinstance(item, Mapping)
                )
                if isinstance(options, list)
                else (),
                reason_code=str(raw_action.get("reason_code", "")),
            )
        )
    plan = PlainReadingRepairPlan(plan_hash, tuple(actions))
    validate_plain_reading_repair_plan(plan, tasks)
    return plan


def validate_plain_reading_repair_plan(
    plan: PlainReadingRepairPlan,
    tasks: Sequence[PlainReadingRepairTask],
) -> None:
    task_by_axis: dict[str, list[PlainReadingRepairTask]] = {}
    for task in tasks:
        task_by_axis.setdefault(task.ambiguity_id, []).append(task)
    seen: set[tuple[str, str, str]] = set()
    actions_by_axis: dict[str, set[str]] = {}
    for action in plan.actions:
        matching = [
            task
            for task in task_by_axis.get(action.ambiguity_id, ())
            if action.action in task.allowed_actions
            and (
                action.action != ACTION_REPLACE_OPTION_TEXT
                or task.option_id == action.option_id
            )
        ]
        if not matching:
            raise PatchValidationError(
                "plain-reading repair target/action is not authorized: "
                f"{action.ambiguity_id}:{action.option_id}:{action.action}"
            )
        key = (action.ambiguity_id, action.option_id, action.action)
        if key in seen:
            raise PatchValidationError("duplicate plain-reading repair action")
        seen.add(key)
        actions_by_axis.setdefault(action.ambiguity_id, set()).add(
            action.action
        )
        if action.action == ACTION_REPLACE_OPTION_TEXT:
            if action.option_id not in {"A", "B"}:
                raise PatchValidationError("replace_option_text requires A or B")
            if not action.plain_interpretation.strip():
                raise PatchValidationError(
                    "replace_option_text requires complete non-empty text"
                )
        elif action.action == ACTION_REPLACE_OPTION_PAIR:
            _validate_option_pair(action.options)
        elif action.action != ACTION_REMOVE_AXIS:
            raise PatchValidationError(
                "unsupported plain-reading repair action: " + action.action
            )
    for axis_id, kinds in actions_by_axis.items():
        if ACTION_REMOVE_AXIS in kinds and len(kinds) != 1:
            raise PatchValidationError(
                f"remove_axis conflicts with replacements for {axis_id}"
            )
        if (
            ACTION_REPLACE_OPTION_PAIR in kinds
            and ACTION_REPLACE_OPTION_TEXT in kinds
        ):
            raise PatchValidationError(
                f"replace_option_pair conflicts with option text for {axis_id}"
            )


def compile_plain_reading_repair_plan(
    current: Mapping[str, Any],
    tasks: Sequence[PlainReadingRepairTask],
    plan: PlainReadingRepairPlan,
    *,
    forced_generation: bool = False,
) -> CandidateCompilation:
    """Compile a validated stable-ID plan into a complete candidate artifact."""

    if artifact_hash(current) != plan.artifact_hash:
        raise PatchValidationError("plain-reading repair plan is stale")
    validate_plain_reading_repair_plan(plan, tasks)
    candidate = deepcopy(dict(current))
    axes = candidate.get("ambiguities")
    if not isinstance(axes, list):
        raise PatchValidationError("plain-reading candidate has no axis array")
    changed: list[str] = []

    by_id = {
        str(axis.get("ambiguity_id", "")): axis
        for axis in axes
        if isinstance(axis, dict)
    }
    if len(by_id) != len(axes):
        raise PatchValidationError(
            "plain-reading repair requires unique stable ambiguity ids"
        )
    removed: set[str] = set()
    for action in plan.actions:
        axis = by_id.get(action.ambiguity_id)
        if axis is None:
            raise PatchValidationError(
                "plain-reading repair target no longer exists"
            )
        if action.action == ACTION_REMOVE_AXIS:
            removed.add(action.ambiguity_id)
            changed.append(f"axis:{action.ambiguity_id}:removed")
        elif action.action == ACTION_REPLACE_OPTION_TEXT:
            option = _option_by_id(axis, action.option_id)
            option["plain_interpretation"] = action.plain_interpretation
            changed.append(
                f"axis:{action.ambiguity_id}:option:{action.option_id}"
            )
        elif action.action == ACTION_REPLACE_OPTION_PAIR:
            axis["options"] = [
                deepcopy(dict(item)) for item in action.options
            ]
            changed.append(f"axis:{action.ambiguity_id}:option_pair")
    axes = [
        axis
        for axis in axes
        if str(axis.get("ambiguity_id", "")) not in removed
    ]

    if forced_generation and not axes:
        raise PatchValidationError(
            "forced ambiguity generation cannot remove the final axis"
        )
    candidate["ambiguities"] = axes
    candidate["overall_status"] = (
        "pending_plain_review" if axes else "no_ambiguity"
    )
    if not changed:
        raise PatchValidationError("plain-reading repair plan makes no changes")
    fingerprint = _digest(
        {
            "plan": asdict(plan),
            "compiler_version": PLAIN_READING_REPAIR_COMPILER_VERSION,
        },
        64,
    )
    return CandidateCompilation(
        candidate=candidate,
        changed_targets=tuple(sorted(changed)),
        plan_fingerprint=fingerprint,
    )


def deterministic_axis_removal_plan(
    current: Mapping[str, Any],
    tasks: Sequence[PlainReadingRepairTask],
    *,
    allowed_issue_codes: frozenset[str],
) -> PlainReadingRepairPlan | None:
    actions = []
    for task in tasks:
        if (
            task.target_kind != "axis"
            or not set(task.issue_codes) <= allowed_issue_codes
            or ACTION_REMOVE_AXIS not in task.allowed_actions
        ):
            return None
        actions.append(
            PlainReadingRepairAction(
                action=ACTION_REMOVE_AXIS,
                ambiguity_id=task.ambiguity_id,
                reason_code=task.issue_codes[0] if task.issue_codes else "",
            )
        )
    if not actions:
        return None
    return PlainReadingRepairPlan(
        artifact_hash=artifact_hash(current),
        actions=tuple(actions),
    )


def _issue_target(
    issue: ValidationIssue,
    axes: Sequence[object],
    axis_ids: Sequence[str],
) -> tuple[str, str]:
    axis_id = str(issue.context.get("ambiguity_id", ""))
    option_id = str(issue.context.get("option_id", ""))
    match = re.match(r"^/ambiguities/(\d+)(?:/options/(\d+))?", issue.path)
    if match:
        axis_index = int(match.group(1))
        if not axis_id and 0 <= axis_index < len(axis_ids):
            axis_id = axis_ids[axis_index]
        if not option_id and match.group(2) is not None:
            option_index = int(match.group(2))
            if 0 <= axis_index < len(axes):
                axis = axes[axis_index]
                options = (
                    axis.get("options", ())
                    if isinstance(axis, Mapping)
                    else ()
                )
                if (
                    isinstance(options, (list, tuple))
                    and 0 <= option_index < len(options)
                    and isinstance(options[option_index], Mapping)
                ):
                    option_id = str(
                        options[option_index].get("option_id", "")
                    )
    return axis_id, option_id


def _target_kind(
    issue: ValidationIssue,
    axis_id: str,
    option_id: str,
) -> str:
    if not axis_id:
        return "axis_collection"
    if issue.code.startswith("AXIS_") or issue.code in {
        "PARAPHRASE_OR_NON_DISTINCT_AXIS",
    }:
        return "axis"
    if option_id and issue.path.endswith("/plain_interpretation"):
        return "option_text"
    if issue.code.startswith("OPTION_") and option_id:
        return "option_text"
    return "option_pair"


def _allowed_actions(
    target_kind: str,
    *,
    axis_id: str,
    axis_count: int,
    forced_generation: bool,
) -> tuple[str, ...]:
    removal = (
        ()
        if forced_generation and axis_count <= 1
        else (ACTION_REMOVE_AXIS,)
    )
    if target_kind == "axis":
        return (*removal, ACTION_REPLACE_OPTION_PAIR)
    if target_kind == "option_text":
        return (ACTION_REPLACE_OPTION_TEXT, *removal)
    if target_kind == "option_pair":
        return (ACTION_REPLACE_OPTION_PAIR, *removal)
    return ()


def _validate_option_pair(options: Sequence[Mapping[str, Any]]) -> None:
    if len(options) != 2:
        raise PatchValidationError("replace_option_pair requires exactly two options")
    ids = {str(item.get("option_id", "")) for item in options}
    if ids != {"A", "B"}:
        raise PatchValidationError("replace_option_pair requires option ids A and B")
    for item in options:
        if not str(item.get("plain_interpretation", "")).strip():
            raise PatchValidationError(
                "replace_option_pair requires complete interpretations"
            )
        if not str(item.get("difference_summary", "")).strip():
            raise PatchValidationError(
                "replace_option_pair requires difference summaries"
            )


def _option_by_id(axis: dict[str, Any], option_id: str) -> dict[str, Any]:
    options = axis.get("options")
    if not isinstance(options, list):
        raise PatchValidationError("plain-reading axis has no option array")
    matches = [
        item
        for item in options
        if isinstance(item, dict)
        and str(item.get("option_id", "")) == option_id
    ]
    if len(matches) != 1:
        raise PatchValidationError(
            "plain-reading option target is missing or ambiguous"
        )
    return matches[0]


def _digest(value: object, length: int) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:length]


__all__ = [
    "ACTION_REMOVE_AXIS",
    "ACTION_REPLACE_OPTION_PAIR",
    "ACTION_REPLACE_OPTION_TEXT",
    "CandidateCompilation",
    "PlainReadingRepairAction",
    "PlainReadingRepairPlan",
    "PlainReadingRepairTask",
    "compile_plain_reading_repair_plan",
    "deterministic_axis_removal_plan",
    "normalize_plain_reading_issues",
    "parse_plain_reading_repair_plan",
    "repair_plan_response_schema",
    "validate_plain_reading_repair_plan",
]

"""Contract-driven generation, validation, and bounded repair orchestration.

The module deliberately has no dependency on a concrete agent or LLM client.
Transport retries stay in the client; this runner only accounts for domain
proposal and repair work.
"""

from __future__ import annotations
from execution_observer import emit

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, IntEnum
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence


CONTRACT_VERSION = "validated-generation-contract-v2"
VALIDATOR_VERSION = "validated-generation-validator-v2"
CANONICALIZER_VERSION = "validated-generation-canonicalizer-v1"
REPAIR_POLICY_VERSION = "validated-generation-repair-policy-v2"
PATCH_AUTHORIZATION_CONTRACT_VERSION = "patch-authorization-v1"


class RepairClass(str, Enum):
    TRANSPORT = "transport"
    JSON_SYNTAX = "json_syntax"
    DETERMINISTIC = "deterministic"
    CONSTRAINED = "constrained"
    SEMANTIC_PATCH = "semantic_patch"
    UNSATISFIABLE = "unsatisfiable"
    HUMAN_REQUIRED = "human_required"
    INTERNAL_BUG = "internal_bug"


class Severity(IntEnum):
    INFO = 10
    WARNING = 20
    ERROR = 30
    CRITICAL = 40

    @classmethod
    def parse(cls, value: object) -> "Severity":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value or "error").strip().upper()]


@dataclass(frozen=True)
class ValidationIssue:
    """A stable, machine-actionable validator finding."""

    code: str
    stage: str
    path: str
    observed: Any = None
    expected: Any = None
    context: Mapping[str, Any] = field(default_factory=dict)
    repair_class: RepairClass = RepairClass.HUMAN_REQUIRED
    allowed_paths: tuple[str, ...] = ()
    message: str = ""
    severity: Severity = Severity.ERROR

    def __post_init__(self) -> None:
        if not self.code or not self.stage:
            raise ValueError("ValidationIssue requires non-empty code and stage")
        parse_json_pointer(self.path)
        for path in self.allowed_paths:
            parse_json_pointer(path)
        object.__setattr__(self, "repair_class", RepairClass(self.repair_class))
        object.__setattr__(self, "severity", Severity.parse(self.severity))
        object.__setattr__(self, "allowed_paths", tuple(self.allowed_paths))
        object.__setattr__(self, "context", dict(self.context))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "path": self.path,
            "observed": _json_value(self.observed),
            "expected": _json_value(self.expected),
            "context": _json_value(self.context),
            "repair_class": self.repair_class.value,
            "allowed_paths": list(self.allowed_paths),
            "message": self.message,
            "severity": self.severity.name.lower(),
        }


@dataclass(frozen=True)
class CanonicalizedArtifact:
    artifact: Any
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True)
class PatchOperation:
    op: str
    path: str
    value: Any = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PatchOperation":
        operation = value.get("op")
        if operation not in {"add", "replace", "remove"}:
            raise PatchValidationError(
                "unsupported patch operation: " + str(operation or "")
            )
        expected_keys = (
            {"op", "path"}
            if operation == "remove"
            else {"op", "path", "value"}
        )
        if set(value) != expected_keys:
            if operation in {"add", "replace"} and "value" not in value:
                raise PatchValidationError(
                    f"{operation} patch operation requires value"
                )
            if operation == "remove" and "value" in value:
                raise PatchValidationError(
                    "remove patch operation must not contain value"
                )
            raise PatchValidationError(
                "patch operation contains unexpected fields"
            )
        if not isinstance(value.get("path"), str):
            raise PatchValidationError("patch operation path must be a string")
        return cls(
            op=operation,
            path=value["path"],
            value=value.get("value"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {"op": self.op, "path": self.path}
        if self.op != "remove":
            result["value"] = _json_value(self.value)
        return result


class PatchValidationError(ValueError):
    """Raised before an unauthorized or malformed patch can mutate state."""


@dataclass(frozen=True)
class PatchTarget:
    """One exact JSON Pointer target and its authorized operations."""

    path: str
    allowed_operations: tuple[str, ...]
    value_schema: Mapping[str, Any] = field(default_factory=dict)
    exists: bool | None = None
    required: bool | None = None

    def __post_init__(self) -> None:
        parse_json_pointer(self.path)
        operations = tuple(dict.fromkeys(str(item) for item in self.allowed_operations))
        if not operations or any(
            operation not in {"add", "replace", "remove"}
            for operation in operations
        ):
            raise ValueError("PatchTarget requires supported allowed_operations")
        object.__setattr__(self, "allowed_operations", operations)
        object.__setattr__(self, "value_schema", deepcopy(dict(self.value_schema)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "required": self.required,
            "allowed_operations": list(self.allowed_operations),
            "value_schema": deepcopy(dict(self.value_schema)),
        }


@dataclass(frozen=True)
class PatchAuthorization:
    """Single source of truth for prompt, response schema, and execution."""

    targets: tuple[PatchTarget, ...]
    immutable_paths: tuple[str, ...] = ()
    contract_version: str = PATCH_AUTHORIZATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.targets:
            raise PatchValidationError("patch authorization has no executable targets")
        paths = [target.path for target in self.targets]
        if len(paths) != len(set(paths)):
            raise ValueError("patch authorization contains duplicate target paths")
        for path in self.immutable_paths:
            parse_json_pointer(path)
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "immutable_paths", tuple(self.immutable_paths))

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(target.path for target in self.targets)

    @property
    def fingerprint(self) -> str:
        payload = {
            "contract_version": self.contract_version,
            "targets": [target.to_dict() for target in self.targets],
            "immutable_paths": list(self.immutable_paths),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def target_for(self, path: str) -> PatchTarget | None:
        return next((target for target in self.targets if target.path == path), None)

    def require_operation(self, operation: PatchOperation) -> PatchTarget:
        target = self.target_for(operation.path)
        if target is None:
            allowed = ", ".join(self.paths)
            raise PatchValidationError(
                "patch path is not an exact authorized target: "
                f"{operation.path}; allowed paths: {allowed}"
            )
        if operation.op not in target.allowed_operations:
            allowed = ", ".join(target.allowed_operations)
            raise PatchValidationError(
                f"patch operation {operation.op} is not allowed for {operation.path}; "
                f"allowed operations: {allowed}"
            )
        return target

    def to_prompt_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "path": target.path,
                "exists": target.exists,
                "allowed_operations": list(target.allowed_operations),
            }
            for target in self.targets
        ]


@dataclass(frozen=True)
class PatchSubmission:
    """An LLM patch response bound to the authorization used to request it."""

    response: Any
    authorization: PatchAuthorization


@dataclass(frozen=True)
class CandidateSubmission:
    """A domain compiler's complete in-memory candidate.

    This bypasses only JSON-Patch parsing.  Canonicalization, full validation,
    progress detection, rollback, and repair budgets remain unchanged.
    """

    candidate: object
    changed_targets: tuple[str, ...] = ()
    contract_fingerprint: str = ""


PatchValueContractFn = Callable[
    [object | None, str], tuple[Mapping[str, Any], bool | None]
]


def build_patch_authorization(
    document: object | None,
    paths: Sequence[str],
    *,
    immutable_paths: Sequence[str] = (),
    value_contract_fn: PatchValueContractFn | None = None,
) -> PatchAuthorization:
    """Compile exact patch targets once for prompting, validation, and execution."""

    targets: list[PatchTarget] = []
    for path in sorted({str(path) for path in paths if str(path)}):
        parse_json_pointer(path)
        if path == "":
            continue
        if any(
            document is not None
            and json_pointer_exists(document, immutable_path)
            and _paths_overlap(path, immutable_path)
            for immutable_path in immutable_paths
        ):
            continue
        value_schema, required = (
            value_contract_fn(document, path)
            if value_contract_fn is not None
            else ({}, None)
        )
        exists = json_pointer_exists(document, path) if document is not None else None
        if exists is True:
            operations = ("replace",) if required is True else ("replace", "remove")
        elif exists is False:
            parent_tokens = parse_json_pointer(path)[:-1]
            parent_path = json_pointer(*parent_tokens)
            if parent_tokens and not json_pointer_exists(document, parent_path):
                continue
            operations = ("add",)
        else:
            operations = (
                ("add", "replace")
                if required is True
                else ("add", "replace", "remove")
            )
        targets.append(
            PatchTarget(
                path=path,
                allowed_operations=operations,
                value_schema=value_schema,
                exists=exists,
                required=required,
            )
        )
    return PatchAuthorization(
        targets=tuple(targets),
        immutable_paths=tuple(immutable_paths),
    )


@dataclass(frozen=True)
class RepairPolicy:
    max_domain_repairs: int = 2
    max_constrained_repairs: int = 1
    max_semantic_patches: int = 1
    immutable_paths: tuple[str, ...] = (
        "/source_text",
        "/source_hash",
        "/schema_version",
    )
    reject_full_artifact_patch: bool = True
    escalate_constrained_to_semantic: bool = True
    contract_version: str = CONTRACT_VERSION
    canonicalizer_version: str = CANONICALIZER_VERSION
    validator_version: str = VALIDATOR_VERSION
    repair_policy_version: str = REPAIR_POLICY_VERSION

    def __post_init__(self) -> None:
        if min(
            self.max_domain_repairs,
            self.max_constrained_repairs,
            self.max_semantic_patches,
        ) < 0:
            raise ValueError("repair budgets must be non-negative")
        for path in self.immutable_paths:
            parse_json_pointer(path)


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    repair_strategy: str
    artifact_hash_before: str
    artifact_hash_after: str
    issue_fingerprint_before: str
    issue_fingerprint_after: str
    issue_count_before: Mapping[str, int]
    issue_count_after: Mapping[str, int]
    issue_weight_before: Mapping[str, int] = field(default_factory=dict)
    issue_weight_after: Mapping[str, int] = field(default_factory=dict)
    proposed_patch_paths: tuple[str, ...] = ()
    proposed_patch_operations: tuple[Mapping[str, Any], ...] = ()
    accepted_patch_paths: tuple[str, ...] = ()
    resolved_issues: tuple[str, ...] = ()
    introduced_issues: tuple[str, ...] = ()
    newly_exposed_issues: tuple[str, ...] = ()
    repair_error: str = ""
    repair_error_code: str = ""
    repair_stage: str = ""
    patch_contract_fingerprint: str = ""
    progress: bool = False
    accepted: bool = False
    reason_code: str = ""
    issues: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["proposed_patch_paths"] = list(self.proposed_patch_paths)
        value["proposed_patch_operations"] = [
            _json_value(item) for item in self.proposed_patch_operations
        ]
        value["accepted_patch_paths"] = list(self.accepted_patch_paths)
        value["resolved_issues"] = list(self.resolved_issues)
        value["introduced_issues"] = list(self.introduced_issues)
        value["newly_exposed_issues"] = list(self.newly_exposed_issues)
        value["issues"] = [dict(item) for item in self.issues]
        return value


@dataclass(frozen=True)
class GenerationResult:
    status: str
    reason_code: str
    artifact: Any
    issues: tuple[ValidationIssue, ...]
    attempts: tuple[AttemptRecord, ...]
    contract_version: str
    canonicalizer_version: str
    validator_version: str
    repair_policy_version: str
    prompt_version: str = ""
    artifact_hash: str = ""
    issue_fingerprint: str = ""
    rejected_candidate: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def diagnostic(self) -> dict[str, Any]:
        diagnostic = {
            "status": self.status,
            "reason_code": self.reason_code,
            "contract_version": self.contract_version,
            "canonicalizer_version": self.canonicalizer_version,
            "validator_version": self.validator_version,
            "repair_policy_version": self.repair_policy_version,
            "prompt_version": self.prompt_version,
            "artifact_hash": self.artifact_hash,
            "issue_fingerprint": self.issue_fingerprint,
            "issues": [issue.to_dict() for issue in self.issues],
            "current_issues": [issue.to_dict() for issue in self.issues],
            "repair_history": [attempt.to_dict() for attempt in self.attempts],
        }
        if self.rejected_candidate:
            diagnostic["rejected_candidate"] = _json_value(
                self.rejected_candidate
            )
        return diagnostic


class GenerationTerminatedError(RuntimeError):
    def __init__(self, result: GenerationResult):
        super().__init__(
            f"validated generation terminated: {result.reason_code}"
            + (
                ": " + "; ".join(issue.message or issue.code for issue in result.issues)
                if result.issues
                else ""
            )
        )
        self.result = result
        self.diagnostics = result.diagnostic()


ProposalFn = Callable[[], Any]
CanonicalizeFn = Callable[[Any], Any]
ValidateFn = Callable[[Any], Iterable[ValidationIssue]]
DeterministicRepairFn = Callable[[Any, Sequence[ValidationIssue]], Any]
JSONRepairFn = Callable[[Any, Sequence[ValidationIssue]], Any]
PatchFn = Callable[[Any, Sequence[ValidationIssue], RepairClass], Any]
FeasibilityFn = Callable[[Any, Sequence[ValidationIssue]], Iterable[ValidationIssue]]


class ValidatedGenerationRunner:
    """Generate once, then use bounded deterministic or atomic patch repair."""

    def __init__(self, policy: RepairPolicy | None = None):
        self.policy = policy or RepairPolicy()

    def generate(
        self,
        *,
        proposal_fn: ProposalFn,
        validate_fn: ValidateFn,
        canonicalize_fn: CanonicalizeFn | None = None,
        json_repair_fn: JSONRepairFn | None = None,
        deterministic_repair_fn: DeterministicRepairFn | None = None,
        patch_fn: PatchFn | None = None,
        feasibility_fn: FeasibilityFn | None = None,
        policy: RepairPolicy | None = None,
        prompt_version: str = "",
        raise_on_failure: bool = False,
    ) -> GenerationResult:
        active_policy = policy or self.policy
        # Transport exceptions intentionally escape and consume no domain budget.
        artifact = proposal_fn()
        artifact, canonical_issues = _canonicalize(artifact, canonicalize_fn)
        issues = _collect_issues(canonical_issues, validate_fn(artifact))
        if feasibility_fn:
            issues = _collect_issues(issues, feasibility_fn(artifact, issues))
        attempts: list[AttemptRecord] = [
            AttemptRecord(
                attempt=1,
                repair_strategy="initial_proposal",
                artifact_hash_before="",
                artifact_hash_after=artifact_hash(artifact),
                issue_fingerprint_before="",
                issue_fingerprint_after=issue_fingerprint(issues),
                issue_count_before={},
                issue_count_after=issue_count_by_severity(issues),
                issue_weight_before={},
                issue_weight_after=issue_weight_by_severity(issues),
                progress=not issues,
                accepted=True,
                reason_code="VALID" if not issues else "VALIDATION_FAILED",
                issues=tuple(issue.to_dict() for issue in issues),
            )
        ]
        if not issues:
            return self._result(
                "succeeded", "INITIAL_VALID", artifact, issues, attempts,
                active_policy, prompt_version,
            )

        processed: set[tuple[str, str, str]] = set()
        constrained_count = 0
        semantic_count = 0
        repair_count = 0

        while issues and repair_count < active_policy.max_domain_repairs:
            terminal = _terminal_reason(issues)
            if terminal is not None:
                result = self._result(
                    terminal[0], terminal[1], artifact, issues, attempts,
                    active_policy, prompt_version,
                )
                return _raise_or_return(result, raise_on_failure)

            repair_class = _next_repair_class(issues)
            strategy = repair_class.value
            key = (artifact_hash(artifact), issue_fingerprint(issues), strategy)
            if key in processed:
                attempts.append(
                    _no_progress_record(len(attempts) + 1, strategy, artifact, issues)
                )
                result = self._result(
                    "blocked", "NO_PROGRESS", artifact, issues, attempts,
                    active_policy, prompt_version,
                )
                return _raise_or_return(result, raise_on_failure)
            processed.add(key)

            selected = tuple(issue for issue in issues if issue.repair_class == repair_class)
            before = deepcopy(artifact)
            before_issues = issues
            accepted_paths: tuple[str, ...] = ()
            proposed_paths: tuple[str, ...] = ()
            proposed_operations: tuple[Mapping[str, Any], ...] = ()
            patch_contract_fingerprint = ""
            reason = ""
            try:
                if repair_class == RepairClass.DETERMINISTIC:
                    if deterministic_repair_fn is None:
                        reason = "DETERMINISTIC_REPAIR_UNAVAILABLE"
                        raise PatchValidationError(reason)
                    candidate = deterministic_repair_fn(deepcopy(artifact), selected)
                elif repair_class == RepairClass.JSON_SYNTAX:
                    if json_repair_fn is None:
                        reason = "JSON_REPAIR_UNAVAILABLE"
                        raise PatchValidationError(reason)
                    candidate = json_repair_fn(deepcopy(artifact), selected)
                elif repair_class in {RepairClass.CONSTRAINED, RepairClass.SEMANTIC_PATCH}:
                    if patch_fn is None:
                        reason = "PATCH_REPAIR_UNAVAILABLE"
                        raise PatchValidationError(reason)
                    if repair_class == RepairClass.CONSTRAINED:
                        if constrained_count >= active_policy.max_constrained_repairs:
                            if active_policy.escalate_constrained_to_semantic:
                                selected = tuple(
                                    ValidationIssue(
                                        code=issue.code,
                                        stage=issue.stage,
                                        path=issue.path,
                                        observed=issue.observed,
                                        expected=issue.expected,
                                        context=issue.context,
                                        repair_class=RepairClass.SEMANTIC_PATCH,
                                        allowed_paths=issue.allowed_paths,
                                        message=issue.message,
                                        severity=issue.severity,
                                    )
                                    for issue in selected
                                )
                                repair_class = RepairClass.SEMANTIC_PATCH
                                strategy = repair_class.value
                            else:
                                reason = "CONSTRAINED_REPAIR_BUDGET_EXHAUSTED"
                                raise PatchValidationError(reason)
                        else:
                            constrained_count += 1
                    if repair_class == RepairClass.SEMANTIC_PATCH:
                        if semantic_count >= active_policy.max_semantic_patches:
                            reason = "SEMANTIC_PATCH_BUDGET_EXHAUSTED"
                            raise PatchValidationError(reason)
                        semantic_count += 1
                        emit("semantic_revision", source="validated_generation_semantic_patch")
                    patch_result = patch_fn(deepcopy(artifact), selected, repair_class)
                    if isinstance(patch_result, CandidateSubmission):
                        candidate = deepcopy(patch_result.candidate)
                        proposed_paths = tuple(patch_result.changed_targets)
                        accepted_paths = tuple(patch_result.changed_targets)
                        patch_contract_fingerprint = (
                            patch_result.contract_fingerprint
                        )
                    elif isinstance(patch_result, PatchSubmission):
                        authorization = patch_result.authorization
                        patch_response = patch_result.response
                        declared_paths = set(_allowed_paths(before_issues))
                        undeclared = set(authorization.paths) - declared_paths
                        if undeclared:
                            raise PatchValidationError(
                                "patch authorization contains targets not declared by "
                                "current validation issues: "
                                + ", ".join(sorted(undeclared))
                            )
                    else:
                        patch_response = patch_result
                        authorization = build_patch_authorization(
                            artifact,
                            _allowed_paths(selected),
                            immutable_paths=active_policy.immutable_paths,
                        )
                    if not isinstance(patch_result, CandidateSubmission):
                        patch_contract_fingerprint = authorization.fingerprint
                        operations = parse_patch_operations(patch_response)
                        proposed_paths = tuple(operation.path for operation in operations)
                        proposed_operations = tuple(
                            operation.to_dict() for operation in operations
                        )
                        candidate = apply_patch_atomic(
                            artifact,
                            operations,
                            authorization=authorization,
                            immutable_paths=active_policy.immutable_paths,
                            reject_full_artifact_patch=active_policy.reject_full_artifact_patch,
                        )
                        accepted_paths = tuple(operation.path for operation in operations)
                else:
                    reason = "UNSUPPORTED_REPAIR_CLASS"
                    raise PatchValidationError(reason)
            except (PatchValidationError, KeyError, IndexError, TypeError, ValueError) as exc:
                reason = reason or "REPAIR_REJECTED"
                repair_error_code = _patch_error_code(exc)
                attempts.append(
                    AttemptRecord(
                        attempt=len(attempts) + 1,
                        repair_strategy=strategy,
                        artifact_hash_before=artifact_hash(before),
                        artifact_hash_after=artifact_hash(before),
                        issue_fingerprint_before=issue_fingerprint(before_issues),
                        issue_fingerprint_after=issue_fingerprint(before_issues),
                        issue_count_before=issue_count_by_severity(before_issues),
                        issue_count_after=issue_count_by_severity(before_issues),
                        issue_weight_before=issue_weight_by_severity(before_issues),
                        issue_weight_after=issue_weight_by_severity(before_issues),
                        proposed_patch_paths=proposed_paths,
                        proposed_patch_operations=proposed_operations,
                        accepted_patch_paths=accepted_paths,
                        repair_error=str(exc),
                        repair_error_code=repair_error_code,
                        repair_stage=(
                            "deterministic_repair"
                            if repair_class == RepairClass.DETERMINISTIC
                            else "patch_contract"
                        ),
                        patch_contract_fingerprint=patch_contract_fingerprint,
                        progress=False,
                        accepted=False,
                        reason_code=reason,
                        issues=tuple(
                            [issue.to_dict() for issue in before_issues]
                            + [{"repair_error": str(exc)}]
                        ),
                    )
                )
                status = "internal_error" if repair_class == RepairClass.DETERMINISTIC else "blocked"
                code = "INTERNAL_DETERMINISTIC_REPAIR_BUG" if repair_class == RepairClass.DETERMINISTIC else reason
                result = self._result(
                    status, code, artifact, issues, attempts, active_policy, prompt_version,
                )
                return _raise_or_return(result, raise_on_failure)

            candidate, canonical_issues = _canonicalize(candidate, canonicalize_fn)
            candidate_issues = _collect_issues(canonical_issues, validate_fn(candidate))
            if feasibility_fn:
                candidate_issues = _collect_issues(
                    candidate_issues,
                    feasibility_fn(candidate, candidate_issues),
                )
            issue_delta = classify_issue_delta(
                before,
                candidate,
                before_issues,
                candidate_issues,
            )
            progress = repair_made_progress(before, candidate, before_issues, candidate_issues)
            accepted = progress or not candidate_issues
            rejected_reason = (
                "NO_PROGRESS"
                if artifact_hash(before) == artifact_hash(candidate)
                else "REGRESSION_ROLLED_BACK"
            )
            attempts.append(
                AttemptRecord(
                    attempt=len(attempts) + 1,
                    repair_strategy=strategy,
                    artifact_hash_before=artifact_hash(before),
                    artifact_hash_after=artifact_hash(candidate),
                    issue_fingerprint_before=issue_fingerprint(before_issues),
                    issue_fingerprint_after=issue_fingerprint(candidate_issues),
                    issue_count_before=issue_count_by_severity(before_issues),
                    issue_count_after=issue_count_by_severity(candidate_issues),
                    issue_weight_before=issue_weight_by_severity(before_issues),
                    issue_weight_after=issue_weight_by_severity(candidate_issues),
                    proposed_patch_paths=proposed_paths,
                    proposed_patch_operations=proposed_operations,
                    accepted_patch_paths=accepted_paths if accepted else (),
                    patch_contract_fingerprint=patch_contract_fingerprint,
                    resolved_issues=issue_delta["resolved"],
                    introduced_issues=issue_delta["introduced"],
                    newly_exposed_issues=issue_delta["newly_exposed"],
                    progress=progress,
                    accepted=accepted,
                    reason_code=(
                        "VALID" if not candidate_issues else
                        "PROGRESS" if progress else rejected_reason
                    ),
                    issues=tuple(issue.to_dict() for issue in candidate_issues),
                )
            )
            repair_count += 1
            if accepted:
                artifact = candidate
                issues = candidate_issues
                if not issues:
                    return self._result(
                        "succeeded", "REPAIRED_AND_VALIDATED", artifact, issues,
                        attempts, active_policy, prompt_version,
                    )
                continue

            # Candidate is rejected atomically; the original artifact remains current.
            result = self._result(
                "blocked", rejected_reason, artifact, issues, attempts,
                active_policy, prompt_version,
                rejected_candidate={
                    "reason_code": rejected_reason,
                    "artifact_hash": artifact_hash(candidate),
                    "issue_fingerprint": issue_fingerprint(candidate_issues),
                    "issues": [
                        issue.to_dict() for issue in candidate_issues
                    ],
                    "proposed_patch_paths": list(proposed_paths),
                    "proposed_patch_operations": [
                        _json_value(item) for item in proposed_operations
                    ],
                    "resolved_issues": list(issue_delta["resolved"]),
                    "introduced_issues": list(issue_delta["introduced"]),
                    "newly_exposed_issues": list(issue_delta["newly_exposed"]),
                },
            )
            return _raise_or_return(result, raise_on_failure)

        result = self._result(
            "blocked", "REPAIR_BUDGET_EXHAUSTED", artifact, issues, attempts,
            active_policy, prompt_version,
        )
        return _raise_or_return(result, raise_on_failure)

    @staticmethod
    def _result(
        status: str,
        reason_code: str,
        artifact: Any,
        issues: Sequence[ValidationIssue],
        attempts: Sequence[AttemptRecord],
        policy: RepairPolicy,
        prompt_version: str,
        rejected_candidate: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            status=status,
            reason_code=reason_code,
            artifact=artifact,
            issues=tuple(issues),
            attempts=tuple(attempts),
            contract_version=policy.contract_version,
            canonicalizer_version=policy.canonicalizer_version,
            validator_version=policy.validator_version,
            repair_policy_version=policy.repair_policy_version,
            prompt_version=prompt_version,
            artifact_hash=artifact_hash(artifact),
            issue_fingerprint=issue_fingerprint(issues),
            rejected_candidate=dict(rejected_candidate or {}),
        )


def parse_json_pointer(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"invalid JSON Pointer: {pointer!r}")
    return tuple(
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer[1:].split("/")
    )


def json_pointer(*tokens: object) -> str:
    if not tokens:
        return ""
    return "/" + "/".join(
        str(token).replace("~", "~0").replace("/", "~1") for token in tokens
    )


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    current = document
    for token in parse_json_pointer(pointer):
        if isinstance(current, Mapping):
            current = current[token]
        elif isinstance(current, (list, tuple)):
            current = current[int(token)]
        else:
            raise KeyError(pointer)
    return current


def json_pointer_exists(document: Any, pointer: str) -> bool:
    try:
        resolve_json_pointer(document, pointer)
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return True


def patch_path_states(
    document: Any,
    paths: Sequence[str] | PatchAuthorization,
) -> tuple[Mapping[str, Any], ...]:
    """Describe exact target existence from the same JSON Pointer implementation."""

    if isinstance(paths, PatchAuthorization):
        return tuple(paths.to_prompt_payload())

    states: list[Mapping[str, Any]] = []
    for path in sorted({str(path) for path in paths}):
        exists = json_pointer_exists(document, path)
        states.append(
            {
                "path": path,
                "exists": exists,
                "allowed_operations": (
                    ["replace", "remove"] if exists else ["add"]
                ),
            }
        )
    return tuple(states)


def parse_patch_operations(value: Any) -> tuple[PatchOperation, ...]:
    if not isinstance(value, Mapping) or set(value) != {"operations"}:
        raise PatchValidationError(
            "patch response must be an object containing operations only"
        )
    value = value["operations"]
    if not isinstance(value, list):
        raise PatchValidationError("patch operations must be an array")
    if not value:
        raise PatchValidationError("patch operations must be non-empty objects")
    operations: list[PatchOperation] = []
    for item in value:
        if isinstance(item, PatchOperation):
            operation = item
        elif isinstance(item, Mapping):
            operation = PatchOperation.from_mapping(item)
        else:
            raise PatchValidationError("patch operations must be objects")
        operations.append(operation)
    return tuple(operations)


def apply_patch_atomic(
    artifact: Any,
    operations: Sequence[PatchOperation],
    *,
    allowed_paths: Sequence[str] = (),
    authorization: PatchAuthorization | None = None,
    immutable_paths: Sequence[str] = (),
    reject_full_artifact_patch: bool = True,
) -> Any:
    """Apply a small RFC-6902 subset to a copy after all authorization checks."""

    if authorization is None:
        if not allowed_paths:
            raise PatchValidationError("no allowed paths were supplied")
        authorization = build_patch_authorization(
            artifact,
            allowed_paths,
            immutable_paths=immutable_paths,
        )
    candidate = deepcopy(artifact)
    seen: set[str] = set()
    for operation in operations:
        if operation.op not in {"add", "replace", "remove"}:
            raise PatchValidationError("unsupported patch operation: " + operation.op)
        parse_json_pointer(operation.path)
        if reject_full_artifact_patch and operation.path == "":
            raise PatchValidationError("full artifact replacement is forbidden")
        if operation.path in seen:
            raise PatchValidationError("a patch path may only be modified once")
        seen.add(operation.path)
        authorization.require_operation(operation)
        for immutable_path in immutable_paths:
            if not _paths_overlap(operation.path, immutable_path):
                continue
            # A missing backend-derived subtree may be introduced by a shape
            # repair; canonicalization will populate its authoritative value.
            # Existing immutable data can never be replaced through a parent.
            try:
                resolve_json_pointer(artifact, immutable_path)
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            raise PatchValidationError("patch overlaps immutable path: " + operation.path)
    for operation in operations:
        try:
            _apply_one(candidate, operation)
        except PatchValidationError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise PatchValidationError(
                f"{operation.op} target cannot be resolved: {operation.path}"
            ) from exc
    return candidate


def diff_as_patch(before: Any, after: Any, path: str = "") -> tuple[PatchOperation, ...]:
    """Create field-level operations; useful for adapting legacy full responses safely."""

    if type(before) is not type(after):
        return (PatchOperation("replace", path, deepcopy(after)),)
    if isinstance(before, Mapping):
        operations: list[PatchOperation] = []
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys - after_keys, key=str):
            operations.append(PatchOperation("remove", json_pointer(*parse_json_pointer(path), key)))
        for key in sorted(after_keys - before_keys, key=str):
            operations.append(PatchOperation("add", json_pointer(*parse_json_pointer(path), key), deepcopy(after[key])))
        for key in sorted(before_keys & after_keys, key=str):
            child = json_pointer(*parse_json_pointer(path), key)
            operations.extend(diff_as_patch(before[key], after[key], child))
        return tuple(operations)
    if isinstance(before, list):
        if len(before) != len(after):
            return (PatchOperation("replace", path, deepcopy(after)),)
        operations = []
        for index, (left, right) in enumerate(zip(before, after)):
            operations.extend(diff_as_patch(left, right, json_pointer(*parse_json_pointer(path), index)))
        return tuple(operations)
    if before != after:
        return (PatchOperation("replace", path, deepcopy(after)),)
    return ()


def artifact_hash(artifact: Any) -> str:
    normalized = _without_artifact_hash(_json_value(artifact))
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def issue_fingerprint(issues: Iterable[ValidationIssue]) -> str:
    rows = sorted(
        canonical_json(
            {
                "code": issue.code,
                "path": issue.path,
                "observed": issue.observed,
            }
        )
        for issue in issues
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def issue_count_by_severity(issues: Iterable[ValidationIssue]) -> dict[str, int]:
    counts = {severity.name.lower(): 0 for severity in Severity}
    for issue in issues:
        counts[issue.severity.name.lower()] += 1
    return counts


def issue_weight_by_severity(issues: Iterable[ValidationIssue]) -> dict[str, int]:
    """Count atomic violations, not only their grouped issue envelopes."""

    weights = {severity.name.lower(): 0 for severity in Severity}
    for issue in issues:
        raw_weight = issue.context.get("violation_count", 1)
        try:
            weight = max(1, int(raw_weight))
        except (TypeError, ValueError):
            weight = 1
        weights[issue.severity.name.lower()] += weight
    return weights


def classify_issue_delta(
    before_artifact: Any,
    after_artifact: Any,
    before_issues: Sequence[ValidationIssue],
    after_issues: Sequence[ValidationIssue],
) -> dict[str, tuple[str, ...]]:
    """Classify resolved findings and whether new findings touch patched paths.

    This is diagnostic evidence, not permission to accept a regression.  A new
    issue outside every changed subtree is recorded as newly exposed; one that
    overlaps a changed subtree is recorded as introduced by the candidate.
    """

    before_by_identity = {_issue_identity(issue): issue for issue in before_issues}
    after_by_identity = {_issue_identity(issue): issue for issue in after_issues}
    changed_paths = tuple(
        operation.path for operation in diff_as_patch(before_artifact, after_artifact)
    )
    resolved = tuple(
        sorted(
            _issue_label(before_by_identity[identity])
            for identity in before_by_identity.keys() - after_by_identity.keys()
        )
    )
    introduced: list[str] = []
    newly_exposed: list[str] = []
    for identity in sorted(after_by_identity.keys() - before_by_identity.keys()):
        issue = after_by_identity[identity]
        label = _issue_label(issue)
        if any(_paths_overlap(issue.path, path) for path in changed_paths):
            introduced.append(label)
        else:
            newly_exposed.append(label)
    return {
        "resolved": resolved,
        "introduced": tuple(introduced),
        "newly_exposed": tuple(newly_exposed),
    }


def repair_made_progress(
    before_artifact: Any,
    after_artifact: Any,
    before_issues: Sequence[ValidationIssue],
    after_issues: Sequence[ValidationIssue],
) -> bool:
    if artifact_hash(before_artifact) == artifact_hash(after_artifact):
        return False
    before_max = max((issue.severity for issue in before_issues), default=Severity.INFO)
    after_max = max((issue.severity for issue in after_issues), default=Severity.INFO)
    if after_max > before_max:
        return False
    if after_max < before_max:
        return True
    severity_order = ("critical", "error", "warning", "info")
    before_weights = issue_weight_by_severity(before_issues)
    after_weights = issue_weight_by_severity(after_issues)
    before_rank = tuple(before_weights[name] for name in severity_order)
    after_rank = tuple(after_weights[name] for name in severity_order)
    if after_rank < before_rank or len(after_issues) < len(before_issues):
        return True
    if after_rank > before_rank or len(after_issues) > len(before_issues):
        return False

    before_identities = {_issue_identity(issue) for issue in before_issues}
    after_identities = {_issue_identity(issue) for issue in after_issues}
    if not (before_identities - after_identities):
        return False
    introduced = tuple(
        issue
        for issue in after_issues
        if _issue_identity(issue) not in before_identities
    )
    changed_paths = tuple(
        operation.path
        for operation in diff_as_patch(before_artifact, after_artifact)
    )
    if introduced and any(
        any(_paths_overlap(issue.path, changed_path) for changed_path in changed_paths)
        for issue in introduced
    ):
        return False
    return True


def _issue_identity(issue: ValidationIssue) -> tuple[str, str, str]:
    return issue.code, issue.stage, issue.path


def _issue_label(issue: ValidationIssue) -> str:
    return f"{issue.code}@{issue.path}"


def _patch_error_code(exc: Exception) -> str:
    message = str(exc)
    if "patch response must" in message or "patch operations must" in message:
        return "PATCH_ENVELOPE_INVALID"
    if "patch operation requires value" in message or "unexpected fields" in message:
        return "PATCH_OPERATION_INVALID"
    if "must not contain value" in message or "unsupported patch operation" in message:
        return "PATCH_OPERATION_INVALID"
    if (
        "patch path is not allowed" in message
        or "not an exact authorized target" in message
        or "targets not declared" in message
    ):
        return "PATCH_PATH_UNAUTHORIZED"
    if "is not allowed for" in message and "patch operation" in message:
        return "PATCH_OPERATION_UNAUTHORIZED"
    if "overlaps immutable path" in message:
        return "PATCH_IMMUTABLE_OVERLAP"
    if "target does not exist" in message or "target cannot be resolved" in message:
        return "PATCH_TARGET_MISSING"
    if "target parent is not a container" in message:
        return "PATCH_TARGET_INVALID"
    return "PATCH_INVALID"


def _apply_one(document: Any, operation: PatchOperation) -> None:
    tokens = parse_json_pointer(operation.path)
    if not tokens:
        raise PatchValidationError("root patch is forbidden")
    parent = document
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    token = tokens[-1]
    if isinstance(parent, list):
        if operation.op == "add" and token == "-":
            parent.append(deepcopy(operation.value))
        elif operation.op == "add":
            parent.insert(int(token), deepcopy(operation.value))
        elif operation.op == "replace":
            parent[int(token)] = deepcopy(operation.value)
        else:
            del parent[int(token)]
    elif isinstance(parent, dict):
        if operation.op == "replace" and token not in parent:
            raise PatchValidationError(
                "replace target does not exist: " + operation.path
            )
        if operation.op == "remove" and token not in parent:
            raise PatchValidationError(
                "remove target does not exist: " + operation.path
            )
        if operation.op == "remove":
            del parent[token]
        else:
            parent[token] = deepcopy(operation.value)
    else:
        raise PatchValidationError(
            f"{operation.op} target parent is not a container: {operation.path}"
        )


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=canonical_json)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _without_artifact_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_artifact_hash(item)
            for key, item in value.items()
            if key != "artifact_hash"
        }
    if isinstance(value, list):
        return [_without_artifact_hash(item) for item in value]
    return value


def _canonicalize(
    artifact: Any,
    canonicalize_fn: CanonicalizeFn | None,
) -> tuple[Any, tuple[ValidationIssue, ...]]:
    if canonicalize_fn is None:
        return artifact, ()
    result = canonicalize_fn(deepcopy(artifact))
    if isinstance(result, CanonicalizedArtifact):
        return result.artifact, tuple(result.issues)
    return result, ()


def _collect_issues(*groups: Iterable[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    result: list[ValidationIssue] = []
    seen: set[str] = set()
    for group in groups:
        for issue in group:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("validators must return ValidationIssue objects")
            key = canonical_json(issue.to_dict())
            if key not in seen:
                seen.add(key)
                result.append(issue)
    return tuple(result)


def _terminal_reason(issues: Sequence[ValidationIssue]) -> tuple[str, str] | None:
    classes = {issue.repair_class for issue in issues}
    if RepairClass.INTERNAL_BUG in classes:
        return "internal_error", "INTERNAL_BUG"
    if RepairClass.UNSATISFIABLE in classes:
        return "unsatisfiable", "UNSATISFIABLE_GENERATION_CONTRACT"
    if RepairClass.HUMAN_REQUIRED in classes:
        return "human_required", "HUMAN_REVIEW_REQUIRED"
    if RepairClass.TRANSPORT in classes:
        return "internal_error", "TRANSPORT_ISSUE_MISROUTED"
    return None


def _next_repair_class(issues: Sequence[ValidationIssue]) -> RepairClass:
    for repair_class in (
        RepairClass.DETERMINISTIC,
        RepairClass.JSON_SYNTAX,
        RepairClass.CONSTRAINED,
        RepairClass.SEMANTIC_PATCH,
    ):
        if any(issue.repair_class == repair_class for issue in issues):
            return repair_class
    return RepairClass.HUMAN_REQUIRED


def _allowed_paths(issues: Sequence[ValidationIssue]) -> tuple[str, ...]:
    paths = {
        path
        for issue in issues
        for path in (issue.allowed_paths or (issue.path,))
    }
    return tuple(sorted(paths))


def _is_same_or_descendant(path: str, parent: str) -> bool:
    path_tokens = parse_json_pointer(path)
    parent_tokens = parse_json_pointer(parent)
    return path_tokens[: len(parent_tokens)] == parent_tokens


def _paths_overlap(left: str, right: str) -> bool:
    return _is_same_or_descendant(left, right) or _is_same_or_descendant(right, left)


def _no_progress_record(
    attempt: int,
    strategy: str,
    artifact: Any,
    issues: Sequence[ValidationIssue],
) -> AttemptRecord:
    digest = artifact_hash(artifact)
    fingerprint = issue_fingerprint(issues)
    counts = issue_count_by_severity(issues)
    weights = issue_weight_by_severity(issues)
    return AttemptRecord(
        attempt=attempt,
        repair_strategy=strategy,
        artifact_hash_before=digest,
        artifact_hash_after=digest,
        issue_fingerprint_before=fingerprint,
        issue_fingerprint_after=fingerprint,
        issue_count_before=counts,
        issue_count_after=counts,
        issue_weight_before=weights,
        issue_weight_after=weights,
        progress=False,
        accepted=False,
        reason_code="NO_PROGRESS",
        issues=tuple(issue.to_dict() for issue in issues),
    )


def _raise_or_return(result: GenerationResult, raise_on_failure: bool) -> GenerationResult:
    if raise_on_failure and not result.succeeded:
        raise GenerationTerminatedError(result)
    return result


__all__ = [
    "AttemptRecord",
    "CANONICALIZER_VERSION",
    "CandidateSubmission",
    "CONTRACT_VERSION",
    "CanonicalizedArtifact",
    "GenerationResult",
    "GenerationTerminatedError",
    "PATCH_AUTHORIZATION_CONTRACT_VERSION",
    "PatchAuthorization",
    "PatchOperation",
    "PatchSubmission",
    "PatchTarget",
    "PatchValidationError",
    "REPAIR_POLICY_VERSION",
    "RepairClass",
    "RepairPolicy",
    "Severity",
    "VALIDATOR_VERSION",
    "ValidatedGenerationRunner",
    "ValidationIssue",
    "apply_patch_atomic",
    "artifact_hash",
    "build_patch_authorization",
    "canonical_json",
    "diff_as_patch",
    "issue_count_by_severity",
    "issue_weight_by_severity",
    "issue_fingerprint",
    "json_pointer_exists",
    "json_pointer",
    "parse_json_pointer",
    "parse_patch_operations",
    "patch_path_states",
    "repair_made_progress",
    "resolve_json_pointer",
]

"""Canonical candidate persistence and the sole trace-ready promotion gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping

from contract_versions import (
    AUTHORITATIVE_TRACE_SESSION_CONTRACT_VERSION,
    SEMANTIC_PROMOTION_CONTRACT_VERSION,
)
from failure_routing import failure_route_for_code
from semantic_feasibility import (
    SemanticFeasibilityReport,
    audit_semantic_feasibility,
)
from semantic_policy_binding import (
    SemanticPolicyBindingError,
    SemanticPolicyTargetBinding,
    validate_semantic_policy_target_binding,
)
from semantic_readiness import (
    CAN_GENERATE_AUTHORITATIVE_TRACES,
    SemanticReadinessReport,
    audit_semantic_readiness,
)
from semantic_source import validate_semantic_source_binding


TRACE_READY = "trace_ready"


@dataclass(frozen=True)
class SemanticPromotionOutcome:
    disposition: str
    requirement_id: str
    ir_version: str
    candidate_id: str
    candidate_path: str
    trace_ready_path: str
    readiness: SemanticReadinessReport
    feasibility: SemanticFeasibilityReport | None
    reason_code: str
    message: str
    promotion_fingerprint: str
    capabilities: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()
    unresolved_axis_ids: tuple[str, ...] = ()
    policy_binding_ids: tuple[str, ...] = ()
    grounding_fingerprint: str = ""
    readiness_fingerprint: str = ""
    feasibility_fingerprint: str = ""
    next_stage: str = ""
    expected_pending: bool = False
    retryable: bool = False
    contract_version: str = SEMANTIC_PROMOTION_CONTRACT_VERSION

    @property
    def route(self) -> str:
        return self.disposition


@dataclass(frozen=True)
class AuthoritativeTraceSession:
    session_id: str
    requirement_id: str
    ir_version: str
    candidate_id: str
    candidate_path: str
    trace_ready_path: str
    readiness_fingerprint: str
    feasibility_fingerprint: str
    ap_grounding_fingerprint: str
    policy_binding_fingerprints: tuple[str, ...]
    obligation_plan_fingerprint: str
    promotion_fingerprint: str
    contract_version: str = AUTHORITATIVE_TRACE_SESSION_CONTRACT_VERSION


def promote_semantic_candidate(
    semantic: object,
    *,
    artifact_root: str,
) -> SemanticPromotionOutcome:
    ir = getattr(semantic, "faithful_ir", None)
    if ir is None:
        raise ValueError("semantic promotion requires FaithfulSemanticIR")
    root = Path(artifact_root)
    candidate_payload = _semantic_payload(semantic, promotion_status="candidate")
    candidate_hash = _fingerprint(candidate_payload)
    candidate_id = "SEMCAND-" + candidate_hash[:20]
    requirement_path_key = _requirement_path_component(ir.requirement_id)
    candidate_path = (
        root
        / "semantic_candidates"
        / requirement_path_key
        / f"{candidate_id}.json"
    )
    _atomic_write_json(
        candidate_path,
        {
            "candidate_id": candidate_id,
            "artifact_hash": candidate_hash,
            **candidate_payload,
        },
    )
    _write_current_pointer(
        root / "semantic_candidates" / "current" / f"{requirement_path_key}.json",
        artifact_path=candidate_path,
        artifact_hash=candidate_hash,
        requirement_id=ir.requirement_id,
        source_hash=ir.source_hash,
        ir_version=ir.ir_version,
        contract_fingerprints={
            "grounding": ir.ap_grounding_fingerprint,
            "node_contract": ir.node_contract_fingerprint,
            "semantic_source": ir.effective_source_fingerprint,
        },
    )

    readiness = audit_semantic_readiness(ir)
    try:
        validate_semantic_source_binding(semantic)
    except ValueError as exc:
        return _need_view_outcome(
            root,
            ir,
            candidate_id,
            candidate_path,
            readiness,
            None,
            "EFFECTIVE_SEMANTIC_SOURCE_BINDING_INVALID",
            str(exc),
        )
    binding_fingerprints: list[str] = []
    try:
        for raw_binding in getattr(
            semantic, "semantic_policy_target_bindings", ()
        ):
            if not isinstance(raw_binding, Mapping):
                raise SemanticPolicyBindingError(
                    "persisted policy target binding is malformed"
                )
            binding = _binding_from_mapping(raw_binding)
            validate_semantic_policy_target_binding(
                binding,
                ir,
                require_seed_version=False,
            )
            binding_fingerprints.append(binding.binding_fingerprint)
            binding_path = (
                root
                / "policy_target_bindings"
                / requirement_path_key
                / f"{binding.binding_id}.json"
            )
            _atomic_write_json(binding_path, asdict(binding))
    except SemanticPolicyBindingError as exc:
        return _need_view_outcome(
            root,
            ir,
            candidate_id,
            candidate_path,
            readiness,
            None,
            "SEMANTIC_POLICY_TARGET_BINDING_INVALID",
            str(exc),
        )

    if not readiness.has_capability(CAN_GENERATE_AUTHORITATIVE_TRACES):
        return _need_view_outcome(
            root,
            ir,
            candidate_id,
            candidate_path,
            readiness,
            None,
            "SEMANTIC_READINESS_PENDING",
            "; ".join(
                readiness.messages_for(
                    CAN_GENERATE_AUTHORITATIVE_TRACES
                )
            )
            or "semantic candidate lacks authoritative trace capability",
        )

    feasibility = audit_semantic_feasibility(
        ir,
        policy_binding_fingerprints=tuple(binding_fingerprints),
    )
    feasibility_path = (
        root
        / "feasibility_reports"
        / requirement_path_key
        / f"{candidate_id}.json"
    )
    _atomic_write_json(feasibility_path, asdict(feasibility))
    if not feasibility.trace_ready:
        code = (
            "IR_ROOT_RELATION_ONE_SIDED_IN_DOMAIN"
            if feasibility.relation_classification
            in {"positive_only_in_domain", "negative_only_in_domain"}
            else "IR_WITNESS_RESOURCE_LIMIT"
            if feasibility.relation_classification == "unknown_resource_limit"
            else "EXECUTOR_UNSUPPORTED"
            if feasibility.relation_classification == "unsupported_semantics"
            else "IR_WITNESS_UNSAT_IN_DOMAIN"
            if feasibility.relation_classification == "bounded_unsat"
            else "SEMANTIC_FEASIBILITY_STALE_CONTRACT"
            if feasibility.relation_classification == "stale_contract"
            else "SEMANTIC_FEASIBILITY_INVALID"
        )
        return _need_view_outcome(
            root,
            ir,
            candidate_id,
            candidate_path,
            readiness,
            feasibility,
            code,
            "semantic candidate is not eligible for balanced authoritative traces",
        )

    promotion_payload = {
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "requirement_id": ir.requirement_id,
        "source_hash": ir.source_hash,
        "ir_version": ir.ir_version,
        "readiness_fingerprint": readiness.fingerprint,
        "feasibility_fingerprint": feasibility.fingerprint,
        "ap_grounding_fingerprint": ir.ap_grounding_fingerprint,
        "policy_binding_fingerprints": binding_fingerprints,
        "obligation_plan_fingerprint": (
            feasibility.obligation_plan_fingerprint
        ),
        "contract_version": SEMANTIC_PROMOTION_CONTRACT_VERSION,
    }
    promotion_fingerprint = _fingerprint(promotion_payload)
    trace_ready_path = (
        root
        / "semantic_trace_ready"
        / requirement_path_key
        / f"{candidate_id}.json"
    )
    _atomic_write_json(
        trace_ready_path,
        {
            "promotion_status": TRACE_READY,
            "promotion_fingerprint": promotion_fingerprint,
            "feasibility_report_path": str(feasibility_path),
            **promotion_payload,
            "semantic": asdict(semantic),
        },
    )
    _write_current_pointer(
        root / "semantic_trace_ready" / "current" / f"{requirement_path_key}.json",
        artifact_path=trace_ready_path,
        artifact_hash=_fingerprint(_read_json_mapping(trace_ready_path)),
        requirement_id=ir.requirement_id,
        source_hash=ir.source_hash,
        ir_version=ir.ir_version,
        contract_fingerprints={
            "promotion": promotion_fingerprint,
            "readiness": readiness.fingerprint,
            "feasibility": feasibility.fingerprint,
            "grounding": ir.ap_grounding_fingerprint,
        },
    )
    return SemanticPromotionOutcome(
        disposition=TRACE_READY,
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        candidate_id=candidate_id,
        candidate_path=str(candidate_path),
        trace_ready_path=str(trace_ready_path),
        readiness=readiness,
        feasibility=feasibility,
        reason_code="SEMANTIC_TRACE_READY",
        message="semantic candidate passed readiness and feasibility promotion",
        promotion_fingerprint=promotion_fingerprint,
        capabilities=readiness.capabilities,
        finding_codes=tuple(
            dict.fromkeys(item.code for item in readiness.findings)
        ),
        unresolved_axis_ids=readiness.unresolved_ambiguity_ids,
        policy_binding_ids=tuple(
            str(item.get("binding_id", ""))
            for item in getattr(
                semantic,
                "semantic_policy_target_bindings",
                (),
            )
            if isinstance(item, Mapping) and item.get("binding_id")
        ),
        grounding_fingerprint=ir.ap_grounding_fingerprint,
        readiness_fingerprint=readiness.fingerprint,
        feasibility_fingerprint=feasibility.fingerprint,
        next_stage="authoritative_trace",
    )


def create_authoritative_trace_session(
    outcome: SemanticPromotionOutcome,
) -> AuthoritativeTraceSession:
    if outcome.disposition != TRACE_READY or outcome.feasibility is None:
        raise ValueError("authoritative trace session requires trace-ready promotion")
    _validate_promotion_artifacts(outcome)
    payload = {
        "requirement_id": outcome.requirement_id,
        "ir_version": outcome.ir_version,
        "candidate_id": outcome.candidate_id,
        "readiness_fingerprint": outcome.readiness.fingerprint,
        "feasibility_fingerprint": outcome.feasibility.fingerprint,
        "obligation_plan_fingerprint": (
            outcome.feasibility.obligation_plan_fingerprint
        ),
        "promotion_fingerprint": outcome.promotion_fingerprint,
        "contract_version": AUTHORITATIVE_TRACE_SESSION_CONTRACT_VERSION,
    }
    session_id = "ATRACE-" + _fingerprint(payload)[:20]
    return AuthoritativeTraceSession(
        session_id=session_id,
        requirement_id=outcome.requirement_id,
        ir_version=outcome.ir_version,
        candidate_id=outcome.candidate_id,
        candidate_path=outcome.candidate_path,
        trace_ready_path=outcome.trace_ready_path,
        readiness_fingerprint=outcome.readiness.fingerprint,
        feasibility_fingerprint=outcome.feasibility.fingerprint,
        ap_grounding_fingerprint=(
            outcome.feasibility.ap_grounding_fingerprint
        ),
        policy_binding_fingerprints=(
            outcome.feasibility.policy_binding_fingerprints
        ),
        obligation_plan_fingerprint=(
            outcome.feasibility.obligation_plan_fingerprint
        ),
        promotion_fingerprint=outcome.promotion_fingerprint,
    )


def validate_authoritative_trace_session(
    session: AuthoritativeTraceSession,
    semantic: object,
) -> None:
    if session.contract_version != AUTHORITATIVE_TRACE_SESSION_CONTRACT_VERSION:
        raise ValueError("unsupported authoritative trace session contract")
    ir = getattr(semantic, "faithful_ir", None)
    if (
        ir is None
        or ir.requirement_id != session.requirement_id
        or ir.ir_version != session.ir_version
        or ir.ap_grounding_fingerprint != session.ap_grounding_fingerprint
    ):
        raise ValueError("authoritative trace session semantic lineage is stale")
    candidate = _read_json_mapping(Path(session.candidate_path))
    trace_ready = _read_json_mapping(Path(session.trace_ready_path))
    expected_candidate_payload = _semantic_payload(
        semantic,
        promotion_status="candidate",
    )
    expected_candidate_hash = _fingerprint(expected_candidate_payload)
    if (
        candidate.get("candidate_id") != session.candidate_id
        or candidate.get("artifact_hash") != expected_candidate_hash
        or _fingerprint(_candidate_body(candidate)) != expected_candidate_hash
    ):
        raise ValueError("authoritative trace session candidate artifact is stale")
    expected_trace_fields = {
        "promotion_status": TRACE_READY,
        "candidate_id": session.candidate_id,
        "candidate_hash": expected_candidate_hash,
        "requirement_id": session.requirement_id,
        "ir_version": session.ir_version,
        "readiness_fingerprint": session.readiness_fingerprint,
        "feasibility_fingerprint": session.feasibility_fingerprint,
        "ap_grounding_fingerprint": session.ap_grounding_fingerprint,
        "policy_binding_fingerprints": list(
            session.policy_binding_fingerprints
        ),
        "obligation_plan_fingerprint": (
            session.obligation_plan_fingerprint
        ),
        "promotion_fingerprint": session.promotion_fingerprint,
    }
    if any(trace_ready.get(key) != value for key, value in expected_trace_fields.items()):
        raise ValueError("authoritative trace session trace-ready artifact is stale")
    if _fingerprint(trace_ready.get("semantic")) != _fingerprint(asdict(semantic)):
        raise ValueError("authoritative trace session semantic artifact is stale")
    promotion_payload = _promotion_body(trace_ready)
    if _fingerprint(promotion_payload) != session.promotion_fingerprint:
        raise ValueError("authoritative trace session promotion proof is invalid")


def _need_view_outcome(
    root: Path,
    ir: object,
    candidate_id: str,
    candidate_path: Path,
    readiness: SemanticReadinessReport,
    feasibility: SemanticFeasibilityReport | None,
    reason_code: str,
    message: str,
) -> SemanticPromotionOutcome:
    route = failure_route_for_code(reason_code)
    payload = {
        "status": route.disposition,
        "finalization_allowed": False,
        "requirement_id": ir.requirement_id,
        "source_hash": ir.source_hash,
        "ir_version": ir.ir_version,
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path),
        "reason_code": reason_code,
        "message": message,
        "readiness": asdict(readiness),
        "feasibility": asdict(feasibility) if feasibility is not None else None,
        "contract_version": SEMANTIC_PROMOTION_CONTRACT_VERSION,
    }
    fingerprint = _fingerprint(payload)
    path = (
        root
        / "semantic_need_view"
        / _requirement_path_component(ir.requirement_id)
        / f"{candidate_id}.json"
    )
    _atomic_write_json(path, {**payload, "promotion_fingerprint": fingerprint})
    return SemanticPromotionOutcome(
        disposition=route.disposition,
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        candidate_id=candidate_id,
        candidate_path=str(candidate_path),
        trace_ready_path="",
        readiness=readiness,
        feasibility=feasibility,
        reason_code=reason_code,
        message=message,
        promotion_fingerprint=fingerprint,
        capabilities=readiness.capabilities,
        finding_codes=tuple(
            dict.fromkeys(
                (
                    *(item.code for item in readiness.findings),
                    reason_code,
                )
            )
        ),
        unresolved_axis_ids=readiness.unresolved_ambiguity_ids,
        grounding_fingerprint=str(
            getattr(ir, "ap_grounding_fingerprint", "")
        ),
        readiness_fingerprint=readiness.fingerprint,
        feasibility_fingerprint=(
            feasibility.fingerprint if feasibility is not None else ""
        ),
        next_stage=route.safe_restart_stage,
        expected_pending=route.expected_pending,
        retryable=route.retryable,
    )


def _binding_from_mapping(
    mapping: Mapping[str, Any],
) -> SemanticPolicyTargetBinding:
    values = dict(mapping)
    for name in (
        "source_span",
        "supporting_segment_ids",
        "controlled_parameter_names",
    ):
        values[name] = tuple(values.get(name, ()))
    return SemanticPolicyTargetBinding(**values)


def _validate_promotion_artifacts(outcome: SemanticPromotionOutcome) -> None:
    candidate = _read_json_mapping(Path(outcome.candidate_path))
    trace_ready = _read_json_mapping(Path(outcome.trace_ready_path))
    candidate_hash = str(candidate.get("artifact_hash", ""))
    if (
        candidate.get("candidate_id") != outcome.candidate_id
        or not candidate_hash
        or _fingerprint(_candidate_body(candidate)) != candidate_hash
    ):
        raise ValueError("semantic candidate artifact content is invalid")
    if (
        trace_ready.get("promotion_status") != TRACE_READY
        or trace_ready.get("candidate_id") != outcome.candidate_id
        or trace_ready.get("candidate_hash") != candidate_hash
        or trace_ready.get("requirement_id") != outcome.requirement_id
        or trace_ready.get("ir_version") != outcome.ir_version
        or trace_ready.get("readiness_fingerprint")
        != outcome.readiness.fingerprint
        or trace_ready.get("feasibility_fingerprint")
        != outcome.feasibility.fingerprint
        or trace_ready.get("promotion_fingerprint")
        != outcome.promotion_fingerprint
    ):
        raise ValueError("semantic trace-ready artifact content is invalid")
    if _fingerprint(_promotion_body(trace_ready)) != outcome.promotion_fingerprint:
        raise ValueError("semantic trace-ready promotion proof is invalid")


def _candidate_body(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in candidate.items()
        if key not in {"candidate_id", "artifact_hash"}
    }


def _promotion_body(trace_ready: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id",
        "candidate_hash",
        "requirement_id",
        "source_hash",
        "ir_version",
        "readiness_fingerprint",
        "feasibility_fingerprint",
        "ap_grounding_fingerprint",
        "policy_binding_fingerprints",
        "obligation_plan_fingerprint",
        "contract_version",
    )
    return {key: trace_ready.get(key) for key in keys}


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"authoritative trace artifact cannot be read: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"authoritative trace artifact is not an object: {path}")
    return payload


def _semantic_payload(
    semantic: object,
    *,
    promotion_status: str,
) -> dict[str, object]:
    ir = getattr(semantic, "faithful_ir")
    return {
        "promotion_status": promotion_status,
        "requirement_id": ir.requirement_id,
        "source_hash": ir.source_hash,
        "ir_version": ir.ir_version,
        "ap_grounding_fingerprint": ir.ap_grounding_fingerprint,
        "node_contract_fingerprint": ir.node_contract_fingerprint,
        "temporal_language_contract_version": (
            ir.temporal_language_contract_version
        ),
        "semantic_source_contract_version": (
            ir.semantic_source_contract_version
        ),
        "effective_source_fingerprint": ir.effective_source_fingerprint,
        "effective_source_kind": ir.effective_source_kind,
        "effective_source_authority_record_id": (
            ir.effective_source_authority_record_id
        ),
        "semantic": asdict(semantic),
        "contract_version": SEMANTIC_PROMOTION_CONTRACT_VERSION,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            ))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 5:
                    exc.operation = "semantic_promotion_atomic_replace"
                    exc.source_path_length = len(str(temporary.resolve()))
                    exc.target_path_length = len(str(path.resolve()))
                    raise
                time.sleep(.01 * 2 ** attempt)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_current_pointer(
    path: Path,
    *,
    artifact_path: Path,
    artifact_hash: str,
    requirement_id: str,
    source_hash: str,
    ir_version: str,
    contract_fingerprints: Mapping[str, str],
) -> None:
    if not artifact_path.is_file():
        raise ValueError("current pointer target does not exist")
    payload = {
        "requirement_id": requirement_id,
        "artifact_path": str(artifact_path),
        "artifact_hash": artifact_hash,
        "source_hash": source_hash,
        "ir_version": ir_version,
        "contract_fingerprints": {
            str(key): str(value)
            for key, value in contract_fingerprints.items()
            if value
        },
        "contract_version": SEMANTIC_PROMOTION_CONTRACT_VERSION,
    }
    payload["pointer_fingerprint"] = _fingerprint(payload)
    _atomic_write_json(path, payload)


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _requirement_path_component(requirement_id: str) -> str:
    """Return a stable Windows-safe path component for a requirement ID."""
    raw = str(requirement_id)
    slug = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw)
    slug = re.sub(r"\s+", "_", slug).strip(" ._")[:64] or "requirement"
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{suffix}"


__all__ = [
    "TRACE_READY",
    "AuthoritativeTraceSession",
    "SemanticPromotionOutcome",
    "create_authoritative_trace_session",
    "promote_semantic_candidate",
    "validate_authoritative_trace_session",
]

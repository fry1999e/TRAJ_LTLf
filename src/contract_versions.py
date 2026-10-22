"""Authoritative versions for persisted evidence and review contracts.

Keep these values in one dependency-light module.  Checkpoints, RAG chunks, and
load-time filters must agree on the same version; otherwise freshly written
artifacts can be rejected as legacy by another part of the pipeline.
"""

from __future__ import annotations

WORKING_CONTRACT_VERSION = "working-branches-v8-ir-contract-recovery"


# v11 separates unresolved semantic-discrimination plans from executable
# witness plans and binds both to a deterministic semantic-readiness audit.
OBLIGATION_COMPILER_VERSION = 14
# Boundary-evidence source selection is an outer experiment policy; it does
# not mutate the v11 canonical trace contract or persisted warmup gold.
TRACE_CONTRACT_VERSION = 12
# Canonical semantic evidence remains v5.  Policy-effective completeness is
# stored alongside it in run/checkpoint metadata instead of rewriting gold.
SEMANTIC_EVIDENCE_CONTRACT_VERSION = 5
SEMANTIC_READINESS_CONTRACT_VERSION = "semantic-readiness-v2"
TRACE_EVIDENCE_PURPOSE_CONTRACT_VERSION = "trace-evidence-purpose-v1"
IR_AMBIGUITY_LIFECYCLE_CONTRACT_VERSION = "ir-ambiguity-lifecycle-v1"
IR_CANDIDATE_ADJUDICATION_CONTRACT_VERSION = "ir-candidate-adjudication-v1"

# Schema-v6 establishes the typed semantic-axis registry, independent NL
# evidence layer, symbolic AP grounding, and certified compiler lowering as one
# online contract.  Older artifacts are migration input only.
FAITHFUL_IR_SCHEMA_VERSION = "7"
FAITHFUL_IR_CONTRACT_VERSION = "faithful-ir-contract-v8"
FAITHFUL_IR_CANONICALIZER_VERSION = "faithful-ir-canonicalizer-v7"
FAITHFUL_IR_VALIDATOR_VERSION = "faithful-ir-validator-v11"
SEMANTIC_TOPOLOGY_CONTRACT_VERSION = "semantic-topology-contract-v5"
NLU_SEMANTIC_PROMPT_VERSION = "semantic-ir-v12"

# runtime separates reviewable incompleteness from internal failures and makes
# rerun invalidation a typed, versioned protocol.
BATCH_DISPOSITION_CONTRACT_VERSION = "batch-disposition-v1"
FEEDBACK_RERUN_SCHEMA_VERSION = 2
BATCH_OUTPUT_MANIFEST_SCHEMA_VERSION = 3

# runtime keeps AP identity at the pre-IR source layer.  These versions are
# deliberately independent from the FaithfulSemanticIR and trace contracts.
AP_REPRESENTATION_PROFILE_CONTRACT_VERSION = "ap-representation-profile-v1"
EFFECTIVE_SEMANTIC_SOURCE_CONTRACT_VERSION = "effective-semantic-source-v1"
REVIEWED_FORMAL_SOURCE_AUTHORITY_CONTRACT_VERSION = (
    "reviewed-formal-source-authority-v1"
)
AP_BINDING_CONTEXT_CONTRACT_VERSION = "ap-binding-context-v2-casefold"
AP_GROUNDING_CONTRACT_VERSION = "ap-grounding-contract-v2-casefold"
AP_GROUNDING_EXTRACTOR_REGISTRY_VERSION = "ap-grounding-extractors-v1"
PLAIN_READING_REPAIR_PLAN_CONTRACT_VERSION = "plain-reading-repair-plan-v1"
PLAIN_READING_REPAIR_COMPILER_VERSION = "plain-reading-repair-compiler-v1"
PLAIN_READING_LEXICAL_CONTRACT_VERSION = "plain-reading-lexical-v1"
PLAIN_READING_NEED_VIEW_CONTRACT_VERSION = "plain-reading-need-view-v1"
SEMANTIC_GRAPH_REPAIR_CONTRACT_VERSION = "semantic-graph-repair-v1"
TEMPORAL_LANGUAGE_CONTRACT_VERSION = "temporal-language-v1"
SEMANTIC_POLICY_BINDING_CONTRACT_VERSION = "semantic-policy-binding-v1"
SEMANTIC_PROMOTION_CONTRACT_VERSION = "semantic-promotion-v1"
FAILURE_ROUTING_CONTRACT_VERSION = "failure-routing-v1"
SEMANTIC_FEASIBILITY_CONTRACT_VERSION = "semantic-feasibility-v1"
WITNESS_DOMAIN_POLICY_VERSION = "witness-domain-v1"
AUTHORITATIVE_TRACE_SESSION_CONTRACT_VERSION = "authoritative-trace-session-v1"
CHECKPOINT_MIGRATION_CONTRACT_VERSION = "checkpoint-migration-v1"
IR_RELATION_FEASIBILITY_CONTRACT_VERSION = "ir-relation-feasibility-v1"
IR_WITNESS_SOLVER_CONTRACT_VERSION = "ir-witness-solver-v1"
AP_INVENTORY_SCHEMA_VERSION = 1
AP_INVENTORY_CONTRACT_VERSION = "ap-inventory-v1"
AP_INVENTORY_CANONICALIZER_VERSION = "ap-inventory-canonicalizer-v1"
AP_INVENTORY_VALIDATOR_VERSION = "ap-inventory-validator-v1"
OVERRIDE_AP_COMPARISON_CONTRACT_VERSION = "override-ap-identity-v2"
HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION = (
    "human-semantic-override-v1-symbolic-token"
)
HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION = "human-semantic-override-v2"

# runtime's routing decision is independent of the persisted PlainReadingReview
# schema and every downstream Semantic IR/Trace/Compiler contract.
PLAIN_READING_READINESS_CONTRACT_VERSION = "plain-reading-readiness-v1"


def ap_identity_runtime_contract() -> dict[str, object]:
    """Return the complete pre-IR AP identity runtime identity."""

    return {
        "representation_profile_contract_version": (
            AP_REPRESENTATION_PROFILE_CONTRACT_VERSION
        ),
        "binding_context_contract_version": AP_BINDING_CONTEXT_CONTRACT_VERSION,
        "inventory_schema_version": AP_INVENTORY_SCHEMA_VERSION,
        "inventory_contract_version": AP_INVENTORY_CONTRACT_VERSION,
        "inventory_canonicalizer_version": AP_INVENTORY_CANONICALIZER_VERSION,
        "inventory_validator_version": AP_INVENTORY_VALIDATOR_VERSION,
        "override_comparison_contract_version": (
            OVERRIDE_AP_COMPARISON_CONTRACT_VERSION
        ),
        "human_semantic_override_contract_version": (
            HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION
        ),
    }


def faithful_ir_runtime_contract() -> dict[str, object]:
    """Return the complete semantic-runtime identity used by cache gates."""

    return {
        "schema_version": FAITHFUL_IR_SCHEMA_VERSION,
        "contract_version": FAITHFUL_IR_CONTRACT_VERSION,
        "canonicalizer_version": FAITHFUL_IR_CANONICALIZER_VERSION,
        "validator_version": FAITHFUL_IR_VALIDATOR_VERSION,
        "semantic_evidence_contract_version": SEMANTIC_EVIDENCE_CONTRACT_VERSION,
        "semantic_readiness_contract_version": SEMANTIC_READINESS_CONTRACT_VERSION,
        "obligation_compiler_version": OBLIGATION_COMPILER_VERSION,
        "trace_evidence_purpose_contract_version": (
            TRACE_EVIDENCE_PURPOSE_CONTRACT_VERSION
        ),
        "ir_ambiguity_lifecycle_contract_version": (
            IR_AMBIGUITY_LIFECYCLE_CONTRACT_VERSION
        ),
        "ir_candidate_adjudication_contract_version": (
            IR_CANDIDATE_ADJUDICATION_CONTRACT_VERSION
        ),
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "effective_semantic_source_contract_version": (
            EFFECTIVE_SEMANTIC_SOURCE_CONTRACT_VERSION
        ),
        "ap_binding_context_contract_version": AP_BINDING_CONTEXT_CONTRACT_VERSION,
        "ir_relation_feasibility_contract_version": (
            IR_RELATION_FEASIBILITY_CONTRACT_VERSION
        ),
        "ir_witness_solver_contract_version": IR_WITNESS_SOLVER_CONTRACT_VERSION,
    }

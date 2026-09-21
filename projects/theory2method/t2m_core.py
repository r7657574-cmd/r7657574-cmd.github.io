"""Minimal auditable core of a theory-first method-discovery workflow.

The proposer may be a person, a language model, or another search process.  It is
outside the trusted core.  This module records typed claims, freezes candidates,
and applies conjunctive gates: one failed premise cannot be compensated by a
high score elsewhere.

This is a clean public reconstruction, not the private experiment controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json


class Stage(str, Enum):
    CONTRACT = "contract"
    FOUNDATION = "foundation"
    FINITE_OBJECTS = "finite_objects"
    FREEZE = "freeze"
    PRIOR_AUDIT = "prior_audit"
    HEADROOM = "headroom"
    P0 = "p0"
    MATCHED_TRAINING = "matched_training"
    OOD_SCALE = "ood_scale"
    DECISION = "decision"


class Route(str, Enum):
    NUMERICAL = "numerical"
    NEURAL = "neural"
    HYBRID = "hybrid"
    REJECT = "reject"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    HOLD = "hold"
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"


class Decision(str, Enum):
    REJECT = "reject"
    HOLD = "hold"
    AUTHORIZE_P0 = "authorize_p0"
    AUTHORIZE_TRAINING = "authorize_training"
    AUTHORIZE_OOD = "authorize_ood"
    VALIDATED_NUMERICAL = "validated_numerical"
    VALIDATED_NEURAL = "validated_neural"


@dataclass(frozen=True)
class TheoryCard:
    theorem_or_principle: str
    source_ids: tuple[str, ...]
    premises: tuple[str, ...]
    solver_structure: tuple[str, ...]
    unresolved_computation: str
    unresolved_type: str
    falsifier: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    route: Route
    representation: str
    prediction_target: str
    method: str
    validity_conditions: tuple[str, ...]
    matched_comparator: str | None = None
    no_learning_shadow: str | None = None

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Evidence:
    theory_instance_valid: bool | None = None
    finite_object_constructed: bool | None = None
    verifier_passed: bool | None = None
    measurable_headroom: bool | None = None
    prior_art_audited: bool | None = None
    p0_passed: bool | None = None
    matched_comparator_improved: bool | None = None
    ood_or_scale_check_passed: bool | None = None
    falsifier_triggered: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    reason: str


@dataclass
class AuditRecord:
    candidate_id: str
    frozen_fingerprint: str
    current_fingerprint: str
    gates: list[GateResult]
    decision: Decision
    next_stage: Stage
    failure_signature: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def freeze(candidate: Candidate) -> str:
    """Return the immutable comparison key recorded before prior-art audit."""
    return candidate.fingerprint()


def _gate(name: str, value: bool | None, reason_pass: str, reason_fail: str) -> GateResult:
    if value is None:
        return GateResult(name, GateStatus.NOT_RUN, "evidence not recorded")
    if value:
        return GateResult(name, GateStatus.PASS, reason_pass)
    return GateResult(name, GateStatus.FAIL, reason_fail)


def evaluate(
    theory: TheoryCard,
    candidate: Candidate,
    evidence: Evidence,
    frozen_fingerprint: str,
) -> AuditRecord:
    """Evaluate one frozen candidate with ordered, non-compensatory gates."""
    current = candidate.fingerprint()
    gates: list[GateResult] = []

    theory_contract = bool(
        theory.source_ids
        and theory.premises
        and theory.solver_structure
        and theory.unresolved_computation.strip()
        and theory.unresolved_type.strip()
        and theory.falsifier.strip()
    )
    gates.append(_gate(
        "G0_theory",
        theory_contract and evidence.theory_instance_valid,
        "source, premises, structure, and instance check are present",
        "theory contract is incomplete or invalid for this instance",
    ))

    gates.append(_gate(
        "G1_finite_object",
        evidence.finite_object_constructed,
        "solver-facing object was constructed",
        "theory was not grounded into a finite computational object",
    ))
    gates.append(_gate(
        "G2_verifier",
        evidence.verifier_passed,
        "independent verifier passed",
        "derived object failed its verifier",
    ))
    gates.append(_gate(
        "G3_headroom",
        evidence.measurable_headroom,
        "the unresolved computation has measurable headroom",
        "no useful unresolved computation remains",
    ))
    gates.append(_gate(
        "G4_prior_audit",
        evidence.prior_art_audited,
        "post-freeze prior-art audit completed",
        "novelty or comparator context is not audited",
    ))
    gates.append(_gate(
        "G5_p0",
        evidence.p0_passed,
        "small falsifiable pilot passed",
        "the candidate did not survive the pilot",
    ))

    requires_training = candidate.route in {Route.NEURAL, Route.HYBRID}
    if requires_training:
        learned_contract = bool(candidate.matched_comparator and candidate.no_learning_shadow)
        matched_value = (
            evidence.matched_comparator_improved if learned_contract else False
        )
        gates.append(_gate(
            "G6_matched_training",
            matched_value,
            "matched comparator improved and no-learning shadow is specified",
            "learned route lacks a matched improvement or no-learning shadow",
        ))
    else:
        gates.append(GateResult(
            "G6_matched_training",
            GateStatus.NOT_APPLICABLE,
            "candidate contains no learned component",
        ))
    gates.append(_gate(
        "G7_ood_scale",
        evidence.ood_or_scale_check_passed,
        "out-of-distribution or scale check passed",
        "transfer or scaling check failed",
    ))

    failures: list[str] = []
    if current != frozen_fingerprint:
        failures.append("candidate_changed_after_freeze")
    if evidence.falsifier_triggered:
        failures.append("declared_falsifier_triggered")
    failures.extend(result.gate for result in gates if result.status == GateStatus.FAIL)

    not_run = [result.gate for result in gates if result.status in {GateStatus.NOT_RUN, GateStatus.HOLD}]
    gate_map = {result.gate: result.status for result in gates}
    foundation = ("G0_theory", "G1_finite_object", "G2_verifier", "G3_headroom", "G4_prior_audit")

    if candidate.route == Route.REJECT or failures:
        decision = Decision.REJECT
        next_stage = Stage.DECISION
    elif not all(gate_map[name] == GateStatus.PASS for name in foundation):
        decision = Decision.HOLD
        next_stage = Stage.FOUNDATION
    elif gate_map["G5_p0"] != GateStatus.PASS:
        decision = Decision.AUTHORIZE_P0
        next_stage = Stage.P0
    elif requires_training and gate_map["G6_matched_training"] != GateStatus.PASS:
        decision = Decision.AUTHORIZE_TRAINING
        next_stage = Stage.MATCHED_TRAINING
    elif gate_map["G7_ood_scale"] != GateStatus.PASS:
        decision = Decision.AUTHORIZE_OOD
        next_stage = Stage.OOD_SCALE
    else:
        decision = Decision.VALIDATED_NEURAL if requires_training else Decision.VALIDATED_NUMERICAL
        next_stage = Stage.DECISION

    return AuditRecord(
        candidate_id=candidate.candidate_id,
        frozen_fingerprint=frozen_fingerprint,
        current_fingerprint=current,
        gates=gates,
        decision=decision,
        next_stage=next_stage,
        failure_signature=failures + [f"not_run:{name}" for name in not_run],
    )


def compile_revision(record: AuditRecord) -> tuple[str, ...]:
    """Turn an evidenced failure into bounded obligations for a new lineage.

    The compiler does not invent a replacement method.  It states what a future
    proposal must repair while preserving the frozen run as negative evidence.
    """
    obligations: list[str] = []
    signatures = set(record.failure_signature)
    if "candidate_changed_after_freeze" in signatures:
        obligations.append("open a new lineage; never edit the frozen candidate in place")
    if "declared_falsifier_triggered" in signatures:
        obligations.append("preserve the counterexample and revise the invalid theory-to-method link")
    mapping = {
        "G0_theory": "repair the premise match or reject the theory instance",
        "G1_finite_object": "construct a solver-facing finite object before proposing an architecture",
        "G2_verifier": "supply an independent verifier for the derived object",
        "G3_headroom": "demonstrate unresolved computational headroom or close the lineage",
        "G4_prior_audit": "complete the post-freeze prior-art and comparator audit",
        "G5_p0": "replace the failed mechanism with a substantively new frozen proposal",
        "G6_matched_training": "isolate value beyond the matched no-learning shadow",
        "G7_ood_scale": "repair the scale or distribution failure and test on fresh evidence",
    }
    obligations.extend(mapping[name] for name in mapping if name in signatures)
    if obligations:
        obligations.extend((
            "keep parent examples development-only",
            "freeze the revised mechanism and pass rules before new confirmation evidence",
        ))
    return tuple(dict.fromkeys(obligations))

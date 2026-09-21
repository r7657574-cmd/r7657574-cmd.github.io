"""Small example of freezing and auditing a theory-guided candidate."""

import json

from t2m_core import Candidate, Evidence, Route, TheoryCard, evaluate, freeze


theory = TheoryCard(
    theorem_or_principle="A source-grounded structural identity",
    source_ids=("original-source:section",),
    premises=("well-posed parameterized PDE family",),
    solver_structure=("exact transformation", "typed residual"),
    unresolved_computation="estimate the transformed residual",
    unresolved_type="bounded regression target",
    falsifier="the transformation residual exceeds tolerance",
)

candidate = Candidate(
    candidate_id="candidate-001",
    route=Route.HYBRID,
    representation="transformed coordinates plus verified reconstruction",
    prediction_target="only the unresolved residual",
    method="numerical transformation followed by a learned residual model",
    validity_conditions=("all theorem premises pass",),
    matched_comparator="same model and budget without the theory-derived representation",
    no_learning_shadow="transformation with a numerical residual estimate",
)

frozen = freeze(candidate)
evidence = Evidence(
    theory_instance_valid=True,
    finite_object_constructed=True,
    verifier_passed=True,
    measurable_headroom=True,
    prior_art_audited=True,
    p0_passed=True,
    matched_comparator_improved=True,
    ood_or_scale_check_passed=True,
)

print(json.dumps(evaluate(theory, candidate, evidence, frozen).to_dict(), indent=2))

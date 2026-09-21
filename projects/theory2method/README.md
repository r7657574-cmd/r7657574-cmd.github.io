# Theory2Method — public logical core

Theory2Method is a theory-first workflow for turning source-grounded mathematics into a numerical, neural, hybrid, or rejection decision. This directory exposes only a clean reconstruction of its audit logic.

## Trusted boundary

The proposer can be an LLM, a person, symbolic search, or replay. It does not receive execution authority. The trusted core:

1. records a typed theory card with premises, solver-facing structure, an unresolved computation, and a falsifier;
2. freezes a candidate before prior-art comparison;
3. requires route-specific evidence through the ordered `G0`–`G7` gates;
4. derives only the next authorized stage—pilot, matched training, transfer/scale evaluation, or final decision;
5. compiles evidenced failures into bounded obligations for a new lineage instead of editing the frozen parent.

Neural and hybrid routes additionally require a matched comparator and a no-learning shadow. A failed premise cannot be offset by a stronger downstream score.

## Run

```bash
python demo.py
```

The module uses only the Python standard library.

## Deliberately omitted

The manuscript, prompts, provider configuration, paper library, experiment runs, raw traces, private validators, hidden rubrics, and unpublished numerical results are not included.

# Theory2Method: Auditable Multi-Agent Reasoning for Theory-First PDE Solver Design

Theory2Method is a theory-first workflow for turning source-grounded mathematics into a numerical, learned, hybrid, narrowed, or rejection route. This directory exposes only a clean reconstruction of the workflow's logical core.

## Theory-to-method design

The public core represents four parts of the design:

1. search results are recorded with sources and applicability conditions;
2. the selected theory is translated into a representation, theory-fixed relations, preserved properties, a validity boundary, and a typed unresolved computation;
3. the unresolved computation determines the numerical, neural, hybrid, or rejection route;
4. theory-instance validity and method effectiveness are evaluated separately.

The implementation keeps candidate identity and ordered evidence checks as supporting audit machinery. Those checks serve the theory-to-method design; they are not the research objective by themselves. Neural and hybrid routes additionally require a matched comparator and a no-learning shadow, and a failed mathematical premise cannot be offset by a stronger downstream score.

## Run

```bash
python demo.py
```

The module uses only the Python standard library.

## Deliberately omitted

The manuscript, prompts, provider configuration, paper library, experiment runs, raw traces, private validators, hidden rubrics, and unpublished numerical results are not included.

# Multi-Agent Large Language Models for Connectivity Restoration in Damaged Graphs

This directory contains a compact public reconstruction of the project's core graph logic. It is based on the research code's central sequence: inspect the damaged topology, ground user intent in graph objectives, compare a controlled set of repair strategies, and validate the selected edit plan against hard constraints.

## What is included

- `graph_repair_core.py` extracts an explicit graph profile, interprets intent as visible objective weights, builds candidate repairs on graph copies, and returns a validated repair plan.
- `demo.py` runs the core on a small disconnected network.
- `manuscript.pdf` is the paper associated with this project.

The strategy library contains shortest-bridge, component-tree, and latency-oriented repairs. It also shows one bounded extension, `balanced_tree_latency`, formed by composing audited graph operators. The public core never executes model-generated Python.

## Run

```bash
python -m pip install -r requirements.txt
python demo.py
```

## Publication boundary

The repository omits model-provider integration, private prompts, datasets, training and reinforcement-learning code, bulk generated files, raw traces, and the full experiment harness. This is the logical core, not a complete reproduction package.

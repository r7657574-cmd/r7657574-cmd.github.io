"""Run the public graph-repair core on a small damaged network."""

import json

import networkx as nx

from graph_repair_core import restore


positions = {
    "a": (0.0, 0.0),
    "b": (1.0, 0.0),
    "c": (2.0, 0.2),
    "d": (4.0, 0.1),
    "e": (5.0, 0.0),
    "f": (6.0, 0.3),
    "g": (3.0, 2.2),
}

damaged = nx.Graph()
damaged.add_nodes_from(positions)
damaged.add_edges_from(
    [("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")]
)

result = restore(
    damaged,
    positions,
    "Restore connectivity with low construction cost and lower latency",
    forbidden={("c", "d")},
    must_connect={("b", "e")},
    shortcut_budget=1,
    mode="strict",
)

print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

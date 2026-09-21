"""Core logic for intent-aware connectivity restoration in damaged graphs.

The original research system uses language-model agents to interpret intent,
inspect graph structure, and select a repair strategy. This public module keeps
the graph-specific logical center while removing provider calls, private prompts,
datasets, training code, and arbitrary generated-code execution.

The trusted path is deliberately narrow:

1. summarize the damaged graph as explicit structural evidence;
2. translate user intent into inspectable objective weights;
3. construct transparent candidate repairs on graph copies;
4. select a feasible plan and validate every proposed graph edit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot, isfinite
from typing import Hashable, Iterable, Literal

import networkx as nx

Node = Hashable
Edge = tuple[Node, Node]
ConstraintMode = Literal["strict", "relaxed"]


class InfeasibleRepair(RuntimeError):
    """Raised when no repair can satisfy the declared graph constraints."""


def canonical_edge(u: Node, v: Node) -> Edge:
    """Return one deterministic representation for an undirected edge."""
    if u == v:
        raise ValueError("Self-loops are not valid repair edges")
    return (u, v) if repr(u) < repr(v) else (v, u)


def edge_length(
    u: Node,
    v: Node,
    positions: dict[Node, tuple[float, float]],
) -> float:
    """Euclidean construction cost used by the public reference strategies."""
    (x1, y1), (x2, y2) = positions[u], positions[v]
    return hypot(x1 - x2, y1 - y2)


@dataclass(frozen=True)
class GraphProfile:
    """Small structural record shared by intent interpretation and planning."""

    nodes: int
    edges: int
    components: int
    component_sizes: tuple[int, ...]
    isolated_nodes: tuple[Node, ...]
    density: float
    bridges_inside_components: int


@dataclass(frozen=True)
class Intent:
    """Explicit optimization priorities inferred from a short request."""

    connectivity: float = 1.0
    cost: float = 0.35
    latency: float = 0.35
    robustness: float = 0.0

    def normalized(self) -> "Intent":
        values = [
            max(0.0, self.connectivity),
            max(0.0, self.cost),
            max(0.0, self.latency),
            max(0.0, self.robustness),
        ]
        scale = sum(values) or 1.0
        return Intent(*(value / scale for value in values))


@dataclass(frozen=True)
class Constraints:
    """Hard graph-edit constraints and a bounded shortcut allowance."""

    forbidden: frozenset[Edge] = frozenset()
    must_connect: frozenset[Edge] = frozenset()
    max_edge_length: float | None = None
    shortcut_budget: int = 1
    mode: ConstraintMode = "strict"


@dataclass(frozen=True)
class RepairStep:
    operation: str
    edge: Edge
    weight: float
    reason: str


@dataclass
class CandidateEvaluation:
    strategy: str
    feasible: bool
    rationale: str
    utility: float | None = None
    added_edges: int | None = None
    added_length: float | None = None
    weighted_path_length: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class RepairResult:
    graph_profile: GraphProfile
    intent: Intent
    constraints: Constraints
    selected_strategy: str
    selected_rationale: str
    candidates: list[CandidateEvaluation]
    plan: list[RepairStep]
    validation: dict[str, object]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    graph: nx.Graph = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        """Serialize the audit record without serializing the NetworkX object."""
        payload = asdict(self)
        payload.pop("graph", None)
        payload["constraints"]["forbidden"] = sorted(
            (list(edge) for edge in self.constraints.forbidden), key=repr
        )
        payload["constraints"]["must_connect"] = sorted(
            (list(edge) for edge in self.constraints.must_connect), key=repr
        )
        return payload


def analyze_graph(graph: nx.Graph) -> GraphProfile:
    """Extract only graph facts that can be checked against the input object."""
    if graph.is_directed():
        raise ValueError("This public core supports undirected graphs only")
    components = sorted(
        (len(component) for component in nx.connected_components(graph)),
        reverse=True,
    )
    isolated = tuple(sorted(nx.isolates(graph), key=repr))
    return GraphProfile(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        components=len(components),
        component_sizes=tuple(components),
        isolated_nodes=isolated,
        density=float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        bridges_inside_components=sum(1 for _ in nx.bridges(graph)),
    )


def interpret_intent(text: str, profile: GraphProfile) -> Intent:
    """Map natural-language preferences to visible numeric priorities.

    A language model may propose these weights upstream. Only the normalized
    numbers enter planning, so the chosen objective remains inspectable.
    """
    lowered = text.casefold()
    scores = {
        "connectivity": 1.5 if profile.components > 1 else 0.5,
        "cost": 0.35,
        "latency": 0.35,
        "robustness": 0.0,
    }
    lexicon = {
        "connectivity": ("connect", "restore", "reachable", "连通", "恢复", "可达"),
        "cost": ("cost", "cheap", "budget", "length", "成本", "造价", "预算"),
        "latency": ("latency", "delay", "shortest path", "fast", "时延", "延迟", "效率"),
        "robustness": ("robust", "redundant", "backup", "resilien", "鲁棒", "冗余", "备援"),
    }
    for objective, terms in lexicon.items():
        if any(term in lowered for term in terms):
            scores[objective] += 1.0
    return Intent(**scores).normalized()


def normalize_constraints(
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
    forbidden: Iterable[Edge] = (),
    must_connect: Iterable[Edge] = (),
    *,
    max_edge_length: float | None = None,
    shortcut_budget: int = 1,
    mode: ConstraintMode = "strict",
) -> Constraints:
    """Validate and canonicalize every externally supplied graph constraint."""
    if mode not in {"strict", "relaxed"}:
        raise ValueError("mode must be 'strict' or 'relaxed'")
    if max_edge_length is not None and max_edge_length <= 0:
        raise ValueError("max_edge_length must be positive")
    if shortcut_budget < 0:
        raise ValueError("shortcut_budget cannot be negative")

    missing_positions = set(graph.nodes) - set(positions)
    if missing_positions:
        raise ValueError(
            f"Missing positions for nodes: {sorted(missing_positions, key=repr)}"
        )
    for node in graph.nodes:
        x, y = positions[node]
        if not (isfinite(x) and isfinite(y)):
            raise ValueError(f"Non-finite position for node {node!r}")

    forbidden_set = frozenset(canonical_edge(*edge) for edge in forbidden)
    must_set = frozenset(canonical_edge(*edge) for edge in must_connect)
    endpoints = {node for edge in forbidden_set | must_set for node in edge}
    unknown = endpoints - set(graph.nodes)
    if unknown:
        raise ValueError(
            f"Constraint references unknown nodes: {sorted(unknown, key=repr)}"
        )
    conflict = forbidden_set & must_set
    if conflict and mode == "strict":
        raise InfeasibleRepair(
            f"Edges cannot be both required and forbidden: {sorted(conflict, key=repr)}"
        )

    return Constraints(
        forbidden=forbidden_set,
        must_connect=must_set,
        max_edge_length=max_edge_length,
        shortcut_budget=shortcut_budget,
        mode=mode,
    )


def _weighted_copy(
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
) -> nx.Graph:
    weighted = graph.copy()
    for u, v in weighted.edges:
        weighted[u][v].setdefault("weight", edge_length(u, v, positions))
    return weighted


def graph_metrics(graph: nx.Graph) -> dict[str, float]:
    """Compute selection metrics without pretending a damaged graph is connected."""
    if graph.number_of_nodes() == 0:
        return {
            "gcc_ratio": 0.0,
            "aspl": 0.0,
            "weighted_aspl": 0.0,
            "cycle_surplus": 0.0,
        }
    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    view = graph.subgraph(largest)
    if len(largest) <= 1:
        aspl = weighted_aspl = 0.0
    else:
        aspl = float(nx.average_shortest_path_length(view))
        weighted_aspl = float(nx.average_shortest_path_length(view, weight="weight"))
    forest_edges = graph.number_of_nodes() - len(components)
    cycle_surplus = max(0, graph.number_of_edges() - forest_edges) / max(
        graph.number_of_nodes(), 1
    )
    return {
        "gcc_ratio": len(largest) / graph.number_of_nodes(),
        "aspl": aspl,
        "weighted_aspl": weighted_aspl,
        "cycle_surplus": float(cycle_surplus),
    }


def _edge_key(
    u: Node,
    v: Node,
    positions: dict[Node, tuple[float, float]],
    constraints: Constraints,
) -> tuple[int, float] | None:
    edge = canonical_edge(u, v)
    length = edge_length(u, v, positions)
    if constraints.max_edge_length is not None and length > constraints.max_edge_length:
        return None
    violation = int(edge in constraints.forbidden)
    if violation and constraints.mode == "strict":
        return None
    return violation, length


def _add_required_edges(
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
    constraints: Constraints,
) -> tuple[list[RepairStep], list[str]]:
    steps: list[RepairStep] = []
    warnings: list[str] = []
    for edge in sorted(constraints.must_connect, key=repr):
        if graph.has_edge(*edge):
            continue
        length = edge_length(*edge, positions)
        if constraints.max_edge_length is not None and length > constraints.max_edge_length:
            raise InfeasibleRepair(f"Required edge {edge!r} exceeds max_edge_length")
        if edge in constraints.forbidden:
            warnings.append(f"relaxed conflict on required edge {edge!r}")
        graph.add_edge(*edge, weight=length)
        steps.append(RepairStep("add_edge", edge, length, "required connection"))
    return steps, warnings


def _best_component_edge(
    left: set[Node],
    right: set[Node],
    positions: dict[Node, tuple[float, float]],
    constraints: Constraints,
) -> tuple[int, float, Node, Node] | None:
    candidates = []
    for u in left:
        for v in right:
            key = _edge_key(u, v, positions, constraints)
            if key is not None:
                candidates.append((*key, u, v))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (item[0], item[1], repr(item[2]), repr(item[3])),
    )


def _shortest_bridge_plan(
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
    constraints: Constraints,
) -> tuple[nx.Graph, list[RepairStep], list[str]]:
    repaired = graph.copy()
    steps, warnings = _add_required_edges(repaired, positions, constraints)
    while repaired.number_of_nodes() and not nx.is_connected(repaired):
        components = [set(component) for component in nx.connected_components(repaired)]
        options = []
        for index, left in enumerate(components):
            for right in components[index + 1 :]:
                candidate = _best_component_edge(left, right, positions, constraints)
                if candidate is not None:
                    options.append(candidate)
        if not options:
            raise InfeasibleRepair("No feasible edge can connect the remaining components")
        violation, length, u, v = min(
            options,
            key=lambda item: (item[0], item[1], repr(item[2]), repr(item[3])),
        )
        edge = canonical_edge(u, v)
        repaired.add_edge(*edge, weight=length)
        reason = "shortest feasible bridge between components"
        if violation:
            reason += "; relaxed forbidden-edge violation"
            warnings.append(f"used forbidden edge {edge!r} to restore connectivity")
        steps.append(RepairStep("add_edge", edge, length, reason))
    return repaired, steps, warnings


def _component_tree_plan(
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
    constraints: Constraints,
) -> tuple[nx.Graph, list[RepairStep], list[str]]:
    repaired = graph.copy()
    steps, warnings = _add_required_edges(repaired, positions, constraints)
    components = [set(component) for component in nx.connected_components(repaired)]
    if len(components) <= 1:
        return repaired, steps, warnings

    component_graph = nx.Graph()
    component_graph.add_nodes_from(range(len(components)))
    diagonal = _bounding_diagonal(positions)
    violation_penalty = diagonal * (len(components) + 1)
    for i, left in enumerate(components):
        for j, right in enumerate(components[i + 1 :], start=i + 1):
            candidate = _best_component_edge(left, right, positions, constraints)
            if candidate is None:
                continue
            violation, length, u, v = candidate
            component_graph.add_edge(
                i,
                j,
                objective=length + violation * violation_penalty,
                physical_length=length,
                violation=violation,
                pair=canonical_edge(u, v),
            )
    if not nx.is_connected(component_graph):
        raise InfeasibleRepair("The feasible component graph is disconnected")

    tree = nx.minimum_spanning_tree(component_graph, weight="objective")
    for _, _, data in tree.edges(data=True):
        edge = data["pair"]
        repaired.add_edge(*edge, weight=data["physical_length"])
        reason = "minimum-cost edge in the component-level spanning tree"
        if data["violation"]:
            reason += "; relaxed forbidden-edge violation"
            warnings.append(f"used forbidden edge {edge!r} in component tree")
        steps.append(RepairStep("add_edge", edge, data["physical_length"], reason))
    return repaired, steps, warnings


def _best_latency_shortcut(
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
    constraints: Constraints,
) -> tuple[float, float, Node, Node] | None:
    best = None
    for u, v in nx.non_edges(graph):
        key = _edge_key(u, v, positions, constraints)
        if key is None or key[0]:
            continue
        _, direct = key
        current = float(nx.shortest_path_length(graph, u, v, weight="weight"))
        gain = current - direct
        if gain <= 0:
            continue
        candidate = (gain, direct, u, v)
        if best is None or (
            gain,
            -direct,
            repr(u),
            repr(v),
        ) > (best[0], -best[1], repr(best[2]), repr(best[3])):
            best = candidate
    return best


def _add_latency_shortcuts(
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
    constraints: Constraints,
    steps: list[RepairStep],
) -> None:
    for _ in range(constraints.shortcut_budget):
        candidate = _best_latency_shortcut(graph, positions, constraints)
        if candidate is None:
            break
        gain, length, u, v = candidate
        edge = canonical_edge(u, v)
        graph.add_edge(*edge, weight=length)
        steps.append(
            RepairStep(
                "add_edge",
                edge,
                length,
                f"latency shortcut with weighted-path gain {gain:.3f}",
            )
        )


def _run_strategy(
    name: str,
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
    constraints: Constraints,
) -> tuple[nx.Graph, list[RepairStep], list[str]]:
    if name == "shortest_bridge":
        return _shortest_bridge_plan(graph, positions, constraints)
    if name == "component_tree":
        return _component_tree_plan(graph, positions, constraints)
    if name == "latency_repair":
        repaired, steps, warnings = _shortest_bridge_plan(
            graph, positions, constraints
        )
        _add_latency_shortcuts(repaired, positions, constraints, steps)
        return repaired, steps, warnings
    if name == "balanced_tree_latency":
        repaired, steps, warnings = _component_tree_plan(
            graph, positions, constraints
        )
        _add_latency_shortcuts(repaired, positions, constraints, steps)
        return repaired, steps, warnings
    raise ValueError(f"Unknown strategy: {name}")


def _strategy_rationale(name: str, intent: Intent, profile: GraphProfile) -> str:
    descriptions = {
        "shortest_bridge": "Connect the current components greedily with short feasible edges.",
        "component_tree": "Build a minimum-cost spanning structure over the component graph.",
        "latency_repair": "Restore connectivity, then spend a bounded budget on path shortcuts.",
        "balanced_tree_latency": "Compose a component tree with bounded latency shortcuts.",
    }
    context = (
        f"The damaged graph has {profile.components} components; normalized priorities are "
        f"connectivity={intent.connectivity:.2f}, cost={intent.cost:.2f}, "
        f"latency={intent.latency:.2f}, robustness={intent.robustness:.2f}."
    )
    return f"{descriptions[name]} {context}"


def _bounding_diagonal(positions: dict[Node, tuple[float, float]]) -> float:
    if not positions:
        return 1.0
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    return max(hypot(max(xs) - min(xs), max(ys) - min(ys)), 1e-12)


def _utility(
    before: dict[str, float],
    after: dict[str, float],
    added_length: float,
    positions: dict[Node, tuple[float, float]],
    intent: Intent,
) -> float:
    connectivity_gain = after["gcc_ratio"] - before["gcc_ratio"]
    latency_gain = before["weighted_aspl"] - after["weighted_aspl"]
    latency_term = latency_gain / max(before["weighted_aspl"], 1.0)
    robustness_gain = after["cycle_surplus"] - before["cycle_surplus"]
    cost_term = added_length / _bounding_diagonal(positions)
    return (
        4.0 * intent.connectivity * connectivity_gain
        + intent.latency * latency_term
        + intent.robustness * robustness_gain
        - intent.cost * cost_term
    )


def _validate_result(
    graph: nx.Graph,
    plan: list[RepairStep],
    constraints: Constraints,
) -> dict[str, object]:
    added = {step.edge for step in plan}
    forbidden_violations = sorted(added & constraints.forbidden, key=repr)
    missing_required = sorted(
        (edge for edge in constraints.must_connect if not graph.has_edge(*edge)),
        key=repr,
    )
    return {
        "connected": graph.number_of_nodes() == 0 or nx.is_connected(graph),
        "forbidden_violations": [list(edge) for edge in forbidden_violations],
        "must_connect_satisfied": not missing_required,
        "missing_required": [list(edge) for edge in missing_required],
    }


def restore(
    graph: nx.Graph,
    positions: dict[Node, tuple[float, float]],
    intent_text: str,
    *,
    forbidden: Iterable[Edge] = (),
    must_connect: Iterable[Edge] = (),
    max_edge_length: float | None = None,
    shortcut_budget: int = 1,
    mode: ConstraintMode = "strict",
) -> RepairResult:
    """Construct, select, execute, and verify a repair on a graph copy.

    ``balanced_tree_latency`` is a transparent compositional extension of the
    fixed strategy library. It demonstrates how a new strategy can be proposed
    as a declarative combination of audited graph operators rather than as
    arbitrary model-generated Python.
    """
    profile = analyze_graph(graph)
    constraints = normalize_constraints(
        graph,
        positions,
        forbidden,
        must_connect,
        max_edge_length=max_edge_length,
        shortcut_budget=shortcut_budget,
        mode=mode,
    )
    weighted_graph = _weighted_copy(graph, positions)
    intent = interpret_intent(intent_text, profile)
    before = graph_metrics(weighted_graph)

    strategies = ["shortest_bridge", "component_tree", "latency_repair"]
    if intent.cost >= 0.15 and intent.latency >= 0.15:
        strategies.append("balanced_tree_latency")

    candidates: list[CandidateEvaluation] = []
    outcomes: dict[str, tuple[nx.Graph, list[RepairStep], list[str]]] = {}
    for strategy in strategies:
        rationale = _strategy_rationale(strategy, intent, profile)
        try:
            repaired, plan, warnings = _run_strategy(
                strategy, weighted_graph, positions, constraints
            )
            after = graph_metrics(repaired)
            added_length = sum(step.weight for step in plan)
            utility = _utility(before, after, added_length, positions, intent)
            outcomes[strategy] = (repaired, plan, warnings)
            candidates.append(
                CandidateEvaluation(
                    strategy=strategy,
                    feasible=True,
                    rationale=rationale,
                    utility=utility,
                    added_edges=len(plan),
                    added_length=added_length,
                    weighted_path_length=after["weighted_aspl"],
                    warnings=warnings.copy(),
                )
            )
        except InfeasibleRepair as error:
            candidates.append(
                CandidateEvaluation(
                    strategy=strategy,
                    feasible=False,
                    rationale=rationale,
                    warnings=[str(error)],
                )
            )

    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        raise InfeasibleRepair("No candidate can satisfy the declared constraints")
    selected = max(
        feasible,
        key=lambda candidate: (
            candidate.utility,
            -(candidate.added_edges or 0),
            candidate.strategy,
        ),
    )
    repaired, plan, _ = outcomes[selected.strategy]
    validation = _validate_result(repaired, plan, constraints)
    if mode == "strict" and (
        not validation["connected"]
        or validation["forbidden_violations"]
        or not validation["must_connect_satisfied"]
    ):
        raise AssertionError("Post-repair graph validation failed")

    return RepairResult(
        graph_profile=profile,
        intent=intent,
        constraints=constraints,
        selected_strategy=selected.strategy,
        selected_rationale=selected.rationale,
        candidates=candidates,
        plan=plan,
        validation=validation,
        metrics_before=before,
        metrics_after=graph_metrics(repaired),
        graph=repaired,
    )

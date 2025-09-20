#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Element = Tuple[int, int, int]
NodeMap = Dict[int, Tuple[float, float]]
Edge = Tuple[int, int]


def read_fvcom_grd(path: str | Path) -> Tuple[int, int, List[Element], NodeMap]:
    """Parse an FVCOM grid (.dat) file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Grid file not found: {file_path}")

    elements: List[Element] = []
    nodes: NodeMap = {}

    with file_path.open("r", encoding="utf-8") as fh:
        header_nodes = fh.readline()
        header_cells = fh.readline()
        if not header_nodes or not header_cells:
            raise ValueError("Grid file is missing header lines")

        try:
            nnode = int(header_nodes.split("=")[-1].strip())
            nelem = int(header_cells.split("=")[-1].strip())
        except ValueError as exc:
            raise ValueError("Unable to parse node or cell counts from header") from exc

        # Read element connectivity
        while len(elements) < nelem:
            line = fh.readline()
            if not line:
                raise ValueError("Unexpected end of file while reading elements")
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid element line: '{line.strip()}'")
            try:
                n1, n2, n3 = map(int, parts[1:4])
            except ValueError as exc:
                raise ValueError(f"Invalid element indices in line: '{line.strip()}'") from exc
            elements.append((n1, n2, n3))

        # Read node coordinates
        while len(nodes) < nnode:
            line = fh.readline()
            if not line:
                raise ValueError("Unexpected end of file while reading nodes")
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid node line: '{line.strip()}'")
            try:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
            except ValueError as exc:
                raise ValueError(f"Invalid node entry in line: '{line.strip()}'") from exc
            nodes[node_id] = (x, y)

    if len(elements) != nelem or len(nodes) != nnode:
        raise ValueError(
            f"Parsed counts do not match header values (nodes: {len(nodes)} vs {nnode}, "
            f"elements: {len(elements)} vs {nelem})"
        )

    return nnode, nelem, elements, nodes


def find_boundary_edges(elements: Sequence[Element]) -> Set[Edge]:
    """Return the set of boundary edges (appear exactly once)."""
    edge_counts: Dict[Edge, int] = defaultdict(int)

    for n1, n2, n3 in elements:
        triangle_edges = ((n1, n2), (n2, n3), (n3, n1))
        for u, v in triangle_edges:
            key = (u, v) if u < v else (v, u)
            edge_counts[key] += 1

    return {edge for edge, count in edge_counts.items() if count == 1}


def boundary_polylines(boundary_edges: Iterable[Edge]) -> List[Tuple[List[int], bool]]:
    """Order boundary edges into polylines and flag closed loops."""
    edges = {tuple(sorted(edge)) for edge in boundary_edges}
    if not edges:
        return []

    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    visited_edges: Set[Edge] = set()
    polylines: List[Tuple[List[int], bool]] = []

    def trace(start: int) -> Tuple[List[int], bool]:
        path = [start]
        current = start
        while True:
            next_node = None
            for nb in sorted(adjacency[current]):
                edge = (current, nb) if current < nb else (nb, current)
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                next_node = nb
                break
            if next_node is None:
                break
            path.append(next_node)
            current = next_node
            if current == start:
                break
        closed = len(path) > 1 and path[0] == path[-1]
        if closed:
            path = path[:-1]
        return path, closed

    sorted_nodes = sorted(adjacency)

    # First extract open polylines (degree-1 endpoints)
    for node in sorted_nodes:
        if len(adjacency[node]) != 1:
            continue
        path, closed = trace(node)
        if len(path) > 1:
            polylines.append((path, closed))

    # Then handle remaining closed loops
    for node in sorted_nodes:
        for nb in sorted(adjacency[node]):
            edge = (node, nb) if node < nb else (nb, node)
            if edge in visited_edges:
                continue
            path, closed = trace(node)
            if len(path) > 1:
                polylines.append((path, closed))

    return polylines


def write_boundary_nodes(nodes: Sequence[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for node_id in nodes:
            fh.write(f"{node_id}\n")


def write_polylines(polylines: Sequence[Tuple[List[int], bool]], output_path: Path) -> None:
    payload = [
        {"nodes": polyline, "closed": closed}
        for polyline, closed in polylines
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract FVCOM boundary nodes and polylines from a grid file without "
            "using auxiliary metadata."
        )
    )
    parser.add_argument("grid_file", help="Path to the FVCOM grid (.dat) file")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where boundary_nodes.txt and boundary_polylines.json will be written",
    )
    args = parser.parse_args()

    grid_path = Path(args.grid_file)
    output_dir = Path(args.output_dir)

    nnode, nelem, elements, _nodes = read_fvcom_grd(grid_path)
    boundary_edges = find_boundary_edges(elements)
    boundary_nodes = sorted({node for edge in boundary_edges for node in edge})
    polylines = boundary_polylines(boundary_edges)

    write_boundary_nodes(boundary_nodes, output_dir / "boundary_nodes.txt")
    write_polylines(polylines, output_dir / "boundary_polylines.json")

    print(
        f"Parsed {nnode} nodes and {nelem} elements. Found {len(boundary_nodes)} boundary nodes "
        f"and {len(polylines)} boundary polylines."
    )


if __name__ == "__main__":
    main()

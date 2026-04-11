"""
Correlative sparsity pattern + chordal completion + maximal-clique enumeration.

We use the *csp graph* on disk indices 0..n-1, with an edge (i,j) whenever
the variable groups of disk i and disk j appear together in at least one
constraint polynomial. Since non-overlap constraints couple disk i and disk j,
the csp graph *is* the contact graph (disk-wall edges couple a disk with
only its own variables).

Then we compute a chordal completion via min-degree ordering and enumerate
maximal cliques. This is the correlative sparsity clique decomposition used
by Waki-Kim-Kojima-Muramatsu and TSSOS.
"""
import json
from pathlib import Path
import networkx as nx

HERE = Path(__file__).parent
IN = HERE / "contacts.json"
OUT = HERE / "cliques.json"


def chordal_completion_min_degree(G: nx.Graph):
    """Classical min-degree chordal completion. Returns (chordal_graph, fill_edges)."""
    H = G.copy()
    fill = []
    remaining = set(H.nodes())
    # Work on a mutable copy for elimination
    W = H.copy()
    while remaining:
        # pick node of minimum degree in W
        v = min(remaining, key=lambda u: W.degree(u))
        neigh = [w for w in W.neighbors(v) if w in remaining]
        # connect all neighbors in W and in H
        for i in range(len(neigh)):
            for j in range(i + 1, len(neigh)):
                a, b = neigh[i], neigh[j]
                if not W.has_edge(a, b):
                    W.add_edge(a, b)
                    H.add_edge(a, b)
                    fill.append((a, b))
        remaining.discard(v)
    return H, fill


def main():
    data = json.loads(IN.read_text())
    n = data["n"]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in data["disk_disk_edges"]:
        G.add_edge(i, j)

    print(f"Contact graph: |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
    print(f"Is chordal? {nx.is_chordal(G)}")

    H, fill = chordal_completion_min_degree(G)
    print(f"\nChordal completion: added {len(fill)} fill edges")
    print(f"Completed: |V|={H.number_of_nodes()}  |E|={H.number_of_edges()}")
    print(f"Is chordal? {nx.is_chordal(H)}")

    # Enumerate maximal cliques of the chordal graph
    cliques = list(nx.find_cliques(H))
    cliques.sort(key=len, reverse=True)
    sizes = [len(c) for c in cliques]
    print(f"\nMaximal cliques: {len(cliques)}")
    print(f"  max size : {max(sizes)}")
    print(f"  mean size: {sum(sizes)/len(sizes):.2f}")
    print(f"  size histogram: { {s: sizes.count(s) for s in sorted(set(sizes))} }")

    print("\nTop 5 cliques:")
    for c in cliques[:5]:
        print(f"  {sorted(c)}  (size {len(c)})")

    out = {
        "n": n,
        "num_contact_edges": G.number_of_edges(),
        "num_fill_edges": len(fill),
        "num_chordal_edges": H.number_of_edges(),
        "fill_edges": [list(e) for e in fill],
        "num_cliques": len(cliques),
        "max_clique_size": max(sizes),
        "mean_clique_size": sum(sizes) / len(sizes),
        "clique_size_histogram": {str(s): sizes.count(s) for s in sorted(set(sizes))},
        "cliques": [sorted(c) for c in cliques],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

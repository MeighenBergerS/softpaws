"""Extract the flavour-triangle curves of arXiv:2510.24957 Fig. 1.

Reads the vector paths of ``Measurement_11yrs_flav_theory_and_previous.pdf``
from the paper's arXiv source, maps them onto flavour fractions through the
triangle's own vertices, and writes ``flavor2510_fig1.csv``.

Two contours carry gaps that are typography, not data: matplotlib's inline
labels ("68% CL", "95% CL") cut the stroke where the text sits. Those gaps
are bridged with a cubic Hermite arc matched to the contour's tangents on
either side. The 95% contour also leaves the triangle through the
``f_tau = 0`` edge; that gap is physical and stays open.

Usage
-----
    python extract_flavor2510_fig1.py <path to the figure PDF>
"""

import pathlib
import sys

import fitz
import numpy as np

#: Stroke signatures of the four curves: (name, colour, width, dashed).
CURVES = (
    ("contour68", (0.0, 0.0, 0.0), 5.5, False),
    ("contour95", (0.0, 0.0, 0.0), 5.5, True),
    ("icecube2022", (0.541, 0.329, 0.290), 3.0, True),
    ("std_osc", (0.408, 0.408, 0.408), 3.1, True),
)

#: Largest jump [triangle units] still treated as a label gap to bridge; the
#: physical clip of the 95% contour is longer and stays open.
BRIDGE_MAX = 0.2


def polyline(drawing):
    """Vertices of a stroked path, with the indices of subpath breaks."""
    points, breaks = [], []
    for item in drawing["items"]:
        if item[0] != "l":
            continue
        start, end = item[1], item[2]
        if not points:
            points.append((start.x, start.y))
        elif abs(points[-1][0] - start.x) > 1e-6 or abs(points[-1][1] - start.y) > 1e-6:
            breaks.append(len(points))
            points.append((start.x, start.y))
        points.append((end.x, end.y))
    return np.array(points), breaks


def triangle_vertices(drawings):
    """Bottom-left, bottom-right and top vertex of the black triangle frame."""
    edges = [d for d in drawings
             if d.get("color") == (0.0, 0.0, 0.0) and abs((d.get("width") or 0) - 1.5) < 0.05
             and len(d["items"]) == 1 and d["items"][0][0] == "l"]
    ends = np.array([[p.x, p.y] for d in edges for p in d["items"][0][1:]])
    top = ends[np.argmin(ends[:, 1])]
    bottom = ends[np.isclose(ends[:, 1], ends[:, 1].max(), atol=0.5)]
    return bottom[np.argmin(bottom[:, 0])], bottom[np.argmax(bottom[:, 0])], top


def to_fractions(points, bl, br, top):
    """Barycentric map: bottom-left ``nu_tau``, bottom-right ``nu_e``, top ``nu_mu``."""
    matrix = np.array([[bl[0] - top[0], br[0] - top[0]], [bl[1] - top[1], br[1] - top[1]]])
    ab = np.linalg.solve(matrix, (points - top).T).T
    f_tau, f_e = ab[:, 0], ab[:, 1]
    return np.column_stack([f_e, 1.0 - f_tau - f_e, f_tau])


def hermite_bridge(p_before, p0, p1, p_after, step):
    """Points strictly between ``p0`` and ``p1`` on a tangent-matched cubic."""
    gap = np.linalg.norm(p1 - p0)
    t0 = (p0 - p_before) / np.linalg.norm(p0 - p_before) * gap
    t1 = (p_after - p1) / np.linalg.norm(p_after - p1) * gap
    n = max(int(round(gap / step)), 2)
    t = np.linspace(0.0, 1.0, n + 1)[1:-1][:, None]
    return ((2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * t0
            + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * t1)


def bridge_gaps(points, breaks, side, on_edge):
    """Fill label gaps: mid-path breaks, and a short first-last gap unless
    both ends sit on a triangle edge (a clipped contour, left open)."""
    step = np.median(np.linalg.norm(np.diff(points, axis=0), axis=1))
    pieces, last = [], 0
    for b in breaks:
        pieces.append(points[last:b])
        gap = np.linalg.norm(points[b] - points[b - 1]) / side
        if gap <= BRIDGE_MAX:
            pieces.append(hermite_bridge(points[b - 2], points[b - 1],
                                         points[b], points[b + 1], step))
        last = b
    pieces.append(points[last:])
    out = np.vstack(pieces)
    closing = np.linalg.norm(out[0] - out[-1]) / side
    if 0.02 < closing <= BRIDGE_MAX and not (on_edge(out[0]) and on_edge(out[-1])):
        out = np.vstack([out, hermite_bridge(out[-2], out[-1], out[0], out[1], step), out[:1]])
    return out


def main(pdf_path: pathlib.Path) -> None:
    page = fitz.open(pdf_path)[0]
    drawings = page.get_drawings()
    bl, br, top = triangle_vertices(drawings)
    side = np.linalg.norm(br - bl)
    rows = []
    for name, colour, width, dashed in CURVES:
        match = [d for d in drawings
                 if d.get("color") is not None
                 and np.allclose(d["color"], colour, atol=0.01)
                 and abs((d.get("width") or 0) - width) < 0.1
                 and bool(d.get("dashes") not in (None, "[] 0")) == dashed
                 and len(d["items"]) > 20]
        assert len(match) == 1, (name, len(match))
        points, breaks = polyline(match[0])
        n_raw = len(points)

        def on_edge(point):
            return to_fractions(point[None, :], bl, br, top).min() < 0.005

        points = bridge_gaps(points, breaks, side, on_edge)
        fractions = to_fractions(points, bl, br, top)
        print(f"  {name}: {n_raw} vertices, breaks {breaks}, "
              f"{len(points) - n_raw} bridging points added")
        rows += [f"{name},{f[0]:.4f},{f[1]:.4f},{f[2]:.4f}" for f in fractions]
    out = pathlib.Path(__file__).with_name("flavor2510_fig1.csv")
    out.write_text(
        "# Flavour-composition curves of IceCube's three-flavour characterization,\n"
        "# extracted from the vector paths of arXiv:2510.24957 Fig. 1 (MESE fit) by\n"
        "# extract_flavor2510_fig1.py: barycentric map through the triangle's own\n"
        "# vertices (bottom-left nu_tau, bottom-right nu_e, top nu_mu). The inline\n"
        "# '68% CL' / '95% CL' labels cut the strokes in the source; those gaps are\n"
        "# bridged with tangent-matched cubic arcs. The 95% contour leaves the\n"
        "# triangle through f_tau = 0 and stays open there.\n"
        "# name,f_e,f_mu,f_tau\n" + "\n".join(rows) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))

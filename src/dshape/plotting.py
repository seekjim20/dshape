import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from typing import Sequence, Union, Optional
from .geometry import Polygon


def plot_polygons(
    polygons: Union[Polygon, Sequence[Polygon]],
    names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    title: str = "Polygon Plot",
    opacity: float = 0.5,
) -> go.Figure:
    """Plots one or more Polygons using Plotly.

    Args:
        polygons: A single Polygon or a sequence of Polygons.
        names: Optional sequence of names for the legend.
        colors: Optional sequence of colors for the polygons.
        title: Title of the plot.
        opacity: Opacity fill for polygons.

    Returns:
        A plotly.graph_objects.Figure object.
    """
    if isinstance(polygons, Polygon):
        polygons = [polygons]

    if names is None:
        names = [f"Polygon {i+1}" for i in range(len(polygons))]

    if len(names) != len(polygons):
        # Fallback if length mismatch
        names = [f"Polygon {i+1}" for i in range(len(polygons))]

    fig = go.Figure()

    for i, poly in enumerate(polygons):
        # We need to extract rings and separate them with None

        # ring_counts might not be available on old objects if cached?
        # But we updated Polygon class.

        # Helper to get numpy array
        def to_np(arr):
            if hasattr(arr, "device_buffer"):
                return np.array(arr)
            return np.asarray(arr)

        vertices = to_np(poly.vertices)

        if hasattr(poly, "ring_counts") and poly.ring_counts is not None:
            r_counts = to_np(poly.ring_counts)
        else:
            r_counts = np.array([poly.count])

        # Filter valid rings
        r_counts = r_counts[r_counts > 0]

        if len(r_counts) == 0:
            continue

        starts = np.cumsum(np.concatenate(([0], r_counts[:-1]))).astype(int)

        xs = []
        ys = []

        for start, count in zip(starts, r_counts):
            if count < 3:
                continue

            end = start + count
            ring = vertices[start:end]

            # Close loop
            if np.linalg.norm(ring[-1] - ring[0]) > 1e-6:
                ring = np.concatenate([ring, ring[0:1]], axis=0)

            xs.extend(ring[:, 0].tolist())
            ys.extend(ring[:, 1].tolist())

            # Separator
            xs.append(None)
            ys.append(None)

        if not xs:
            continue

        color = colors[i] if colors and i < len(colors) else None

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                name=names[i],
                line=dict(color=color),
                fillcolor=color,
                opacity=opacity,
            )
        )

    fig.update_layout(
        title=title, yaxis=dict(scaleanchor="x", scaleratio=1), showlegend=True
    )

    return fig


import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from typing import Sequence, Union, Optional
from geometry import Polygon

def plot_polygons(
    polygons: Union[Polygon, Sequence[Polygon]],
    names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    title: str = "Polygon Plot",
    opacity: float = 0.5
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
        count = poly.count
        if hasattr(count, 'item'):
             count = int(count.item())
        else:
             count = int(count)

        max_v = poly.vertices.shape[0]
        count = min(count, max_v)
        
        verts = poly.vertices[:count]
        if hasattr(verts, 'device_buffer'):
             verts = np.array(verts)
        
        if count == 0:
            continue

        if np.linalg.norm(verts[-1] - verts[0]) > 1e-6:
             verts = np.concatenate([verts, verts[0:1]], axis=0)

        x = verts[:, 0]
        y = verts[:, 1]
        
        color = colors[i] if colors and i < len(colors) else None
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            fill="toself",
            name=names[i],
            line=dict(color=color),
            fillcolor=color,
            opacity=opacity
        ))

    fig.update_layout(
        title=title,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True
    )
    
    return fig

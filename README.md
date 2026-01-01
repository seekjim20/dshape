# dshape: Differentiable Shape Operations with JAX

`dshape` is a Python library for performing differentiable geometric operations on polygons, built on top of [JAX](https://github.com/google/jax). It enables gradient-based optimization of geometric shapes by providing differentiable implementations of common operations like intersection, union, difference, and buffering.

## Features

- **Differentiable Polygon Operations**:
    - `intersection(p1, p2)`: Sutherland-Hodgman clipping for convex/concave polygons.
    - `union(p1, p2)`: Construct unified geometry.
    - `difference(p1, p2)`: Subtract one polygon from another.
    - `buffer(p, distance)`: Dilate or erode polygons with miter or arc joins.
    - `offset(p, dx, dy)`: Translate polygons.
- **JAX Integration**: Fully compatible with `jax.jit`, `jax.grad`, and `jax.vmap`.
- **Robustness**: Handles degenerate cases, self-intersections (pruning), and floating-point edge cases.

## Usage

```python
import jax.numpy as jnp
from geometry import Polygon
from ops import intersection, buffer

# Define a polygon
verts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
poly = Polygon(vertices=verts, count=3)

# Buffer it (Dilation)
buffered_poly = buffer(poly, distance=0.1)

# Intersect with another polygon
clip_poly = Polygon(vertices=jnp.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]), count=4)
result = intersection(buffered_poly, clip_poly)
```

## Structure

- `src/`: Source code (`geometry.py`, `ops.py`).
- `test/`: Unit tests (using `pytest`).

## Testing

Run the test suite with:

```bash
pytest test/
```

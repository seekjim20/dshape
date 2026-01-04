import jax
import jax.numpy as jnp
import pytest
import dshape.geometry
import dshape.constructive


def test_convex_hull_grad():
    def calculate_area(radius):
        radius = jnp.array(radius)  # Ensure array
        # Create a circle centered at 0,0
        c = dshape.geometry.Circle((0.0, 0.0), radius, num_edges=50)
        # Compute hull
        hull = c.convex_hull
        # For a circle, hull area should be approx equal to circle area
        return hull.area

    # Forward pass works?
    res = calculate_area(1.0)
    print(f"Forward Result: {res}")

    # Backward pass
    grad_fn = jax.grad(calculate_area)

    try:
        g = grad_fn(1.0)
        print(f"Gradient: {g}")
    except Exception as e:
        print(f"Caught Expected Error: {e}")
        raise e


if __name__ == "__main__":
    test_convex_hull_grad()

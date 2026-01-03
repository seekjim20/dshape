import os
import geometry
import plotting
import set_ops


def main():
    print("Running Plotly Demo...")

    # 1. Basic Rects
    p1 = geometry.Rectangle(0, 0, 4, 4)
    p2 = geometry.Rectangle(1, 1, 2, 2)

    # 2. Difference (Bite)
    diff_poly = set_ops.difference(p1, p2)

    # 3. Intersection
    p3 = geometry.Rectangle(2, 2, 3, 3)  # Intersects p1
    int_poly = set_ops.intersection(p1, p3)

    # 4. Circle
    c1 = geometry.Circle(4, 4, 1.5, num_edges=64)

    # Plot Multiple
    polys = [p1, p2, diff_poly, int_poly, c1]
    names = [
        "Outer Rect",
        "Inner Rect",
        "Difference (Hole/Bite)",
        "Intersection",
        "Circle",
    ]

    # Create Figure
    fig = plotting.plot_polygons(
        polys, names=names, title="DShape Polygon Demo", opacity=0.6
    )

    print("Figure created.")

    # Save to HTML
    output_file = "dshape_demo.html"
    fig.write_html(output_file)
    print(f"Saved figure to {output_file}")

    # Verify file exists
    if os.path.exists(output_file):
        print("SUCCESS: HTML file generated.")
    else:
        print("FAIL: HTML file not found.")


if __name__ == "__main__":
    main()

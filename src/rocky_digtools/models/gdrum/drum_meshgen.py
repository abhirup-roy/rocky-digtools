"""Axially periodic drum shell and downward-facing inlet meshes (metres)."""

import math
from pathlib import Path

import gmsh


def drum_dimensions(box_len: float, aspect_ratio: float) -> tuple[float, float]:
    """Return radius and length with L / (2R) = aspect_ratio and πR²L = box_len³."""
    for name, value in (("box_len", box_len), ("aspect_ratio", aspect_ratio)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    radius = box_len / (2 * math.pi * aspect_ratio) ** (1 / 3)
    return radius, 2 * radius * aspect_ratio


def draw_cylinder(radius, length, center):
    """Add a cylindrical shell aligned with Z, without end caps; return its surface tag."""
    x, y, z = center
    circle = gmsh.model.occ.addCircle(x, y, z - length / 2, radius)
    surfaces = gmsh.model.occ.extrude([(1, circle)], 0, 0, length)
    return next(tag for dimension, tag in surfaces if dimension == 2)


def draw_insert(length, width, height):
    """Add an XZ rectangle at Y=height with its normal pointing down."""
    surface = gmsh.model.occ.addRectangle(-width / 2, -length / 2, 0, width, length)
    gmsh.model.occ.rotate([(2, surface)], 0, 0, 0, 1, 0, 0, math.pi / 2)
    gmsh.model.occ.translate([(2, surface)], 0, height, 0)
    return surface


def create_meshes(
    size, meshsize=0.001, out_dir="meshes", aspect_ratio=1.0, *, aspect_ratios=None
):
    """Write drum_shell.stl and insert.stl under aspect_<ratio>/ for each geometry.

    ``size`` is the equivalent cube side length, not the drum diameter.
    Separate ratio directories let sweep/OFAT cases share each geometry safely.
    """
    ratios = sorted(set(aspect_ratios if aspect_ratios is not None else [aspect_ratio]))
    dimensions = [(ratio, *drum_dimensions(size, ratio)) for ratio in ratios]
    if not math.isfinite(meshsize) or meshsize <= 0:
        raise ValueError("meshsize must be finite and > 0")
    owned_session = not gmsh.isInitialized()
    if owned_session:
        gmsh.initialize()
    previous_model = gmsh.model.getCurrent()
    try:
        for ratio, radius, length in dimensions:
            destination = Path(out_dir) / f"aspect_{float(ratio)}"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("drum_shell", "insert"):
                gmsh.model.add(f"gdrum_{name}")
                try:
                    if name == "drum_shell":
                        draw_cylinder(radius, length, (0, 0, 0))
                    else:
                        draw_insert(0.9 * length, radius, radius / 2)
                    gmsh.model.occ.synchronize()
                    # Resolve curvature even when the DOE engine requests a coarse mesh.
                    resolution = min(meshsize, radius / 12, length / 12)
                    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), resolution)
                    gmsh.model.mesh.generate(2)
                    gmsh.write(str(destination / f"{name}.stl"))
                finally:
                    gmsh.model.remove()
    finally:
        if owned_session:
            gmsh.finalize()
        elif previous_model:
            gmsh.model.setCurrent(previous_model)

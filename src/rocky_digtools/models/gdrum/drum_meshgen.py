"""Axially periodic drum shell and downward-facing inlet meshes (metres)."""

import math
from pathlib import Path

import gmsh


def drum_dimensions(
    box_len: float | None = None,
    aspect_ratio: float | None = None,
    *,
    length: float | None = None,
    radius: float | None = None,
) -> tuple[float, float]:
    """Resolve geometry from any two of radius, length and L/D, or box_len plus one."""
    supplied = {
        name: value
        for name, value in (
            ("radius", radius),
            ("length", length),
            ("aspect_ratio", aspect_ratio),
        )
        if value is not None
    }
    required = 2 if box_len is None else 1
    if len(supplied) != required:
        raise ValueError(
            "Provide exactly two of drum_radius, length and aspect_ratio without box_len; "
            "or box_len and exactly one geometry value"
        )
    if box_len is not None:
        supplied["box_len"] = box_len
    for name, value in supplied.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if radius is None:
        if length is not None and aspect_ratio is not None:
            radius = length / (2 * aspect_ratio)
        elif aspect_ratio is not None:
            radius = box_len / (2 * math.pi * aspect_ratio) ** (1 / 3)
        else:
            radius = math.sqrt(box_len**3 / (math.pi * length))
    if length is None:
        length = (
            2 * radius * aspect_ratio
            if aspect_ratio is not None
            else box_len**3 / (math.pi * radius**2)
        )
    return radius, length


def geometry_name(box_len=None, aspect_ratio=None, *, length=None, radius=None):
    """Stable mesh subdirectory for the specified geometry."""
    resolved_radius, resolved_length = drum_dimensions(
        box_len, aspect_ratio, length=length, radius=radius
    )
    if box_len is None:
        return f"radius_{float(resolved_radius)}_length_{float(resolved_length)}"
    if aspect_ratio is not None:
        return f"aspect_{float(aspect_ratio)}"
    if length is not None:
        return f"length_{float(length)}"
    return f"radius_{float(radius)}"


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
    size=None,
    meshsize=0.001,
    out_dir="meshes",
    aspect_ratio=None,
    *,
    length=None,
    radius=None,
    aspect_ratios=None,
    lengths=None,
    geometries=None,
):
    """Write drum shell and inlet meshes; accept a single geometry or a DOE batch.

    ``size`` is an optional equivalent cube side length. Without it, provide
    any two of ``radius``, ``length`` and ``aspect_ratio``.
    """
    if sum(value is not None for value in (aspect_ratios, lengths, geometries)) > 1:
        raise ValueError("Provide only one batch of geometries")
    if any(value is not None for value in (aspect_ratios, lengths, geometries)):
        if any(value is not None for value in (aspect_ratio, length, radius)):
            raise ValueError(
                "Do not combine a geometry batch with individual geometry values"
            )
    if geometries is not None:
        specifications = geometries
    elif aspect_ratios is not None:
        specifications = [
            {"box_len": size, "aspect_ratio": value}
            for value in sorted(set(aspect_ratios))
        ]
    elif lengths is not None:
        specifications = [
            {"box_len": size, "length": value} for value in sorted(set(lengths))
        ]
    else:
        specifications = [
            {
                "box_len": size,
                "aspect_ratio": aspect_ratio,
                "length": length,
                "radius": radius,
            }
        ]
    dimensions = [
        (geometry_name(**spec), *drum_dimensions(**spec)) for spec in specifications
    ]
    if not math.isfinite(meshsize) or meshsize <= 0:
        raise ValueError("meshsize must be finite and > 0")
    owned_session = not gmsh.isInitialized()
    if owned_session:
        gmsh.initialize()
    previous_model = gmsh.model.getCurrent()
    try:
        for directory, radius, length in dimensions:
            destination = Path(out_dir) / directory
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

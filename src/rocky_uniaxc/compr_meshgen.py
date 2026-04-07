#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Abhirup Roy"
__email__ = "axr154@bham.ac.uk"
__status__ = "Development"

"""
This script generates meshes for Uniaxial Copmressions.
"""


import os
import pathlib
import sys
import gmsh


def create_topwall(size, meshsize, out_dir):
    gmsh.model.add("wall1")
    p1 = gmsh.model.geo.addPoint(-size / 2, -size, -size / 2, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(size / 2, -size, -size / 2, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(-size / 2, -size, size / 2, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(size / 2, -size, size / 2, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p4)
    l3 = gmsh.model.geo.addLine(p4, p3)
    l4 = gmsh.model.geo.addLine(p3, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "compressive_wall1.stl"))
    gmsh.model.remove()  # Clear current model


def create_bottomwall(size, meshsize, out_dir):
    gmsh.model.add("wall2")
    p1 = gmsh.model.geo.addPoint(size / 2, size, -size / 2, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(-size / 2, size, -size / 2, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(size / 2, size, size / 2, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-size / 2, size, size / 2, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p4)
    l3 = gmsh.model.geo.addLine(p4, p3)
    l4 = gmsh.model.geo.addLine(p3, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "compressive_wall2.stl"))
    gmsh.model.remove()


def create_insert(size, meshsize, out_dir):
    gmsh.model.add("insert")
    p1 = gmsh.model.geo.addPoint(-size / 2, -size / 2, size / 2, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(size / 2, -size / 2, size / 2, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(size / 2, size / 2, size / 2, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-size / 2, size / 2, size / 2, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "insert.stl"))
    gmsh.model.remove()


def create_meshes(
    size: float, meshsize: float = 0.001, out_dir: str | pathlib.Path = "meshes"
) -> None:
    """Create all required meshes with GMSH and save them to the specified directory.
    Args:
        size (float): Size of the walls and insert.
        meshsize (float, optional): Desired mesh resolution. Defaults to 0.001.
        out_dir (str | pathlib.Path, optional): Directory to save the meshes. Defaults to "meshes".
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GMSH only once
    gmsh.initialize(sys.argv)

    # First wall
    create_topwall(size, meshsize, out_dir)
    # Second wall
    create_bottomwall(size, meshsize, out_dir)
    # Insert
    create_insert(size, meshsize, out_dir)

    # Finalize GMSH
    gmsh.finalize()

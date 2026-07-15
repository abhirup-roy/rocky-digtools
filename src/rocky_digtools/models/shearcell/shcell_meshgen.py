import fcntl
import os
import pathlib

import gmsh
import numpy as np
from stl.mesh import Mesh


class ParallelPlateMesh:
    """
    Class to generate a parallel plate mesh with vanes using gmsh.
    The mesh is generated in a box of specified width. The plates are
    generated at the top and bottom of the box, and specified number of
    regularly spaced vanes are generated.

    Parameters
    ----------
    box_width : float
        Width of the box in which the particles are generated.
    n_vanes : int
        Number of vanes to be generated.
        DEFAULT: 25
    h_vane_frac : float
        Height of the vanes as a fraction of the box width.
        DEFAULT: 0.1
    plate_length : float
        Length of the plate (in meters).
        DEFAULT: 1.0
    mesh_size : float
        Mesh size in Gmsh.
        DEFAULT: 0.001
    save_dir : str
        Directory to save the generated mesh files.
        DEFAULT: 'meshes'
    run : bool
        Whether to run the mesh generation immediately. If False,
        the user can call the gen_topwall and gen_bottomwall methods
        separately.
        DEFAULT: True
    bottom_vanes : bool
        Whether to generate the bottom vanes.
        DEFAULT: False

    Example usage
    --------------
    >>> from parallelplate_meshgen import ParallelPlateMesh
    >>> ppm = ParallelPlateMesh(box_width, run=True)
    """

    def __init__(
        self,
        box_width: float,
        l_vanes_frac: float = 0.2,
        h_vane_frac: float = 0.1,
        plate_length: float = 1.0,
        mesh_size: float = 0.001,
        save_dir: str = "meshes",
        run: bool = True,
        bottom_vanes: bool = True,
        top_mvt: bool = False,
        best_mesh_length: bool = True,
        vane_thickness: float = 1e-5,
    ):
        self.box_width = box_width
        self.mesh_size = mesh_size
        self.h_vane_frac = h_vane_frac
        self.h_vane = box_width * h_vane_frac
        self.plate_length = plate_length
        self.save_dir = save_dir
        self.bottom_vanes = bottom_vanes
        self.l_vanes_frac = l_vanes_frac
        self.top_mvt = top_mvt
        self.vane_thickness = vane_thickness

        if best_mesh_length:
            self.plate_length = self._best_mesh_length(box_width, max_sim_dur=5)

        if run:
            self.gen_topwall(gui=False)
            self.gen_bottomwall(gui=False)
            self.gen_insert(gui=False)

    def gen_topwall(self, gui: bool = False):
        """
        Generate the top wall of the parallel plate mesh with vanes.

        Parameters
        ----------
        gui : bool
            Whether to run gmsh in GUI mode. If True, the GUI will be
            displayed and the mesh will not be saved to a file.
            DEFAULT: False
        """
        # Determine z-interval for the top wall based on top_mvt
        if self.top_mvt:
            # Moving top: wall spans from z = box_width/2 - plate_length to z = box_width/2
            z1 = self.box_width / 2.0
            z0 = z1 - self.plate_length
        else:
            # Centered top: wall spans from -plate_length/2 to +plate_length/2
            z0 = -self.plate_length / 2.0
            z1 = +self.plate_length / 2.0
        # Geometry parameters
        box_w = self.box_width
        l_vane = self.l_vanes_frac * self.box_width

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("topwall")

        plane = gmsh.model.occ.addBox(
            x=-self.box_width / 2,
            y=self.box_width / 2,
            z=z0,
            dx=box_w,
            dy=5e-4,
            dz=z1 - z0,
        )

        comp_coll = []
        l_vane = self.l_vanes_frac * self.box_width
        vane_range = np.arange(start=z0, stop=z1, step=l_vane)

        for z in vane_range:
            vane = gmsh.model.occ.addBox(
                x=-box_w / 2,
                y=self.box_width / 2,
                z=z,
                dx=box_w,
                dy=-self.h_vane,
                dz=self.vane_thickness,
            )
            comp_coll.append((3, vane))

        gmsh.model.occ.fuse([(3, plane)], comp_coll)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)

        if gui:
            gmsh.fltk.run()
        else:
            pwd = os.getcwd()
            save_dir = os.path.join(pwd, self.save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gmsh.write(os.path.join(save_dir, "topwall.stl"))

        gmsh.finalize()

    def gen_bottomwall(self, gui: bool = False):
        """
        Generate the bottom wall of the parallel plate mesh with vanes.

        Parameters
        ----------
        gui : bool
            Whether to run gmsh in GUI mode. If True, the GUI will be
            displayed and the mesh will not be saved to a file.
            DEFAULT: False
        """
        endpoints = [z1 := self.box_width / 2.0, z1 - self.plate_length]

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("bottomwall")

        p1 = gmsh.model.geo.addPoint(
            self.box_width / 2,
            -self.box_width / 2,
            endpoints[0],
            meshSize=self.mesh_size,
        )
        p2 = gmsh.model.geo.addPoint(
            -self.box_width / 2,
            -self.box_width / 2,
            endpoints[0],
            meshSize=self.mesh_size,
        )
        p3 = gmsh.model.geo.addPoint(
            -self.box_width / 2,
            -self.box_width / 2,
            endpoints[1],
            meshSize=self.mesh_size,
        )

        p4 = gmsh.model.geo.addPoint(
            self.box_width / 2,
            -self.box_width / 2,
            endpoints[1],
            meshSize=self.mesh_size,
        )

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.addSurfaceLoop([surf])

        # Create the vanes if specified
        if self.bottom_vanes:
            l_vane = self.l_vanes_frac * self.box_width
            vane_range = np.arange(start=endpoints[1], stop=endpoints[0], step=l_vane)

            for length_coord in vane_range:
                vane_p1 = gmsh.model.geo.addPoint(
                    self.box_width / 2,
                    -self.box_width / 2,
                    length_coord,
                    meshSize=self.mesh_size,
                )
                vane_p2 = gmsh.model.geo.addPoint(
                    self.box_width / 2,
                    -self.box_width / 2 + self.h_vane,
                    length_coord,
                    meshSize=self.mesh_size,
                )
                vane_p3 = gmsh.model.geo.addPoint(
                    -self.box_width / 2,
                    -self.box_width / 2 + self.h_vane,
                    length_coord,
                    meshSize=self.mesh_size,
                )
                vane_p4 = gmsh.model.geo.addPoint(
                    -self.box_width / 2,
                    -self.box_width / 2,
                    length_coord,
                    meshSize=self.mesh_size,
                )

                vane_l1 = gmsh.model.geo.addLine(vane_p1, vane_p2)
                vane_l2 = gmsh.model.geo.addLine(vane_p2, vane_p3)
                vane_l3 = gmsh.model.geo.addLine(vane_p3, vane_p4)
                vane_l4 = gmsh.model.geo.addLine(vane_p4, vane_p1)

                vane_cl = gmsh.model.geo.addCurveLoop(
                    [vane_l1, vane_l2, vane_l3, vane_l4]
                )
                vane_surf = gmsh.model.geo.addPlaneSurface([vane_cl])
                gmsh.model.geo.addSurfaceLoop([vane_surf])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        if gui:
            gmsh.fltk.run()
        else:
            pwd = os.getcwd()
            save_dir = os.path.join(pwd, self.save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gmsh.write(os.path.join(save_dir, "bottomwall.stl"))
        gmsh.finalize()

    def _best_mesh_length(self, box_width: float, max_sim_dur: float = 5) -> float:
        max_speed = 30 / 60_000  # 30 mm/min
        # ensure enough length for 2 shear cycles
        max_length = (max_speed * max_sim_dur) * 2 + box_width
        max_length *= 1.1  # add 10% for safety

        return max_length

    def gen_insert(self, gui: bool = False):
        gmsh.initialize()
        gmsh.model.add("insert")

        # Keep the inlet centered in the clearance between the top wall
        # (y = box_width / 2) and the domain ceiling (y = 3 * box_width / 4).
        inlet_y = self.box_width * 5 / 8

        p1 = gmsh.model.geo.addPoint(
            self.box_width / 2,
            inlet_y,
            self.box_width / 2,
            meshSize=self.mesh_size,
        )
        p2 = gmsh.model.geo.addPoint(
            -self.box_width / 2,
            inlet_y,
            self.box_width / 2,
            meshSize=self.mesh_size,
        )
        p3 = gmsh.model.geo.addPoint(
            -self.box_width / 2,
            inlet_y,
            -self.box_width / 2,
            meshSize=self.mesh_size,
        )

        p4 = gmsh.model.geo.addPoint(
            self.box_width / 2,
            inlet_y,
            -self.box_width / 2,
            meshSize=self.mesh_size,
        )

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.addSurfaceLoop([surf])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        if gui:
            gmsh.fltk.run()
        else:
            pwd = os.getcwd()
            save_dir = os.path.join(pwd, self.save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gmsh.write(os.path.join(save_dir, "insert.stl"))
        gmsh.finalize()


def create_meshes(
    size: float,
    meshsize: float = 0.001,
    out_dir: str | pathlib.Path = "meshes",
) -> ParallelPlateMesh:
    """Create the parallel-plate shear-cell meshes and save them to disk.

    Generates ``topwall.stl``, ``bottomwall.stl``, and ``insert.stl`` in the
    output directory via :class:`ParallelPlateMesh`.

    Args:
        size: Box width of the shear-cell domain (m).
        meshsize: Desired mesh resolution. Defaults to 0.001.
        out_dir: Directory to save the meshes. Defaults to ``"meshes"``.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm = ParallelPlateMesh(
        box_width=size,
        mesh_size=meshsize,
        save_dir=str(out_dir.resolve()),
        run=False,
    )
    with (out_dir / ".create_meshes.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all(
            (out_dir / filename).is_file()
            for filename in ("topwall.stl", "bottomwall.stl", "insert.stl")
        ):
            ppm.gen_topwall(gui=False)
            ppm.gen_bottomwall(gui=False)
            ppm.gen_insert(gui=False)

    return ppm


def get_mesh_metrics(mesh_filepath: str) -> dict:
    """
    Get mesh metrics from an STL file.

    Parameters
    ----------
    mesh_filepath : str
        Filepath to the STL mesh file.

    Returns
    -------
    dict
        A dictionary containing:
        - volume (float): Volume of the mesh.
        - cog (list[float]): Center of gravity coordinates [x, y, z].
        - inertia (list[float]): Inertia tensor components.
    """
    mesh = Mesh.from_file(mesh_filepath)
    volume, cog, inertia_mat = mesh.get_mass_properties()

    pmoment_inert = np.linalg.eig(inertia_mat).eigenvalues
    mesh_metrics = {
        "volume": volume.item(),
        "cog": cog.tolist(),
        "pmoment_inertia": pmoment_inert.tolist(),
    }

    return mesh_metrics

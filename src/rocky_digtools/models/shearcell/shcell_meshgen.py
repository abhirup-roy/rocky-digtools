import fcntl
import os
import pathlib
import warnings

import gmsh
import numpy as np
from stl.mesh import Mesh


def required_plate_length(box_width: float, t_shear: float, shear_vel: float) -> float:
    """Cover the box plus pre-shear and one parallel-shear translation."""
    return (box_width + shear_vel * (2 * t_shear)) * 1.1


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
        best_mesh_length: bool = True,
        vane_thickness: float = 1e-5,
        t_shear: float = 5.0,
        shear_vel: float = 30 / 60_000,
    ):
        self.box_width = box_width
        self.mesh_size = mesh_size
        self.h_vane_frac = h_vane_frac
        self.h_vane = box_width * h_vane_frac
        # Keep the fill floor fixed while starting the top teeth above the inlet.
        self.bottom_wall_y = -box_width / 2
        self.top_wall_y = box_width
        self.inlet_y = box_width * 5 / 8
        self.plate_length: int | float = plate_length
        self.save_dir = save_dir
        self.bottom_vanes = bottom_vanes
        self.l_vanes_frac = l_vanes_frac
        self.vane_thickness = vane_thickness

        if best_mesh_length:
            self.plate_length = self._best_mesh_length(
                box_width, t_shear=t_shear, shear_vel=shear_vel
            )

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
        # Centered top: wall spans from -plate_length/2 to +plate_length/2
        z0 = -self.box_width / 2.0
        z1 = +self.box_width / 2.0
        # Geometry parameters
        box_w = self.box_width
        l_vane = self.l_vanes_frac * self.box_width

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("topwall")

        plane = gmsh.model.occ.addBox(
            x=-self.box_width / 2,
            y=self.top_wall_y,
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
                y=self.top_wall_y,
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
            self.bottom_wall_y,
            endpoints[0],
            meshSize=self.mesh_size,
        )
        p2 = gmsh.model.geo.addPoint(
            -self.box_width / 2,
            self.bottom_wall_y,
            endpoints[0],
            meshSize=self.mesh_size,
        )
        p3 = gmsh.model.geo.addPoint(
            -self.box_width / 2,
            self.bottom_wall_y,
            endpoints[1],
            meshSize=self.mesh_size,
        )

        p4 = gmsh.model.geo.addPoint(
            self.box_width / 2,
            self.bottom_wall_y,
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
                    self.bottom_wall_y,
                    length_coord,
                    meshSize=self.mesh_size,
                )
                vane_p2 = gmsh.model.geo.addPoint(
                    self.box_width / 2,
                    self.bottom_wall_y + self.h_vane,
                    length_coord,
                    meshSize=self.mesh_size,
                )
                vane_p3 = gmsh.model.geo.addPoint(
                    -self.box_width / 2,
                    self.bottom_wall_y + self.h_vane,
                    length_coord,
                    meshSize=self.mesh_size,
                )
                vane_p4 = gmsh.model.geo.addPoint(
                    -self.box_width / 2,
                    self.bottom_wall_y,
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

    def _best_mesh_length(
        self, box_width: float, t_shear: float, shear_vel: float
    ) -> float:
        return required_plate_length(box_width, t_shear, shear_vel)

    def gen_insert(self, gui: bool = False):
        gmsh.initialize()
        gmsh.model.add("insert")

        # The inlet stays above the fill box and below the raised top-wall teeth.
        inlet_y = self.inlet_y

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
    t_shear: float = 5.0,
    shear_vel: float = 30 / 60_000,
) -> ParallelPlateMesh:
    """Create the parallel-plate shear-cell meshes and save them to disk.

    Generates ``topwall.stl``, ``bottomwall.stl``, and ``insert.stl`` in the
    output directory via :class:`ParallelPlateMesh`.

    Args:
        size: Box width of the shear-cell domain (m).
        meshsize: Desired mesh resolution. Defaults to 0.001.
        out_dir: Directory to save the meshes. Defaults to ``"meshes"``.
        t_shear: Duration of each shear phase (s). The plate accommodates two
            phases: pre-shear and one subsequent parallel shear.
        shear_vel: Bottom-wall shear velocity (m/s).
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plate_length = required_plate_length(size, t_shear, shear_vel)

    ppm = ParallelPlateMesh(
        box_width=size,
        mesh_size=meshsize,
        save_dir=str(out_dir.resolve()),
        run=False,
        t_shear=t_shear,
        shear_vel=shear_vel,
    )
    with (out_dir / ".create_meshes.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        length_file = out_dir / ".plate_length"
        try:
            existing_length = float(length_file.read_text())
        except (FileNotFoundError, ValueError):
            existing_length = 0.0
        if existing_length < plate_length or not all(
            (out_dir / filename).is_file()
            for filename in ("topwall.stl", "bottomwall.stl", "insert.stl")
        ):
            ppm.gen_topwall(gui=False)
            ppm.gen_bottomwall(gui=False)
            ppm.gen_insert(gui=False)
            length_file.write_text(str(plate_length))

    return ppm


def get_mesh_metrics(mesh_filepath: str, density: float = 2700.0) -> dict:
    """
    Get mesh metrics from an STL file.

    Parameters
    ----------
    mesh_filepath : str
        Filepath to the STL mesh file.
    density : float
        Material density (kg/m3) used for the mass and inertia.
        Defaults to 2700 (aluminium).

    Returns
    -------
    dict
        A dictionary containing:
        - volume (float): Volume of the mesh.
        - mass (float): Mass of the mesh at ``density``.
        - cog (list[float]): Center of gravity coordinates [x, y, z].
        - pmoment_inertia (list[float]): Moments of inertia [Ixx, Iyy, Izz]
          about the COG at ``density`` (kg m2).
    """
    mesh = Mesh.from_file(mesh_filepath)
    # numpy-stl returns the inertia tensor about the COG for unit density
    volume, cog, inertia_mat = mesh.get_mass_properties()
    inertia_mat = np.asarray(inertia_mat, dtype=float) * density

    # Rocky takes the moments along x, y, z, so the principal axes of the
    # mesh must be (close to) aligned with them
    diag = np.diag(inertia_mat)
    off_diag = inertia_mat - np.diag(diag)
    if np.abs(off_diag).max() > 1e-2 * np.abs(diag).max():
        warnings.warn(
            f"Principal axes of {mesh_filepath} are not aligned with x, y, z; "
            "using the diagonal of the inertia tensor."
        )

    mesh_metrics = {
        "volume": volume.item(),
        "mass": volume.item() * density,
        "cog": cog.tolist(),
        "pmoment_inertia": diag.tolist(),
    }

    return mesh_metrics

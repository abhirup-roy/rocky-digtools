"""Pyrocky API wrapper for uniaxial compression simulations.

Defines a :class:`Settings` dataclass for storing simulation parameters and a
:class:`UniaxialCompressionSimulation` class that encapsulates the entire
workflow of configuring, running, and post-processing a uniaxial compression
test in Ansys Rocky.
"""

import json
import pathlib
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...pyrocky.helpers import pyrocky_run
from .. import PyrockySimulation
from .compr_meshgen import create_meshes

__all__ = ["Settings", "UniaxialCompressionSimulation"]


@dataclass(slots=True)
class Settings:
    """Simulation parameters for a uniaxial compression test.

    Attributes:
        project_dir: Directory where the Rocky project is saved.
        particle_box_len: Side length of the cubic particle domain (m).
        t_fill: Duration of the particle fill phase (s).
        t_settle: Duration of the settling phase (s).
        t_compress: Duration of the compression phase (s).
        p_compress: Applied compression pressure (Pa).
        p_radius: Particle radius (m) or a dict mapping radii to
            probabilities for polydisperse distributions.
        p_density: Particle density (kg/m³).
        p_youngmod: Particle Young's modulus (Pa).
        p_poisson: Particle Poisson's ratio.
        fric_dyn_pp: Dynamic friction coefficient (particle–particle).
        fric_stat_pp: Static friction coefficient (particle–particle).
        cor_pp: Coefficient of restitution (particle–particle).
        fric_dyn_pw: Dynamic friction coefficient (particle–wall).
        fric_stat_pw: Static friction coefficient (particle–wall).
        cor_pw: Coefficient of restitution (particle–wall).
        normal_force_model: Normal contact force model.
        tangential_force_model: Tangential contact force model.
        adhesion_model: Adhesion model.
        rolling_fric: Rolling friction coefficient.
        rolling_model: Rolling resistance model.
        neighbor_search: Neighbour-search algorithm.
        processor: Compute target (``"CPU"`` or ``"GPU"``).
        mesh_dir: Path to the mesh directory. Auto-resolved if ``None``.
        plots_dir: Path to the plots directory. Auto-resolved if ``None``.
        shape_name: Particle shape identifier.
        vert_ar: Vertical aspect ratio.
        horiz_ar: Horizontal aspect ratio.
        n_corners: Number of corners for polyhedral shapes.
        sq_degree: Superquadric degree.
        particle_path: Path to an STL file for custom polyhedra.
        smoothness: Surface smoothness parameter.
    """

    project_dir: str | pathlib.Path

    particle_box_len: float
    t_fill: float
    t_settle: float
    t_compress: float
    p_compress: float

    p_radius: float | dict
    p_density: float
    p_youngmod: float
    p_poisson: float
    fric_dyn_pp: float
    fric_stat_pp: float
    cor_pp: float
    fric_dyn_pw: float
    fric_stat_pw: float
    cor_pw: float

    normal_force_model: Literal["linear_hysteresis", "hertz", "linear_spring"] = (
        "linear_hysteresis"
    )
    tangential_force_model: Literal["coulomb_limit", "linear_spring_coulomb_limit"] = (
        "coulomb_limit"
    )
    adhesion_model: Literal["none", "constant", "linear", "JKR"] = "none"
    # Rolling friction off by default, unreliable for polyhedra
    rolling_fric: float = 0.0
    rolling_model: Literal["none", "type_a", "type_b"] = "none"
    neighbor_search: Literal["BVH", "RegularGrid", "SparseGrid"] = "BVH"
    processor: Literal["CPU", "GPU"] = "GPU"

    mesh_dir: Optional[str | pathlib.Path] = None
    plots_dir: Optional[str | pathlib.Path] = None

    shape_name: Literal[
        "sphere", "polyhedron", "sphero_cylinder", "custom_polyhedron"
    ] = "sphere"
    vert_ar: float = 1.0  # vertical aspect ratio for shaped particles
    horiz_ar: float = 1.0  # horizontal aspect ratio for shaped particles
    n_corners: int = 30  # number of corners for polyhedral particles
    sq_degree: float = 2.0  # superquadric degree for shaped particles
    particle_path: Optional[str] = (
        None  # path to custom particle STL file, required if shape_name is 'custom_polyhedron'
    )
    smoothness: Optional[float] = None

    def __post_init__(self):
        if self.mesh_dir is None:
            self.mesh_dir = (
                pathlib.Path(self.project_dir).parent
                / f"meshes_{self.particle_box_len}"
            )
        else:
            self.mesh_dir = pathlib.Path(self.mesh_dir)

        if not self.mesh_dir.exists():
            create_meshes(
                size=self.particle_box_len,
                out_dir=self.mesh_dir,
            )

        if not self.plots_dir:
            self.plots_dir = pathlib.Path(self.project_dir).parent / "plots"
        else:
            self.plots_dir = pathlib.Path(self.plots_dir)

        self.project_dir = pathlib.Path(self.project_dir)
        self._validate()

    def _validate(self):
        """Validate all parameter constraints and raise on failure.

        Raises:
            ValueError: If any parameter violates its constraints, with a
                summary of all errors.
        """
        errors = []

        # --- Positive floats ---
        positive_fields = {
            "particle_box_len": self.particle_box_len,
            "t_fill": self.t_fill,
            "t_settle": self.t_settle,
            "t_compress": self.t_compress,
            "p_compress": self.p_compress,
            "p_density": self.p_density,
            "p_youngmod": self.p_youngmod,
        }
        for name, val in positive_fields.items():
            if val <= 0:
                errors.append(f"'{name}' must be > 0, got {val}.")

        # --- Bounded [0, 0.5] fields ---
        poisson_fields = {
            "p_poisson": self.p_poisson,
        }
        for name, val in poisson_fields.items():
            if not (0.0 <= val <= 0.5):
                errors.append(f"'{name}' must be in [0, 0.5], got {val}.")

        # --- Bounded [0, 1] fields ---
        unit_fields = {
            "cor_pp": self.cor_pp,
            "cor_pw": self.cor_pw,
        }
        for name, val in unit_fields.items():
            if not (0.0 <= val <= 1.0):
                errors.append(f"'{name}' must be in [0, 1], got {val}.")

        # --- Non-negative floats ---
        nonneg_fields = {
            "fric_dyn_pp": self.fric_dyn_pp,
            "fric_stat_pp": self.fric_stat_pp,
            "fric_dyn_pw": self.fric_dyn_pw,
            "fric_stat_pw": self.fric_stat_pw,
            "rolling_fric": self.rolling_fric,
            "vert_ar": self.vert_ar,
            "horiz_ar": self.horiz_ar,
        }
        for name, val in nonneg_fields.items():
            if val < 0:
                errors.append(f"'{name}' must be >= 0, got {val}.")

        # --- p_radius ---
        if isinstance(self.p_radius, float):
            if self.p_radius <= 0:
                errors.append(f"'p_radius' must be > 0, got {self.p_radius}.")
        elif isinstance(self.p_radius, dict):
            if not self.p_radius:
                errors.append("'p_radius' dict must not be empty.")
            else:
                if any(r <= 0 for r in self.p_radius.keys()):
                    errors.append("All radii in 'p_radius' dict must be > 0.")
                prob_sum = sum(self.p_radius.values())
                if not (np.isclose(prob_sum, 1.0) or np.isclose(prob_sum, 100.0)):
                    errors.append(
                        f"'p_radius' probabilities must sum to 1 or 100, got {prob_sum}."
                    )
        else:
            errors.append(
                f"'p_radius' must be a float or dict, got {type(self.p_radius)}."
            )

        # --- Enum-like string fields ---
        valid_normal = {"linear_hysteresis", "hertz", "linear_spring"}
        if self.normal_force_model not in valid_normal:
            errors.append(
                f"'normal_force_model' must be one of {valid_normal}, "
                f"got '{self.normal_force_model}'."
            )

        valid_tangential = {"coulomb_limit", "linear_spring_coulomb_limit"}
        if self.tangential_force_model not in valid_tangential:
            errors.append(
                f"'tangential_force_model' must be one of {valid_tangential}, "
                f"got '{self.tangential_force_model}'."
            )

        valid_adhesion = {"none", "constant", "linear", "JKR"}
        if self.adhesion_model not in valid_adhesion:
            errors.append(
                f"'adhesion_model' must be one of {valid_adhesion}, "
                f"got '{self.adhesion_model}'."
            )

        valid_rolling = {"none", "type_a", "type_b"}
        if self.rolling_model not in valid_rolling:
            errors.append(
                f"'rolling_model' must be one of {valid_rolling}, "
                f"got '{self.rolling_model}'."
            )

        valid_neighbor = {"BVH", "RegularGrid", "SparseGrid"}
        if self.neighbor_search not in valid_neighbor:
            errors.append(
                f"'neighbor_search' must be one of {valid_neighbor}, "
                f"got '{self.neighbor_search}'."
            )

        valid_processor = {"CPU", "GPU"}
        if self.processor not in valid_processor:
            errors.append(
                f"'processor' must be one of {valid_processor}, got '{self.processor}'."
            )

        valid_shapes = {"sphere", "polyhedron", "sphero_cylinder", "custom_polyhedron"}
        if self.shape_name not in valid_shapes:
            errors.append(
                f"'shape_name' must be one of {valid_shapes}, got '{self.shape_name}'."
            )

        if self.shape_name == "custom_polyhedron":
            if not self.particle_path:
                errors.append(
                    "'particle_path' must be provided when shape_name is 'custom_polyhedron'."
                )
            elif not pathlib.Path(self.particle_path).is_file():
                errors.append(
                    f"'particle_path' does not point to a valid file: {self.particle_path}"
                )

        if self.n_corners < 10:
            errors.append(f"'n_corners' must be >= 10, got {self.n_corners}.")

        if self.sq_degree < 2.0:
            errors.append(f"'sq_degree' must be >= 2.0, got {self.sq_degree}.")

        if errors:
            raise ValueError(
                "Invalid Settings:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @property
    def expected_particle_volume(self) -> float:
        """Expected particle volume, weighting radii for polydisperse distributions."""
        if isinstance(self.p_radius, (int, float)):
            return (4 / 3) * np.pi * self.p_radius**3
        radii = np.array(list(self.p_radius.keys()))
        probs = np.array(list(self.p_radius.values()))
        return float(np.sum((4 / 3) * np.pi * radii**3 * probs / probs.sum()))

    @property
    def avg_particle_radius(self) -> float:
        """Mean particle radius, weighted by probability for polydisperse distributions.

        For a monodisperse distribution this simply returns the scalar radius.
        For a polydisperse dict it computes the probability-weighted mean.

        Returns:
            The average particle radius in metres.
        """
        if isinstance(self.p_radius, (int, float)):
            return self.p_radius
        radii = np.array(list(self.p_radius.keys()))
        probs = np.array(list(self.p_radius.values()))
        return float(np.sum(radii * probs / probs.sum()))

    @classmethod
    def from_json(
        cls, path: str | pathlib.Path, project_dir: str | pathlib.Path
    ) -> "Settings":
        """Create a :class:`Settings` instance from a JSON configuration file.

        Args:
            path: Path to the JSON configuration file.
            project_dir: Directory where the Rocky project will be saved.

        Returns:
            A new ``Settings`` instance populated from the file.
        """
        with open(path, "r") as f:
            data = json.load(f)

        shape = data["shape"]
        props = data["particle_properties"]
        inter = data["inseractions"]  # note: typo in JSON preserved
        exp = data["experim_settings"]
        contact = data["contact_model"]

        return cls(
            project_dir=project_dir,
            # Particle shape
            shape_name=shape["name"],
            vert_ar=shape.get("vert_ar", 1.0),
            horiz_ar=shape.get("horiz_ar", 1.0),
            n_corners=shape.get("n_corners", 30),
            sq_degree=shape.get("sq_degree", 2.0),
            # Particle properties
            p_radius=props["radius"],
            p_density=props["density"],
            p_poisson=props["poisson"],
            p_youngmod=props["youngmod"],
            # Interactions
            fric_dyn_pp=inter["pp"]["fric_dyn"],
            fric_stat_pp=inter["pp"]["fric_stat"],
            cor_pp=inter["pp"]["cor"],
            rolling_fric=inter["pp"].get("fric_rolling", 0.0),
            fric_dyn_pw=inter["pw"]["fric_dyn"],
            fric_stat_pw=inter["pw"]["fric_stat"],
            cor_pw=inter["pw"]["cor"],
            # Experiment settings
            particle_box_len=exp["box_len"],
            p_compress=exp["p_compress"],
            t_fill=exp.get("t_fill", 1.0),
            t_settle=exp.get("t_settle", 0.5),
            t_compress=exp.get("t_compress", 2.0),
            # Contact models
            normal_force_model=contact["normal"],
            tangential_force_model=contact["tangential"],
            rolling_model=contact["rolling"],
            adhesion_model=contact["adhesion"],
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        """Create a :class:`Settings` instance from a dictionary.

        Args:
            data: Dictionary of field names to values.

        Returns:
            A new ``Settings`` instance.

        Raises:
            ValueError: If any required field is missing.
        """
        required_fields = [
            f.name
            for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
        ]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields for Settings: {missing_fields}")

        return cls(**data)


@pyrocky_run()
class UniaxialCompressionSimulation(PyrockySimulation):
    """End-to-end uniaxial compression simulation in Ansys Rocky.

    Wraps the full workflow — project creation, mesh loading, material and
    interaction setup, particle generation, physics configuration, domain
    settings, simulation execution, and post-processing — into a single
    class.

    Args:
        settings: Simulation parameters as a :class:`Settings` instance.
        filename: Name of the Rocky project file. Defaults to
            ``"uniaxial_compression.rocky"``.

    Attributes:
        rocky: The active Rocky API session (injected by
            :class:`~rocky_digtools.pyrocky.helpers.pyrocky_run`).
        settings: The simulation parameters.
        filename: Project file name.
    """

    rocky: Any

    def __init__(
        self,
        settings: Settings,
        filename: str = "uniaxial_compression.rocky",
    ) -> None:
        super().__init__(
            filename,
            settings,
            "Uniaxial Compression",
        )

    def load_meshes(self) -> None:
        """Import the top wall, bottom wall, and optionally the insert surface."""
        assert self.settings.mesh_dir is not None
        mesh_dir = pathlib.Path(self.settings.mesh_dir).resolve()

        top_wall_path = mesh_dir / "compressive_wall1.stl"
        top_wall = self._study.ImportWall(
            str(top_wall_path), import_scale=1.1, convert_yz=False
        )[0]
        top_wall.SetName("Top Wall")
        top_wall.SetBoundaryMass(1e-6)
        top_wall.SetTranslation([0, self.settings.particle_box_len / 2 + 1e-6, 0])

        bottom_wall_path = mesh_dir / "compressive_wall2.stl"
        bottom_wall = self._study.ImportWall(
            str(bottom_wall_path), import_scale=1.1, convert_yz=False
        )[0]
        bottom_wall.SetName("Bottom Wall")

        # Store proxies directly — only used within same session, never passed cross-IPC
        self._mesh["top_wall"] = top_wall
        self._mesh["bottom_wall"] = bottom_wall

        insert_stl_path = (mesh_dir / "insert.stl").resolve()
        insert_inlet = self._study.ImportSurface(
            str(insert_stl_path), import_scale=1.0, convert_yz=True
        )[0]
        insert_inlet.SetName("Insert Inlet")  # <-- set explicit name
        insert_inlet.SetPivotPoint([0, 0, 0])

        current_height = insert_inlet.GetVertices().mean(axis=0)[1]
        target_height = (self.settings.particle_box_len / 2) * 0.99

        insert_inlet.SetTranslation([0, float(target_height - current_height), 0])
        insert_inlet.SetInvertNormal(True)
        self._mesh["insert_inlet"] = insert_inlet

    def move_top_wall(self) -> None:
        """Apply a motion frame to the top wall for settling and compression."""
        frame_source = self._study.GetMotionFrameSource()
        top_wall_frame = frame_source.NewFrame()

        motions = top_wall_frame.GetMotions()

        # drop almost weightless wall
        drop_wall_motion = motions.New()
        drop_wall_motion.SetType("Free Body Translation")
        free_body = drop_wall_motion.GetTypeObject()
        free_body.SetFreeMotionDirection("y")
        drop_wall_motion.SetStartTime(self.settings.t_fill + self.settings.t_settle)

        f_compr = (
            1e-6 * 9.81 - self.settings.p_compress * self.settings.particle_box_len**2
        )
        compr_motion = motions.New()
        compr_motion.SetType("Additional Force")
        add_force = compr_motion.GetTypeObject()
        add_force.SetForceValue([0, f_compr, 0])

        start_time = self.settings.t_fill + self.settings.t_settle + 0.1
        end_time = start_time + self.settings.t_compress
        compr_motion.SetStartTime(start_time)
        compr_motion.SetStopTime(end_time)

        top_wall_frame.ApplyTo(self._ser(self._mesh["top_wall"]))

    def set_domain_settings(self):
        """Configure the simulation domain bounds and periodic boundaries."""
        domain_settings = self._study.GetDomainSettings()
        domain_settings.DisableUseBoundaryLimits()
        domain_settings.DisablePeriodicAtGeometryLimits()

        domain_settings.SetDomainType("CARTESIAN")
        domain_settings.SetCoordinateLimitsMinValues(
            [
                (-self.settings.particle_box_len / 2) * 1.5,
                (-self.settings.particle_box_len / 2) * 1.5,
                (-self.settings.particle_box_len / 2) * 1.5,
            ]
        )
        domain_settings.SetCoordinateLimitsMaxValues(
            [
                (self.settings.particle_box_len / 2) * 1.5,
                (self.settings.particle_box_len / 2) * 1.5,
                (self.settings.particle_box_len / 2) * 1.5,
            ]
        )

        domain_settings.SetCartesianPeriodicDirections("XZ")
        domain_settings.SetPeriodicLimitsMinCoordinates(
            [
                -self.settings.particle_box_len / 2,
                -1e-6,
                -self.settings.particle_box_len / 2,
            ]
        )
        domain_settings.SetPeriodicLimitsMaxCoordinates(
            [
                self.settings.particle_box_len / 2,
                1e-6,
                self.settings.particle_box_len / 2,
            ]
        )

    def load_modules(self):
        """Enable contacts data collection and adhesive contact reporting."""
        contacts_data = self._study.GetContactData()
        contacts_data.EnableCollectContactsData()
        if self.settings.adhesion_model != "none":
            contacts_data.EnableIncludeAdhesiveContacts()

    def _get_cropped_region(self, particles, time_step, sample_frac=0.9):
        """Get or create a cropped cuboid region for sampling.

        The region is centred on the mean particle position and spans
        ``sample_frac`` of the particle coordinate range in each direction.
        Results are cached per time step.

        Args:
            particles: Rocky particles collection.
            time_step: Time-step index.
            sample_frac: Fraction of the coordinate range to sample.

        Returns:
            A Rocky cube-process region object.
        """
        if time_step in self.active_boxes:
            return self.active_boxes[time_step]

        x_coords = particles.GetGridFunction("Coordinate : X").GetArray(
            time_step=time_step
        )
        y_coords = particles.GetGridFunction("Coordinate : Y").GetArray(
            time_step=time_step
        )
        z_coords = particles.GetGridFunction("Coordinate : Z").GetArray(
            time_step=time_step
        )

        positions = np.vstack((x_coords, y_coords, z_coords))
        pos_rngs = np.ptp(positions, axis=1)
        sample_rng = pos_rngs * sample_frac

        processes = self._project.GetUserProcessCollection()

        cube_selection = processes.CreateCubeProcess(particles)
        cube_selection.SetCenter(x_coords.mean(), y_coords.mean(), z_coords.mean())
        cube_selection.SetSize(sample_rng[0], sample_rng[1], sample_rng[2])

        self.active_boxes[time_step] = cube_selection

        return cube_selection

    def _calc_bulk_density(self, particles, time_step, sample_frac=0.9):
        """Calculate the bulk density within a cropped region.

        Args:
            particles: Rocky particles collection.
            time_step: Time-step index.
            sample_frac: Fraction of the domain to sample.

        Returns:
            Bulk density in kg/m³.
        """

        cube_selection = self._get_cropped_region(particles, time_step, sample_frac)

        mass_arr = cube_selection.GetGridFunction("Particle Mass").GetArray(
            time_step=time_step
        )
        sample_mass = mass_arr.sum()

        sample_rng = cube_selection.GetSize()
        sample_vol = np.prod(sample_rng)

        return sample_mass / sample_vol

    def _calc_contact_no(self, particles, time_step, sample_frac=0.9):
        """Calculate the average number of contacts per particle.

        Args:
            particles: Rocky particles collection.
            time_step: Time-step index.
            sample_frac: Fraction of the domain to sample.

        Returns:
            Average contacts per particle.
        """
        cube_selection = self._get_cropped_region(particles, time_step, sample_frac)
        contact_data = self._study.GetContactData()

        all_contacts_x = contact_data.GetGridFunction(
            "Contact : Coordinate : X"
        ).GetArray(time_step=time_step)
        all_contacts_y = contact_data.GetGridFunction(
            "Contact : Coordinate : Y"
        ).GetArray(time_step=time_step)
        all_contacts_z = contact_data.GetGridFunction(
            "Contact : Coordinate : Z"
        ).GetArray(time_step=time_step)

        x_rng, y_rng, z_rng = cube_selection.GetSize()
        x_center, y_center, z_center = cube_selection.GetCenter()

        x_mask = (all_contacts_x >= x_center - x_rng / 2) & (
            all_contacts_x <= x_center + x_rng / 2
        )

        y_mask = (all_contacts_y >= y_center - y_rng / 2) & (
            all_contacts_y <= y_center + y_rng / 2
        )
        z_mask = (all_contacts_z >= z_center - z_rng / 2) & (
            all_contacts_z <= z_center + z_rng / 2
        )

        n_contacts = (
            np.logical_and.reduce((x_mask, y_mask, z_mask)).sum()
            * 2
            / cube_selection.GetNumberOfParticles(time_step=time_step)
        )

        return n_contacts

    def post_process(
        self,
        sample_frac: float = 0.9,
        plot: bool = True,
        return_computed_metrics: bool = False,
    ) -> tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """Post-process simulation results.

        Computes uncompressed and compressed bulk densities, contact numbers,
        and the contacts ratio.  Optionally generates time-series plots and
        appends results to a CSV file.

        Args:
            sample_frac: Fraction of the domain to sample for calculations.
                Defaults to 0.9.
            plot: If ``True``, generate and save time-series plots.
                Defaults to ``True``.
            return_computed_metrics: If ``True``, return computed metrics as
                a tuple. Defaults to ``False``.

        Returns:
            A 5-tuple ``(uncompr_dens, compr_dens, uncompr_contacts,
            compr_contacts, contacts_ratio)`` if
            ``return_computed_metrics`` is ``True``; otherwise a tuple of
            ``None`` values.

        Raises:
            IndexError: If the settled time step cannot be located.
        """
        time_set = self._study.GetTimeSet()
        timeset_arr = time_set.GetValues()
        target_time = self.settings.t_fill + self.settings.t_settle
        try:
            settled_timeset = np.argmin(np.abs(timeset_arr - target_time)).item()
            if abs(timeset_arr[settled_timeset] - target_time) > 1e-3:
                raise IndexError("Matched time step is too far from target time")
        except IndexError:
            raise IndexError(
                "Could not find time step corresponding to end of settling phase."
                "Please ensure that the time step exists and matches the sum of t_fill and t_settle parameters."
                f"Available time steps: {timeset_arr}"
            )

        particles = self._study.GetParticles()

        uncompr_dens = self._calc_bulk_density(
            particles, time_step=settled_timeset, sample_frac=sample_frac
        )
        compr_dens = self._calc_bulk_density(
            particles, time_step=-1, sample_frac=sample_frac
        )

        uncompr_contacts = self._calc_contact_no(
            particles, time_step=settled_timeset, sample_frac=sample_frac
        )
        compr_contacts = self._calc_contact_no(
            particles, time_step=-1, sample_frac=sample_frac
        )
        contacts_ratio = compr_contacts / uncompr_contacts

        # Handle plotting
        if plot:
            bulk_dens = []
            contacts = []

            for timestep in time_set[1:]:
                bulk_dens.append(
                    self._calc_bulk_density(particles, timestep, sample_frac)
                )
                contacts.append(self._calc_contact_no(particles, timestep, sample_frac))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_set[1:], bulk_dens, label="Bulk Density", color="C0")
            ax.set_xlabel("Time (s)", fontsize=16)
            ax.set_ylabel("Bulk Density (kg/m^3)", fontsize=16)

            ax1 = ax.twinx()
            ax1.plot(time_set[1:], contacts, color="C1", label="Average Contacts")
            ax1.set_ylabel("Average Number of Contacts", fontsize=16)

            ax.grid(visible=True)
            ax1.grid(visible=True)
            fig.legend()
            fig.tight_layout()
            fig.savefig(
                pathlib.Path(self.settings.plots_dir) / "ts_bulkdens_contacts.png",
                dpi=300,
            )

        # Write results row
        output_path = pathlib.Path(self.settings.project_dir) / "results.csv"
        row = pd.DataFrame(
            [
                {
                    **asdict(self.settings),
                    "uncompressed_density": uncompr_dens,
                    "compressed_density": compr_dens,
                    "uncompressed_contacts": uncompr_contacts,
                    "compressed_contacts": compr_contacts,
                    "contacts_ratio": contacts_ratio,
                }
            ]
        )
        row.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)

        if return_computed_metrics:
            return (
                uncompr_dens,
                compr_dens,
                uncompr_contacts,
                compr_contacts,
                contacts_ratio,
            )
        else:
            return (None, None, None, None, None)

    def execute(
        self,
        sample_frac: float = 0.9,
        plot: bool = True,
        return_computed_metrics: bool = False,
    ) -> Optional[
        tuple[
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
        ]
    ]:
        """Run the full simulation workflow from setup to post-processing.

        Sequentially calls :meth:`load_meshes`, :meth:`load_material_properties`,
        :meth:`load_interactions`, :meth:`gen_particle`, :meth:`sim_physics`,
        :meth:`insertion_settings`, :meth:`move_top_wall`,
        :meth:`set_domain_settings`, :meth:`load_modules`,
        :meth:`simulate`, and :meth:`post_process`.

        Args:
            sample_frac: Fraction of the domain to sample. Defaults to 0.9.
            plot: Whether to generate plots. Defaults to ``True``.
            return_computed_metrics: Whether to return computed metrics.
                Defaults to ``False``.

        Returns:
            The post-processing result tuple, or ``None`` if no metrics are
            computed.
        """
        self.load_meshes()
        self.load_material_properties()
        self.load_interactions()
        self.gen_particle()
        self.sim_physics()
        self.insertion_settings()
        self.move_top_wall()
        self.set_domain_settings()
        self.load_modules()

        self.simulate(
            sim_time=sum(
                [
                    self.settings.t_fill,
                    self.settings.t_settle,
                    self.settings.t_compress,
                ]
            ),
        )
        res = self.post_process(
            sample_frac=sample_frac,
            plot=plot,
            return_computed_metrics=return_computed_metrics,
        )

        if any(res):
            return res

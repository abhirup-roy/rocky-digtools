"""Pyrocky API wrapper for multiscale shear-cell simulations.

Defines a :class:`Settings` dataclass for storing simulation parameters and a
:class:`ShearCellSimulation` class that encapsulates the entire
workflow of configuring, running, and post-processing a shear-cell test in
Ansys Rocky.
"""

import os
import json
import glob
import sqlite3
import pathlib
import subprocess
from typing import Literal, Optional, Any
from dataclasses import dataclass, asdict, fields, MISSING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize

from ...particles_shapes import normalise_radius
from .shcell_meshgen import create_meshes, get_mesh_metrics
from ...pyrocky.helpers import pyrocky_run
from .. import PyrockySimulation

__all__ = ["Settings", "ShearCellSimulation", "run_shear_point", "aggregate_results"]


def _phase_times(t_fill: float, t_settle: float, t_compression: float, t_shear: float, insert: bool) -> tuple[float, float, float, float]:
    """Return cumulative fill, settle, compression, and shear end times."""
    fill_end = t_fill if insert else 0.0
    settle_end = fill_end + t_settle
    compression_end = settle_end + t_compression
    return fill_end, settle_end, compression_end, compression_end + t_shear


def _target_force(sigma: float, area: float, wall_mass: float) -> float:
    """Additional downward force needed for a target normal stress (N)."""
    return sigma * area - wall_mass * 9.81


@dataclass(slots=True)
class Settings:
    """Simulation parameters for a shear cell test."""

    project_dir: str | pathlib.Path
    particle_box_len: float
    t_settle: float
    t_compression: float
    sigma_pre: float
    n_shear_points: int
    n_procs: int
    t_shear: float
    shear_vel: float

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

    surf_en_pp: float = 0.0
    surf_en_pw: float = 0.0
    normal_force_model: Literal[
        "linear_hysteresis", "linear_elastic_viscous", "damped_hertzian", "custom"
    ] = "linear_hysteresis"
    tangential_force_model: Literal[
        "elastic_coulomb", "coulomb_limit", "mindlin_deresiewicz", "custom"
    ] = "coulomb_limit"
    adhesion_model: Literal["none", "constant", "linear", "JKR"] = "none"
    # Rolling friction off by default, unreliable for polyhedra
    rolling_fric: float = 0.0
    rolling_model: Literal["none", "type_1", "type_3", "custom"] = "none"
    neighbour_search: Literal["BVH", "RegularGrid", "SparseGrid"] = "BVH"
    processor: Literal["CPU", "GPU", "MULTI_GPU"] = "GPU"
    loc: str = "az-gpu"  # SLURM cluster target: 'bb-cpu', 'az-gpu', or 'custom'
    t_fill: float = 0.5  # constant fill time for shear template

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
        self.p_radius = normalise_radius(self.p_radius)
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

        positive_fields = {
            "particle_box_len": self.particle_box_len,
            "t_settle": self.t_settle,
            "t_compression": self.t_compression,
            "sigma_pre": self.sigma_pre,
            "t_shear": self.t_shear,
            "shear_vel": self.shear_vel,
            "p_density": self.p_density,
            "p_youngmod": self.p_youngmod,
        }
        for name, val in positive_fields.items():
            if val <= 0:
                errors.append(f"'{name}' must be > 0, got {val}.")

        poisson_fields = {
            "p_poisson": self.p_poisson,
        }
        for name, val in poisson_fields.items():
            if not (0.0 <= val <= 0.5):
                errors.append(f"'{name}' must be in [0, 0.5], got {val}.")

        unit_fields = {
            "cor_pp": self.cor_pp,
            "cor_pw": self.cor_pw,
        }
        for name, val in unit_fields.items():
            if not (0.0 <= val <= 1.0):
                errors.append(f"'{name}' must be in [0, 1], got {val}.")

        nonneg_fields = {
            "fric_dyn_pp": self.fric_dyn_pp,
            "fric_stat_pp": self.fric_stat_pp,
            "fric_dyn_pw": self.fric_dyn_pw,
            "fric_stat_pw": self.fric_stat_pw,
            "surf_en_pp": self.surf_en_pp,
            "surf_en_pw": self.surf_en_pw,
            "rolling_fric": self.rolling_fric,
            "vert_ar": self.vert_ar,
            "horiz_ar": self.horiz_ar,
        }
        for name, val in nonneg_fields.items():
            if val < 0:
                errors.append(f"'{name}' must be >= 0, got {val}.")

        valid_normal = {
            "linear_hysteresis",
            "linear_elastic_viscous",
            "damped_hertzian",
            "custom",
        }
        if self.normal_force_model not in valid_normal:
            errors.append(
                f"'normal_force_model' must be one of {valid_normal}, "
                f"got '{self.normal_force_model}'."
            )

        valid_tangential = {
            "elastic_coulomb",
            "coulomb_limit",
            "mindlin_deresiewicz",
            "custom",
        }
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
        elif self.adhesion_model == "JKR" and min(
            self.surf_en_pp, self.surf_en_pw
        ) <= 0:
            errors.append(
                "'surf_en_pp' and 'surf_en_pw' must be > 0 for the JKR model."
            )

        valid_rolling = {"none", "type_1", "type_3", "custom"}
        if self.rolling_model not in valid_rolling:
            errors.append(
                f"'rolling_model' must be one of {valid_rolling}, "
                f"got '{self.rolling_model}'."
            )

        valid_neighbour_search = {"BVH", "RegularGrid", "SparseGrid"}
        if self.neighbour_search not in valid_neighbour_search:
            errors.append(
                f"'neighbour_search' must be one of {valid_neighbour_search}, "
                f"got '{self.neighbour_search}'."
            )

        valid_processor = {"CPU", "GPU", "MULTI_GPU"}
        if self.processor not in valid_processor:
            errors.append(
                f"'processor' must be one of {valid_processor}, got '{self.processor}'."
            )

        valid_loc = {"bb-cpu", "az-gpu", "custom"}
        if self.loc not in valid_loc:
            errors.append(f"'loc' must be one of {valid_loc}, got '{self.loc}'.")

        if self.n_shear_points < 1:
            errors.append(f"'n_shear_points' must be >= 1, got {self.n_shear_points}.")
        if self.n_procs < 1:
            errors.append(f"'n_procs' must be >= 1, got {self.n_procs}.")

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
        inter = data["interactions"]
        exp = data["experiment_settings"]
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
            surf_en_pp=inter["pp"].get("surf_en", 0.0),
            rolling_fric=inter["pp"].get("fric_rolling", 0.0),
            fric_dyn_pw=inter["pw"]["fric_dyn"],
            fric_stat_pw=inter["pw"]["fric_stat"],
            cor_pw=inter["pw"]["cor"],
            surf_en_pw=inter["pw"].get("surf_en", 0.0),
            # Experiment settings
            particle_box_len=exp["box_len"],
            t_settle=exp["t_settle"],
            t_compression=exp["t_compression"],
            sigma_pre=exp["sigma_pre"],
            n_shear_points=exp["n_shear_points"],
            n_procs=exp["n_procs"],
            neighbour_search=exp["neighbour_search"],
            t_shear=exp["t_shear"],
            shear_vel=exp["shear_vel"],
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


def _preshear_filename(settings: Settings) -> pathlib.Path:
    return pathlib.Path(settings.project_dir) / "shear_cell_preshear.rocky"


@pyrocky_run()
class ShearCellSimulation(PyrockySimulation):
    """End-to-end parallel-plate shear-cell simulation in Ansys Rocky.

    The shear-cell protocol is a multi-stage experiment:

    1. **Pre-shear** — fill, settle, compress to ``sigma_pre``, then shear the
       bottom wall at ``shear_vel`` for ``t_shear``.  Runs once, in the
       calling process (a single GPU job).
    2. **Shear points** — a yield locus is traced by re-shearing the
       consolidated sample at progressively lower normal stresses
       ``sigma_pre`` → 0.  Each point is an independent restart case
       (:meth:`create_new_cases`) dispatched as a SLURM array job
       (:meth:`slurm_job`); each array task calls
       :func:`run_shear_point`.
    3. **Aggregate** — once every array task has written its shear stress,
       :func:`aggregate_results` fits the Mohr circles and dumps the
       flow-function metrics to a SQLite database.

    Args:
        settings: Simulation parameters as a :class:`Settings` instance.
        insertion: Whether to use surface insertion (``True``) or volumetric
            insertion (``False``).  Defaults to ``True``.
        filename: Name of the pre-shear Rocky project file.  Defaults to
            ``"shear_cell_preshear.rocky"``.

    Attributes:
        rocky: The active Rocky API session (injected by
            :class:`~rocky_digtools.pyrocky.helpers.pyrocky_run`).
        settings: The simulation parameters.
        insertion: Insertion mode flag.
        filename: Pre-shear project file name.
    """

    rocky: Any

    def __init__(
        self,
        settings: Settings,
        insertion: bool = True,
        filename: str = "shear_cell_preshear.rocky",
    ) -> None:
        self.settings = settings
        self.insertion = insertion
        self.filename = filename
        self._outputs_dir = pathlib.Path(settings.project_dir) / "pyoutputs"
        self._outputs_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=filename,
            settings=settings,
            study_name="Shear Cell",
            insertion=insertion,
        )

        self._case_dirs: list[pathlib.Path] = []

    #  Geometry / meshes
    def load_meshes(self, insert: bool = True) -> None:
        """Import the top (compressing) wall, bottom (shearing) wall, and inlet.

        Args:
            insert: If ``True``, also import the particle inlet surface.
                Defaults to ``True``.
        """
        assert self.settings.mesh_dir is not None
        mesh_dir = pathlib.Path(self.settings.mesh_dir).resolve()

        top_wall_path = mesh_dir / "topwall.stl"
        top_wall = self._study.ImportWall(
            str(top_wall_path), import_scale=1.0, convert_yz=False
        )[0]
        top_wall.SetName("Compression Wall 1")

        mesh_metrics = get_mesh_metrics(str(top_wall_path))
        top_wall.SetPrincipalMomentOfInertia(mesh_metrics["pmoment_inertia"])
        top_wall.SetGravityCenter(mesh_metrics["cog"])
        top_wall.SetBoundaryMass(mesh_metrics["volume"] * 2700.0)

        if insert:
            top_wall.SetEnableTime(self.settings.t_fill + 0.25)

        bottom_wall_path = mesh_dir / "bottomwall.stl"
        bottom_wall = self._study.ImportWall(
            str(bottom_wall_path), import_scale=1.0, convert_yz=False
        )[0]
        bottom_wall.SetName("Compression Wall 2")

        self._mesh["top_wall"] = top_wall
        self._mesh["bottom_wall"] = bottom_wall

        if insert:
            insert_path = (mesh_dir / "insert.stl").resolve()
            insert_inlet = self._study.ImportSurface(
                str(insert_path), import_scale=1.0, convert_yz=False
            )[0]
            insert_inlet.SetName("Insert Inlet")
            self._mesh["insert_inlet"] = insert_inlet

    def set_domain_settings(self):
        """Configure the simulation domain bounds and periodic boundaries.

        The shear cell uses ``XZ`` periodicity with unbounded ``Y`` limits so
        the shearing wall can travel freely along ``Z``.
        """
        domain_settings = self._study.GetDomainSettings()
        domain_settings.DisableUseBoundaryLimits()
        domain_settings.DisablePeriodicAtGeometryLimits()

        domain_settings.SetDomainType("CARTESIAN")
        box = self.settings.particle_box_len
        domain_settings.SetCoordinateLimitsMinValues(
            [(-box / 2) * 1.5, (-box / 2) * 1.5, (-box / 2) * 1.5]
        )
        domain_settings.SetCoordinateLimitsMaxValues(
            [(box / 2) * 1.5, (box / 2) * 1.5, (box / 2) * 1.5]
        )

        domain_settings.SetCartesianPeriodicDirections("XZ")
        domain_settings.SetPeriodicLimitsMinCoordinates([-box / 2, -np.inf, -box / 2])
        domain_settings.SetPeriodicLimitsMaxCoordinates([box / 2, np.inf, box / 2])

    #  Wall motion
    def move_walls(self, insert: bool = True) -> None:
        """Apply motion frames to the top (compression) and bottom (shear) walls.

        The top wall gets a free-body translation (settling) plus an
        additional compressive force held for the whole pre-shear.  The
        bottom wall gets a fixed-velocity translation along ``Z`` during the
        shear phase.

        Args:
            insert: If ``True``, timing accounts for the fill phase.
                Defaults to ``True``.
        """
        p = self.settings
        frame_source = self._study.GetMotionFrameSource()

        # top wall
        top_frame = frame_source.NewFrame()
        top_motions = top_frame.GetMotions()

        free_body_motion = top_motions.New()
        free_body_motion.SetType("Free Body Translation")
        free_body = free_body_motion.GetTypeObject()
        free_body.SetFreeMotionDirection("y")
        fill_end, settle_end, compression_end, shear_end = _phase_times(
            p.t_fill, p.t_settle, p.t_compression, p.t_shear, insert
        )
        free_body_motion.SetStartTime(fill_end)
        free_body_motion.SetStopTime(shear_end)

        wall_mass = self._mesh["top_wall"].GetBoundaryMass()
        force_magnitude = _target_force(p.sigma_pre, p.particle_box_len**2, wall_mass)
        force_motion = top_motions.New()
        force_motion.SetType("Additional Force")
        add_force = force_motion.GetTypeObject()
        add_force.SetForceValue([0, -force_magnitude, 0], "N")
        force_motion.SetStartTime(settle_end)
        force_motion.SetStopTime(shear_end)

        top_frame.ApplyTo(self._ser(self._mesh["top_wall"]))

        # bottom wall: shearing translation
        bottom_frame = frame_source.NewFrame()
        bottom_motions = bottom_frame.GetMotions()

        shear_motion = bottom_motions.New()
        shear_motion.SetType("Translation")
        translation = shear_motion.GetTypeObject()
        translation.SetInput("fixed_velocity")
        translation.SetVelocity([0, 0, p.shear_vel], "m/s")
        shear_motion.SetStartTime(compression_end)
        shear_motion.SetStopTime(shear_end)

        bottom_frame.ApplyTo(self._ser(self._mesh["bottom_wall"]))

    def load_modules(self):
        """Enable collision statistics and adhesive-contact reporting."""
        module_collection = self._study.GetModuleCollection()
        bcs = module_collection.GetModule("Boundary Collision Statistics")
        bcs.EnableModule()
        bcs.SetModuleProperty("Intensities", value=True)

        contacts_data = self._study.GetContactData()
        contacts_data.EnableCollectContactsData()
        if self.settings.adhesion_model != "none":
            contacts_data.EnableIncludeAdhesiveContacts()

    def _select_processor(self, solver):
        """Select the simulation processor, writing a warning file on fallback.

        Supports ``CPU``, ``GPU``, and ``MULTI_GPU`` targets.  For ``CPU``
        the processor count is taken from ``settings.n_procs`` or
        ``$SLURM_CPUS_ON_NODE``.
        """
        p = self.settings
        target = p.processor

        if target in ("GPU", "MULTI_GPU"):
            valid = solver.GetValidSimulationTargetValues()
            if target not in valid:
                warning_path = pathlib.Path(p.project_dir) / "warnings.txt"
                with open(warning_path, "a") as f:
                    f.write(f"{target} was not available - switching to CPU\n")
                solver.SetSimulationTarget("CPU")
            else:
                solver.SetSimulationTarget(target)
                if target == "MULTI_GPU":
                    n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", p.n_procs))
                    if hasattr(solver, "SetNumberOfGPUs"):
                        solver.SetNumberOfGPUs(max(n_gpus, 1))
        elif target == "CPU":
            solver.SetSimulationTarget("CPU")
            nprocs = max(
                p.n_procs,
                int(os.environ.get("SLURM_CPUS_ON_NODE", 0)),
            )
            solver.SetNumberOfProcessors(nprocs)

    def simulate(self, insert: bool = True, adaptive_ts: bool = True) -> None:
        """Run the pre-shear simulation to completion.

        Args:
            insert: If ``True``, total duration includes the fill phase.
                Defaults to ``True``.
            adaptive_ts: If ``True``, use a variable timestep; otherwise a
                fixed 1e-6 s timestep.  Defaults to ``True``.
        """
        p = self.settings
        solver = self._study.GetSolver()
        self._select_processor(solver)

        runtime = _phase_times(p.t_fill, p.t_settle, p.t_compression, p.t_shear, insert)[-1]
        solver.SetSimulationDuration(runtime, "s")

        if not adaptive_ts:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(1e-6, "s")

        self._project.SaveProject()

        print(
            f"Starting pre-shear simulation with {solver.GetSimulationTarget()} solver...",
            flush=True,
        )
        self._study.StartSimulation(skip_summary=True)
        while self._study.IsSimulating():
            self._study.RefreshResults()
            print(f"Simulation Progress: {self._study.GetProgress():.2f} %", flush=True)
        self._project.SaveProject()
        print("Pre-shear simulation completed.", flush=True)

    #  Pre-shear post-processing
    def compute_preshear_stress(self, plot: bool = True, window_size: int = 5) -> float:
        """Compute the average shear stress from the pre-shear power curve.

        Saves ``sigma.npy`` and ``shear_stresses.npy`` to the outputs
        directory; the latter is a zero-initialised array that each shear-
        point task will fill in.

        Args:
            plot: If ``True``, save a shear-stress time-series plot.
                Defaults to ``True``.
            window_size: Savitzky-Golay window for the smoothed overlay.
                Defaults to 5.

        Returns:
            The average pre-shear shear stress (Pa).
        """
        p = self.settings
        geom = self._study.GetGeometryCollection()
        bottom_wall = geom.GetGeometry("Compression Wall 2")

        time_arr, power_lst = bottom_wall.GetNumpyCurve("Power")
        power_arr = np.array(power_lst)
        shear_arr = power_arr / (p.particle_box_len**2 * p.shear_vel)

        shear_peaks_idx = signal.find_peaks(shear_arr)[0]
        shear_peaks = shear_arr[shear_peaks_idx]

        if len(shear_peaks) >= 5:
            tau_avg = shear_peaks[-5:].mean().item()
        elif len(shear_peaks) > 0:
            tau_avg = shear_peaks.mean().item()
        else:
            tau_avg = shear_arr.mean().item()

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_arr, shear_arr, label="Shear Stress")
            ax.axhline(
                y=tau_avg, color="r", linestyle="--", label="Average Shear Stress"
            )
            if len(shear_peaks) > 0:
                ax.plot(time_arr[shear_peaks_idx], shear_peaks, "x", label="Peaks")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Shear Stress (Pa)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self._outputs_dir / "shear_stress_preshear.png", dpi=300)
            plt.close(fig)

        sigma_arr = np.linspace(p.sigma_pre, 0.0, p.n_shear_points, endpoint=False)
        tau_arr = np.zeros_like(sigma_arr)
        tau_arr[0] = tau_avg

        np.save(self._outputs_dir / "sigma.npy", sigma_arr)
        np.save(self._outputs_dir / "shear_stresses.npy", tau_arr)

        return tau_avg

    #  Shear-point case generation
    def _new_sim_settings(self, sigma: float) -> None:
        """Rewire the wall motions for a single shear-point restart case.

        Builds a release → compress → shear motion sequence on the existing
        frames, sets the simulation duration, and saves + closes the
        project so it is ready for an array task to resume.

        Args:
            sigma: Target normal stress (Pa) for this shear point.
        """
        p = self.settings
        geom = self._study.GetGeometryCollection()
        top_wall = geom.GetGeometry("Compression Wall 1")
        bottom_wall = geom.GetGeometry("Compression Wall 2")

        frame_source = self._study.GetMotionFrameSource()
        top_frame = frame_source.GetMotionFrame("Frame <01>")
        top_motions = top_frame.GetMotions()

        # brief upward release so the pack relaxes before re-compaction
        release_dist = 1e-5
        t_release = 0.5
        release_motion = top_motions.New()
        release_motion.SetType("Translation")
        release_translation = release_motion.GetTypeObject()
        release_translation.SetInput("fixed_velocity")
        release_translation.SetVelocity([0, release_dist / t_release, 0], "m/s")
        release_motion.SetStartTime(0)
        release_motion.SetStopTime(t_release)
        sim_time = t_release

        force_magnitude = _target_force(
            sigma, p.particle_box_len**2, top_wall.GetBoundaryMass()
        )
        force_motion = top_motions.New()
        force_motion.SetType("Additional Force")
        add_force = force_motion.GetTypeObject()
        add_force.SetForceValue([0, -force_magnitude, 0], "N")
        force_motion.SetStartTime(sim_time)
        force_motion.SetStopTime(sim_time + p.t_compression + p.t_shear)

        freebody_motion = top_motions.New()
        freebody_motion.SetType("Free Body Translation")
        freebody = freebody_motion.GetTypeObject()
        freebody.SetFreeMotionDirection("y")
        freebody_motion.SetStartTime(sim_time)
        top_frame.ApplyTo(top_wall)

        sim_time += p.t_compression

        bottom_frame = frame_source.GetMotionFrame("Frame <02>")
        bottom_motions = bottom_frame.GetMotions()
        shear_motion = bottom_motions.New()
        shear_motion.SetType("Translation")
        translation = shear_motion.GetTypeObject()
        translation.SetInput("fixed_velocity")
        translation.SetVelocity([0, 0, p.shear_vel], "m/s")
        shear_motion.SetStartTime(sim_time)
        shear_motion.SetStopTime(sim_time + p.t_shear)
        bottom_frame.ApplyTo(bottom_wall)

        solver = self._study.GetSolver()
        solver.SetSimulationDuration(sim_time + p.t_shear)
        self._project.SaveProject()
        self._project.CloseProject(check_save_state=False)

    def create_new_cases(self) -> list[pathlib.Path]:
        """Spawn one restart case per shear-point sigma value.

        For each sigma (excluding the pre-shear point at index 0) this saves
        a restart ``.rocky`` file into ``sigma_<kPa>kpa/``, rewires the
        motions via :meth:`_new_sim_settings`, and writes a ``case.json``
        recording the sigma value and index for the array task to consume.

        Returns:
            The list of created case directories.
        """
        p = self.settings
        sigma_arr = np.load(self._outputs_dir / "sigma.npy")
        preshear_path = _preshear_filename(p)

        case_dirs: list[pathlib.Path] = []
        for idx in range(1, len(sigma_arr)):
            sigma = float(sigma_arr[idx])
            case_dir = pathlib.Path(p.project_dir) / f"sigma_{sigma / 1000}kpa"
            case_dir.mkdir(parents=True, exist_ok=True)

            restart_filename = case_dir / f"shear_cell_{sigma / 1000}kpa.rocky"
            self._project.SaveProjectForRestart(
                filename=str(restart_filename), timestep_or_index=-1
            )
            self._new_sim_settings(sigma=sigma)

            with open(case_dir / "case.json", "w") as f:
                json.dump({"sigma": sigma, "sigma_idx": idx}, f, indent=4)

            case_dirs.append(case_dir)

            # reopen the pre-shear project for the next iteration
            self._project = self.rocky.api.OpenProject(str(preshear_path))
            self._study = self._project.GetStudy()

        self._case_dirs = case_dirs
        return case_dirs

    #  SLURM array submission
    def slurm_job(
        self,
        job_name: str = "parallel_shear",
        n_days: int = 5,
        submit: bool = True,
    ) -> pathlib.Path:
        """Write (and optionally submit) a SLURM array job for the shear points.

        Each array task changes into one ``sigma_<kPa>kpa/`` case directory
        and runs ``python -m rocky_digtools.models.shearcell.simulation
        <case_dir>``, which invokes :func:`run_shear_point`.

        Args:
            job_name: SLURM job name.  Defaults to ``"parallel_shear"``.
            n_days: Wall-time limit in days.  Defaults to 5.
            submit: If ``True``, call ``sbatch`` immediately.  Defaults to
                ``True``.

        Returns:
            The path to the written ``run_sims.sh`` script.
        """
        p = self.settings
        n_array = len(self._case_dirs)
        case_relpaths = [str(d.relative_to(p.project_dir)) for d in self._case_dirs]

        module_lines = {
            "bb-cpu": (
                "module purge; module load bluebear\n"
                "module load bear-apps/2023a\n"
                "module load ANSYS_Rocky/2024R2.0"
            ),
            "az-gpu": "ml rocky/25.2.0",
            "custom": "",
        }

        header_common = (
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --array=1-{n_array}\n"
            f"#SBATCH --time={n_days}-0\n"
        )

        if p.loc == "bb-cpu":
            header = (
                header_common + f"#SBATCH --cpus-per-task={p.n_procs}\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --qos=bbdefault\n"
                "#SBATCH --mail-type=ALL\n"
            )
        elif p.loc == "az-gpu":
            header = (
                header_common + "#SBATCH --cpus-per-task=1\n"
                "#SBATCH --gpus=1\n"
                "#SBATCH --gpus-per-task=1\n"
                "#SBATCH -p long-gpu\n"
            )
        else:  # custom
            header = header_common + f"#SBATCH --cpus-per-task={p.n_procs}\n"

        cases_block = " ".join(f'"{c}"' for c in case_relpaths)

        script = (
            "#!/bin/bash\n"
            + header
            + "\nset -e\n\n"
            + module_lines.get(p.loc, "")
            + "\n\n"
            + "cases=( "
            + cases_block
            + " )\n\n"
            + 'case_dir="${cases[$SLURM_ARRAY_TASK_ID-1]}"\n'
            + 'echo "Selected case directory: $case_dir"\n'
            + 'cd "$case_dir" || { echo "FATAL: Could not cd into $case_dir"; exit 1; }\n\n'
            + "python -m rocky_digtools.models.shearcell.simulation . >> rocky.log\n"
        )

        script_path = pathlib.Path(p.project_dir) / "run_sims.sh"
        with open(script_path, "w") as f:
            f.write(script)

        if submit:
            try:
                subprocess.run(
                    ["sbatch", str(script_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Unable to submit shear array: {e.stderr}") from e
            except FileNotFoundError:
                raise RuntimeError("sbatch not found; shear array was not submitted.")

        return script_path

    def execute_preshear(self, adaptive_ts: bool = True) -> float:
        """Run the full pre-shear phase and compute the pre-shear stress.

        Sequentially calls :meth:`load_meshes`,
        :meth:`load_material_properties`, :meth:`load_interactions`,
        :meth:`gen_particle`, :meth:`sim_physics`,
        :meth:`insertion_settings`, :meth:`move_walls`,
        :meth:`set_domain_settings`, :meth:`load_modules`,
        :meth:`simulate`, and :meth:`compute_preshear_stress`.

        Args:
            adaptive_ts: Whether to use an adaptive timestep.  Defaults to
                ``True``.

        Returns:
            The average pre-shear shear stress (Pa).
        """
        self.load_meshes(insert=self.insertion)
        self.load_material_properties()
        self.load_interactions()
        self.gen_particle()
        self.sim_physics()
        self.insertion_settings(insert=self.insertion)
        self.move_walls(insert=self.insertion)
        self.set_domain_settings()
        self.load_modules()

        with open(pathlib.Path(self.settings.project_dir) / "params.json", "w") as f:
            json.dump(asdict(self.settings), f, indent=4, default=str)

        self.simulate(insert=self.insertion, adaptive_ts=adaptive_ts)
        return self.compute_preshear_stress()

    def execute(
        self,
        adaptive_ts: bool = True,
        submit_array: bool = True,
    ) -> dict:
        """Run the pre-shear, spawn shear-point cases, and submit the array job.

        Args:
            adaptive_ts: Whether to use an adaptive timestep for the
                pre-shear.  Defaults to ``True``.
            submit_array: If ``True``, submit the SLURM array job via
                ``sbatch``.  Defaults to ``True``.

        Returns:
            A dict with the pre-shear stress, the list of case directories,
            and the path to the SLURM script.
        """
        tau_pre = self.execute_preshear(adaptive_ts=adaptive_ts)
        case_dirs = self.create_new_cases()
        script_path = self.slurm_job(submit=submit_array)

        return {
            "tau_pre": tau_pre,
            "case_dirs": case_dirs,
            "script_path": script_path,
        }


#  Array-task entry point
@pyrocky_run()
def run_shear_point(case_dir: str, rocky: Any = None) -> float:
    """Run a single shear-point restart case and record its shear stress.

    Intended to be invoked once per SLURM array task.  Opens the restart
    ``.rocky`` in ``case_dir``, (re)runs the simulation, computes the
    average shear stress from the bottom-wall power curve, and writes it
    back into ``../pyoutputs/shear_stresses.npy`` at the sigma index stored
    in ``case.json``.

    Args:
        case_dir: Directory containing the restart ``.rocky`` and
            ``case.json`` for this shear point.
        rocky: Injected Rocky API session (provided by
            :class:`~rocky_digtools.pyrocky.helpers.pyrocky_run`).

    Returns:
        The average shear stress (Pa) for this shear point.
    """
    case_dir = pathlib.Path(case_dir)
    with open(case_dir / "case.json") as f:
        case_info = json.load(f)
    sigma = float(case_info["sigma"])
    sigma_idx = int(case_info["sigma_idx"])

    project_dir = case_dir.parent
    outputs_dir = project_dir / "pyoutputs"
    with open(project_dir / "params.json") as f:
        params = json.load(f)

    rocky_glob = glob.glob(str(case_dir / "*.rocky"))
    if not rocky_glob:
        raise FileNotFoundError(f"No .rocky project found in {case_dir}")
    project = rocky.api.OpenProject(rocky_glob[0])
    study = project.GetStudy()

    solver = study.GetSolver()
    if params.get("processor") == "CPU":
        n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", params.get("n_procs", 1)))
        solver.SetNumberOfProcessors(n_cpus)

    if (not study.HasResults()) or study.CanResumeSimulation():
        study.StartSimulation(skip_summary=True)
        while study.IsSimulating():
            study.RefreshResults()
            print(f"Simulation Progress: {study.GetProgress():.2f} %", flush=True)
        project.SaveProject()

    box_len = params["particle_box_len"]
    shear_vel = params["shear_vel"]

    geom = study.GetGeometryCollection()
    bottom_wall = geom.GetGeometry("Compression Wall 2")
    time_arr, power_lst = bottom_wall.GetNumpyCurve("Power")
    power_arr = np.array(power_lst)
    shear_arr = power_arr / (box_len**2 * shear_vel)

    shear_peaks_idx = signal.find_peaks(shear_arr)[0]
    shear_peaks = shear_arr[shear_peaks_idx]
    if len(shear_peaks) >= 5:
        tau_avg = shear_peaks[-5:].mean().item()
    elif len(shear_peaks) > 0:
        tau_avg = shear_peaks.mean().item()
    else:
        tau_avg = shear_arr.mean().item()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_arr, shear_arr, label="Shear Stress")
    ax.axhline(y=tau_avg, color="r", linestyle="--", label="Average Shear Stress")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Shear Stress (Pa)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs_dir / f"shear_stress_{sigma / 1000}kpa.png", dpi=300)
    plt.close(fig)

    result_path = case_dir / "result.json"
    temp_path = result_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps({"sigma": sigma, "sigma_idx": sigma_idx, "tau": tau_avg}))
    temp_path.replace(result_path)

    return tau_avg


# ====================================================================== #
#  Final aggregation (Mohr-circle / yield-locus fitting)
# ====================================================================== #
def _regression_line(sigma: np.ndarray, tau: np.ndarray):
    """Fit a regression-line yield locus and return Mohr-circle parameters."""
    sigma_locus = sigma[1:]
    tau_locus = tau[1:]
    if len(sigma_locus) < 2 or len(np.unique(sigma_locus)) != len(sigma_locus):
        raise ValueError("At least two distinct post-preshear stresses are required.")
    m, c = np.polyfit(sigma_locus, tau_locus, 1, full=False)

    if c <= 0:
        raise ValueError("y-intercept must be positive for the Mohr circle.")

    r_unc = c / ((m**2 + 1) ** 0.5 - m)

    x1, y1 = sigma[0], tau[0]
    a_quad = 1
    b_quad = -2 * (m**2 * x1 + x1 + m * c)
    c_quad = (m**2 + 1) * (x1**2 + y1**2) - c**2
    discriminant = b_quad**2 - 4 * a_quad * c_quad
    discriminant = 0.0 if np.isclose(discriminant, 0) else discriminant

    if discriminant >= 0 or np.isclose(discriminant, 0):
        centre_conf = (-b_quad - np.sqrt(discriminant)) / (2 * a_quad)
        radius_conf = np.sqrt((x1 - centre_conf) ** 2 + y1**2)
        return m, c, r_unc, centre_conf, radius_conf
    return None


def _straight_sections(sigma: np.ndarray, tau: np.ndarray):
    """Fit a piecewise-linear yield locus and return Mohr-circle parameters."""
    sigma_locus = sigma[1:]
    tau_locus = tau[1:]
    x1, y1 = sigma[0], tau[0]

    m_piece = np.diff(tau_locus) / np.diff(sigma_locus)
    c_piece = tau_locus[:-1] - m_piece * sigma_locus[:-1]

    if c_piece[-1] <= 0:
        raise ValueError("y-intercept must be positive for the Mohr circle.")

    r_unc = c_piece[-1] / ((m_piece[-1] ** 2 + 1) ** 0.5 - m_piece[-1])

    a_quad = 1
    b_quad = -2 * (m_piece[0] ** 2 * x1 + x1 + m_piece[0] * c_piece[0])
    c_quad = (m_piece[0] ** 2 + 1) * (x1**2 + y1**2) - c_piece[0] ** 2
    discriminant = b_quad**2 - 4 * a_quad * c_quad
    discriminant = 0.0 if np.isclose(discriminant, 0) else discriminant

    if discriminant >= 0:
        centre_conf = (-b_quad - np.sqrt(discriminant)) / (2 * a_quad)
        radius_conf = np.sqrt((x1 - centre_conf) ** 2 + y1**2)
        return m_piece, c_piece, r_unc, centre_conf, radius_conf
    return None


def _force_fit(sigma: np.ndarray, tau: np.ndarray):
    """Constrained least-squares fit of the yield locus through the pre-shear."""
    sigma_constr, tau_constr = sigma[0], tau[0]
    sigma_locus = sigma[1:]
    tau_locus = tau[1:]

    def linfit(x, m, c):
        return m * x + c

    def f_obj(params):
        m, c = params
        return np.sum((linfit(sigma_locus, m, c) - tau_locus) ** 2)

    def preshear_constr(params):
        m, c = params
        return linfit(sigma_constr, m, c) - tau_constr

    constr = {"type": "eq", "fun": preshear_constr}
    result = optimize.minimize(f_obj, [1.0, 10.0], method="SLSQP", constraints=constr)
    if not result.success:
        raise RuntimeError(f"Yield-locus fit failed: {result.message}")
    m_fit, c_fit = result.x

    r_unc = c_fit / ((m_fit**2 + 1) ** 0.5 - m_fit)

    a_quad = 1.0
    b_quad = -2 * (m_fit**2 * sigma_constr + sigma_constr + m_fit * c_fit)
    c_quad = (m_fit**2 + 1) * (sigma_constr**2 + tau_constr**2) - c_fit**2
    discriminant = b_quad**2 - 4 * a_quad * c_quad
    discriminant = 0.0 if np.isclose(discriminant, 0) else discriminant

    if discriminant >= 0:
        centre_conf = (-b_quad - np.sqrt(discriminant)) / (2 * a_quad)
        radius_conf = np.sqrt((sigma_constr - centre_conf) ** 2 + tau_constr**2)
        return m_fit, c_fit, r_unc, centre_conf, radius_conf
    return None


def aggregate_results(
    project_dir: str | pathlib.Path,
    plot: bool = True,
    db_path: str | pathlib.Path | None = None,
) -> dict:
    """Fit Mohr circles to the collected shear stresses and dump metrics.

    Loads ``pyoutputs/sigma.npy`` and ``pyoutputs/shear_stresses.npy`` from
    ``project_dir``; if any shear point is still zero the function returns
    ``None`` (the array job is not done yet).  Otherwise it fits the yield
    locus (regression line → straight sections → constrained force-fit),
    plots the Mohr circles, and appends a row to a SQLite database.

    Args:
        project_dir: Directory containing the ``pyoutputs/`` folder and
            ``params.json``.
        plot: If ``True``, save the Mohr-circle figure.  Defaults to ``True``.
        db_path: SQLite database path.  Defaults to
            ``<project_dir>/../results.db``.

    Returns:
        A dict of computed flow metrics (``ffc``, ``sigma_unc``,
        ``sigma_conf``, ``phi_i``, ``phi_eff``, ``fit_method``), or ``None``
        if not all shear points have completed.
    """
    project_dir = pathlib.Path(project_dir)
    outputs_dir = project_dir / "pyoutputs"
    sigma = np.load(outputs_dir / "sigma.npy")
    tau = np.load(outputs_dir / "shear_stresses.npy")
    for case_dir in project_dir.glob("sigma_*kpa"):
        result_path = case_dir / "result.json"
        if not result_path.exists():
            return None
        result = json.loads(result_path.read_text())
        idx = result.get("sigma_idx")
        if not isinstance(idx, int) or not 0 < idx < len(tau) or result.get("sigma") != sigma[idx]:
            raise ValueError(f"Invalid shear result provenance: {result_path}")
        tau[idx] = result.get("tau", np.nan)

    if len(sigma) < 3 or len(tau) != len(sigma):
        raise ValueError("At least three matching shear stress points are required.")
    if not np.isfinite(sigma).all() or not np.isfinite(tau).all():
        return None

    if (result := _regression_line(sigma, tau)) is not None:
        method = "regression_line"
    elif (result := _straight_sections(sigma, tau)) is not None:
        method = "straight_sections"
    elif (result := _force_fit(sigma, tau)) is not None:
        method = "force_fit"
    else:
        raise RuntimeError("No valid method found for Mohr circle fitting.")

    m, c, r_unc, centre_conf, radius_conf = result
    sigma_c = centre_conf + radius_conf
    sigma_u = r_unc
    if sigma_c <= 0 or sigma_u <= 0 or centre_conf == 0:
        raise ValueError("Nonphysical Mohr-circle radii.")
    ffc = sigma_c / sigma_u
    phi_i_arg = (r_unc - radius_conf) / (r_unc - centre_conf)
    phi_eff_arg = radius_conf / centre_conf
    if not (-1 <= phi_i_arg <= 1 and -1 <= phi_eff_arg <= 1):
        raise ValueError("Nonphysical Mohr-circle angle.")
    phi_i = np.rad2deg(np.arcsin(phi_i_arg))
    phi_eff = np.rad2deg(np.arcsin(phi_eff_arg))

    if plot:
        sigma_fit = np.linspace(0, sigma.max(), 100)
        m_eff = np.tan(np.deg2rad(phi_eff))
        tau_eff = m_eff * sigma_fit

        fig, ax_dict = plt.subplot_mosaic(
            [["A", "A"], ["B", "C"]], layout="constrained"
        )

        unc = plt.Circle((r_unc, 0), r_unc, color="black", fill=False)
        conf = plt.Circle((centre_conf, 0), radius_conf, color="black", fill=False)
        ax_dict["A"].add_artist(unc)
        ax_dict["A"].add_artist(conf)
        ax_dict["A"].plot(
            sigma_fit, tau_eff, color="black", linestyle=":", label="Effective Locus"
        )

        if method == "regression_line" or method == "force_fit":
            ax_dict["A"].scatter(
                sigma[1:], tau[1:], label="Shear points", color="black", marker="o"
            )
            ax_dict["A"].scatter(
                sigma[0], tau[0], label="Pre-shear", color="black", marker="s"
            )
            ax_dict["A"].plot(
                sigma_fit,
                m * sigma_fit + c,
                label="Yield Locus",
                linestyle="--",
                color="black",
            )
        else:  # straight_sections
            ax_dict["A"].plot(
                sigma[1:], tau[1:], "o-", label="Shear Points", color="black"
            )
            ax_dict["A"].scatter(
                sigma[0], tau[0], color="black", marker="s", label="Pre-shear"
            )

        ax_dict["A"].set_aspect("equal", adjustable="box")
        ax_dict["A"].set_xlabel(r"$\sigma$ (Pa)")
        ax_dict["A"].set_ylabel(r"$\tau$ (Pa)")
        ax_dict["A"].set_xlim(0, sigma_c * 1.05)
        ax_dict["A"].set_ylim(0, tau.max() * 1.05)

        row_names = [
            r"$\sigma_{unc}$",
            r"$\sigma_{conf}$",
            r"FFC",
            r"$\phi_{i}$",
            r"$\phi_{eff}$",
            "Fit Method",
        ]
        table_vals = [
            [sigma_u.round(2)],
            [sigma_c.round(2)],
            [ffc.round(2)],
            [phi_i.round(2)],
            [phi_eff.round(2)],
            [method],
        ]
        ax_dict["B"].axis("off")
        tab = ax_dict["B"].table(cellText=table_vals, rowLabels=row_names, loc="top")
        tab.auto_set_column_width(col=0)

        handles, labels = ax_dict["A"].get_legend_handles_labels()
        ax_dict["C"].axis("off")
        ax_dict["C"].legend(handles, labels, loc="upper center", frameon=False)
        fig.tight_layout()
        fig.savefig(outputs_dir / "mohr_circles_plots.png", dpi=300)
        plt.close(fig)

    metrics = {
        "ffc": float(ffc),
        "sigma_unc": float(sigma_u),
        "sigma_conf": float(sigma_c),
        "phi_i": float(phi_i),
        "phi_eff": float(phi_eff),
        "fit_method": method,
    }

    if db_path is None:
        db_path = project_dir.parent / "results.db"
    with open(project_dir / "params.json") as f:
        params = json.load(f)
    df = pd.json_normalize(params)
    df.columns = [col.split(".")[-1] if "." in col else col for col in df.columns]
    for k, v in metrics.items():
        df[k] = v
    with sqlite3.connect(str(db_path)) as conn:
        df.to_sql("parallel_shear", conn, if_exists="append", index=False)

    return metrics


if __name__ == "__main__":
    import sys

    _case_dir = sys.argv[1]
    run_shear_point(_case_dir)

"""Model-specific packages for rocky_digtools.

Each submodule implements a particular DEM test/experiment (e.g. uniaxial
compression, shear cell) on top of the general case-setup and Rocky API
utilities provided by the top-level package, and on the shared DOE
sweep/OFAT engine in :mod:`rocky_digtools.models.doe`.
"""

import abc as _abc
import os
import pathlib as _pathlib
import subprocess
from typing import Any

import numpy as _np

from .. import particles_shapes as _particles_shapes
from ..pyrocky.helpers import pyrocky_run as _pyrocky_run

__all__ = ["doe", "uniax", "shearcell"]


@_pyrocky_run()
class PyrockySimulation(_abc.ABC):
    """Abstract base class for DEM simulations.

    This class defines the interface and common functionality for all DEM
    simulations in the `rocky_digtools.models` package. Subclasses must
    implement the abstract methods to provide specific simulation behavior.
    """

    rocky: Any  # Type hint for the Rocky API proxy, to be set externally

    def __init__(
        self,
        filename: str,
        settings: Any,
        study_name: str,
    ) -> None:

        self.settings = settings
        self.filename = filename
        self._study_name = study_name
        self.setup()

        self._particle = None
        self._mesh = {}
        self._materials = {}

    @staticmethod
    def _ser(proxy) -> dict:
        """Serialise a Rocky API proxy for cross-call argument passing.

        Args:
            proxy: A Rocky API proxy object.

        Returns:
            Serialised representation suitable for passing to another API call.
        """
        """Serialise a Rocky API proxy for passing as an argument to another API call."""
        return proxy.serialize(proxy)

    def setup(self):
        """Create and save a new Rocky project and study."""

        if hasattr(self.rocky, "api") and self.rocky.api is not None:
            self._project = self.rocky.api.CreateProject()
        else:
            raise RuntimeError(
                "Rocky API is not available. Ensure that the Rocky API is properly initialized."
            )

        self._project.SaveProject(
            str(_pathlib.Path(self.settings.project_dir) / self.filename)
        )
        self._study = self._project.GetStudy()
        self._study.SetName(self._study_name)

    @_abc.abstractmethod
    def load_meshes(self):
        """Load meshes into the Rocky project.

        This method must be implemented by subclasses to load the specific
        meshes required for the simulation.
        """
        pass

    def load_interactions(self):
        """Configure particle–particle and particle–wall interaction parameters."""
        pm = self._materials["particle_mat"]
        wm = self._materials["wall_mat"]

        interaction_collection = self._study.GetMaterialsInteractionCollection()
        pp_interaction = interaction_collection.GetMaterialsInteraction(
            self._ser(pm), self._ser(pm)
        )
        pw_interaction = interaction_collection.GetMaterialsInteraction(
            self._ser(pm), self._ser(wm)
        )

        pp_interaction.SetRestitutionCoefficient(self.settings.cor_pp)
        pp_interaction.SetDynamicFriction(self.settings.fric_dyn_pp)
        pp_interaction.SetStaticFriction(self.settings.fric_stat_pp)

        pw_interaction.SetRestitutionCoefficient(self.settings.cor_pw)
        pw_interaction.SetDynamicFriction(self.settings.fric_dyn_pw)
        pw_interaction.SetStaticFriction(self.settings.fric_stat_pw)

        if self.settings.adhesion_model == "JKR":
            pp_interaction.SetSurfaceEnergy(self.settings.surf_en_pp, "J/m2")
            pw_interaction.SetSurfaceEnergy(self.settings.surf_en_pw, "J/m2")

    def gen_particle(self):
        """Create a particle in the study and configure its shape and size.

        Raises:
            ValueError: If the shape type is unsupported or required files
                are missing.
        """
        self._particle = self._study.CreateParticle()
        self._particle.SetName("Particle")

        match self.settings.shape_name:
            case "sphere":
                shape = _particles_shapes.Sphere(radius=self.settings.p_radius)
            case "polyhedron":
                shape = _particles_shapes.Polyhedron(
                    radius=self.settings.p_radius,
                    vert_ar=self.settings.vert_ar,
                    horiz_ar=self.settings.horiz_ar,
                    n_corners=self.settings.n_corners,
                    superquadric_degree=self.settings.sq_degree,
                )
            case "sphero_cylinder":
                shape = _particles_shapes.SpheroCylinder(
                    radius=self.settings.p_radius, vert_ar=self.settings.vert_ar
                )
            case "custom_polyhedron":
                if (
                    not self.settings.particle_path
                    or not _pathlib.Path(self.settings.particle_path).is_file()
                ):
                    raise ValueError(
                        "Particle path must be provided for custom polyhedron shape."
                    )
                shape = _particles_shapes.CustomPolyhedron(
                    stl_path=self.settings.particle_path, radius=self.settings.p_radius
                )
            case _:
                raise ValueError(
                    f"Unsupported shape type: {self.settings.shape_name}. "
                    "Supported shapes are: 'sphere', 'polyhedron', 'sphero_cylinder', and 'custom_polyhedron'."
                )
        pm = self._materials["particle_mat"]
        shape.particle2rocky(
            particle=self._particle,
            material=self._ser(pm),
            rolling_friction=self.settings.rolling_fric,
        )

    def sim_physics(self):
        """Set the contact force models and gravity direction."""
        physics = self._study.GetPhysics()
        physics.SetNormalForceModel(self.settings.normal_force_model)
        physics.SetTangentialForceModel(self.settings.tangential_force_model)
        physics.SetAdhesionModel(self.settings.adhesion_model)

        physics.SetGravityXDirection(0)
        physics.SetGravityYDirection(-9.81)
        physics.SetGravityZDirection(0)

    def insertion_settings(self) -> None:
        """Configure the particle inlet for the fill phase."""

        fill_box_vol = self.settings.particle_box_len**3
        particle_vol = self.settings.expected_particle_volume
        n_particles = int(
            _np.rint(fill_box_vol / particle_vol * 0.5)
        )  # target 50% fill
        mass_particles = particle_vol * self.settings.p_density * n_particles

        inlet = self._mesh["insert_inlet"]
        particle_inlet = self._study.CreateParticleInlet(
            self._ser(inlet),
            self._ser(self._particle),
        )
        flowr = mass_particles / self.settings.t_fill

        input_property_lst = particle_inlet.GetInputPropertiesList()
        input_property_lst[0].SetMassFlowRate(flowr, "kg/s")

        particle_inlet.SetStartTime(0.0, "s")
        particle_inlet.SetStopTime(self.settings.t_fill, "s")
        particle_inlet.DisablePeriodic()

    def load_material_properties(self):
        """Create particle and wall materials and assign them to the geometry."""
        material_collection = self._study.GetMaterialCollection()

        particle_mat = material_collection.AddSolidMaterial()
        particle_mat.SetName("Particle Material")
        particle_mat.SetDensity(self.settings.p_density)
        particle_mat.SetYoungsModulus(self.settings.p_youngmod)
        particle_mat.SetPoissonRatio(self.settings.p_poisson)
        particle_mat.SetUseBulkDensity(False)

        wall_mat = material_collection.AddSolidMaterial()
        wall_mat.SetName("Wall Material")
        wall_mat.SetDensity(2700)
        wall_mat.SetYoungsModulus(1e9)
        wall_mat.SetPoissonRatio(0.3)
        wall_mat.SetUseBulkDensity(False)

        self._mesh["top_wall"].SetMaterial(self._ser(wall_mat))
        self._mesh["bottom_wall"].SetMaterial(self._ser(wall_mat))
        self._materials["particle_mat"] = particle_mat
        self._materials["wall_mat"] = wall_mat

    @_abc.abstractmethod
    def set_domain_settings(self):
        """Configure the simulation domain bounds and periodic boundaries."""
        pass

    def _check_nvidia_gpu(self) -> int:
        """Count available NVIDIA GPUs on the system.

        Returns:
            Number of NVIDIA GPUs detected, or 0 on failure.
        """
        try:
            output = subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8")
            return sum(1 for line in output.strip().splitlines() if line)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0

    def _select_processor(self, solver):
        """Select the simulation processor (CPU/GPU) on the solver.

        Falls back to CPU if the requested GPU is unavailable.

        Args:
            solver: The Rocky solver object.
        """
        if self.settings.processor == "GPU":
            if not (n_gpus := self._check_nvidia_gpu()):
                print("Warning: No NVIDIA GPU detected. Falling back to CPU.")
                solver.SetSimulationTarget("CPU")
            else:
                if n_gpus >= 1:
                    solver.SetSimulationTarget("GPU")
                # TODO: Add support for multi-GPU setups

        elif self.settings.processor == "CPU":
            solver.SetSimulationTarget("CPU")

            cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
            solver.SetNumberOfProcessors(cpus)

    @_abc.abstractmethod
    def load_modules(self):
        """Enable contacts data collection and adhesive contact reporting."""
        pass

    def simulate(self, sim_time: float, adaptive_ts: bool = True) -> None:
        """Run the simulation to completion.

        Args:
            sim_time: The total simulation time.
            adaptive_ts: If ``True``, use adaptive time stepping. Defaults to ``True``.
        """
        solver = self._study.GetSolver()
        self._select_processor(solver)

        solver.SetTimestepModel("variable" if adaptive_ts else "constant")
        solver.SetSimulationDuration(sim_time, "s")

        self._project.SaveProject()

        print(f"Starting simulation with {solver.GetSimulationTarget()} solver...")
        self._study.StartSimulation(non_blocking=True)

        while self._study.IsSimulating():
            self._study.RefreshResults()
            print(f"Simulation Progress: {self._study.GetProgress():.2f} %")

    @_abc.abstractmethod
    def post_process(self):
        """Post-process simulation results and generate plots."""
        pass


# Avoiding circular import
from . import doe, shearcell, uniax  # noqa: E402

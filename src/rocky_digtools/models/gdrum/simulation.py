"""Pyrocky rotating drum: fill, settle, then rotate about the horizontal Z axis."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ...particles_shapes import normalise_radius
from .. import PyrockySimulation
from ..doe.schema import ShapeConfig, field_values
from .drum_meshgen import create_meshes, drum_dimensions, geometry_name


def validate_rpm(rpm):
    if (
        isinstance(rpm, bool)
        or not isinstance(rpm, (int, float))
        or not math.isfinite(rpm)
        or rpm <= 0
    ):
        raise ValueError("rpm must be finite and > 0")


@dataclass
class Settings:
    """Specify two of radius, length and L/D, or particle_box_len plus one.

    Lengths are in metres and times in seconds.
    ``fill_fraction`` is bulk volume / drum volume. ``packing_fraction``
    estimates solid volume / bulk volume when converting that target to mass.
    """

    project_dir: str | Path
    rpm: float
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
    particle_box_len: float | None = None
    aspect_ratio: float | None = None
    length: float | None = None
    radius: float | None = None
    tang_stiff_ratio_pp: float | None = None
    tang_stiff_ratio_pw: float | None = None
    surf_en_pp: float = 0.0
    surf_en_pw: float = 0.0
    normal_force_model: str = "linear_hysteresis"
    tangential_force_model: str = "coulomb_limit"
    adhesion_model: str = "none"
    rolling_fric: float = 0.0
    rolling_model: str = "none"
    neighbor_search: str = "BVH"
    processor: str = "GPU"
    mesh_dir: str | Path | None = None
    plots_dir: str | Path | None = None
    shape_name: str = "sphere"
    vert_ar: float = 1.0
    horiz_ar: float = 1.0
    n_corners: int = 30
    sq_degree: float = 2.0
    particle_path: str | None = None
    smoothness: float | None = None
    t_fill: float = 1.0
    t_settle: float = 0.5
    n_revolutions: float = 5.0
    fill_fraction: float = 0.5
    packing_fraction: float = 0.6

    def __post_init__(self):
        self.p_radius = normalise_radius(self.p_radius)
        self._validate()
        self.project_dir = Path(self.project_dir)
        self.mesh_dir = (
            Path(self.mesh_dir)
            if self.mesh_dir is not None
            else (self.project_dir.parent / f"meshes_{self.equivalent_box_len}")
        )
        self.plots_dir = (
            Path(self.plots_dir) if self.plots_dir else self.project_dir / "plots"
        )
        if not all(
            (self.geometry_dir / name).is_file()
            for name in ("drum_shell.stl", "insert.stl")
        ):
            geometry = {
                key: value
                for key, value in (
                    ("length", self.length),
                    ("aspect_ratio", self.aspect_ratio),
                    ("radius", self.radius),
                )
                if value is not None
            }
            create_meshes(
                self.particle_box_len,
                out_dir=self.mesh_dir,
                **geometry,
            )
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def _validate(self):
        drum_dimensions(
            self.particle_box_len,
            self.aspect_ratio,
            length=self.length,
            radius=self.radius,
        )
        validate_rpm(self.rpm)
        for name in (
            "p_density",
            "p_youngmod",
            "t_fill",
            "n_revolutions",
            "vert_ar",
            "horiz_ar",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in (
            "t_settle",
            "fric_dyn_pp",
            "fric_stat_pp",
            "fric_dyn_pw",
            "fric_stat_pw",
            "surf_en_pp",
            "surf_en_pw",
            "rolling_fric",
            "tang_stiff_ratio_pp",
            "tang_stiff_ratio_pw",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and >= 0")
        for name, low, high in (
            ("p_poisson", 0, 0.5),
            ("cor_pp", 0, 1),
            ("cor_pw", 0, 1),
            ("fill_fraction", 0, 1),
            ("packing_fraction", 0, 1),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if self.fill_fraction == 0 or self.packing_fraction == 0:
            raise ValueError("fill_fraction and packing_fraction must be > 0")
        for name, allowed in {
            "normal_force_model": ("linear_hysteresis", "hertz", "linear_spring"),
            "tangential_force_model": ("coulomb_limit", "linear_spring_coulomb_limit"),
            "adhesion_model": ("none", "constant", "linear", "JKR"),
            "rolling_model": ("none", "type_a", "type_b"),
            "neighbor_search": ("BVH", "RegularGrid", "SparseGrid"),
            "processor": ("CPU", "GPU"),
            "shape_name": (
                "sphere",
                "polyhedron",
                "sphero_cylinder",
                "custom_polyhedron",
            ),
        }.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.adhesion_model == "JKR" and min(self.surf_en_pp, self.surf_en_pw) <= 0:
            raise ValueError("JKR requires positive surf_en_pp and surf_en_pw")
        if self.shape_name == "custom_polyhedron" and (
            not self.particle_path or not Path(self.particle_path).is_file()
        ):
            raise ValueError("custom_polyhedron requires a valid particle_path")
        if (
            not math.isfinite(self.n_corners)
            or self.n_corners < 10
            or int(self.n_corners) != self.n_corners
        ):
            raise ValueError("n_corners must be an integer >= 10")
        if not math.isfinite(self.sq_degree) or self.sq_degree < 2:
            raise ValueError("sq_degree must be >= 2")
        largest_radius = (
            max(self.p_radius) if isinstance(self.p_radius, dict) else self.p_radius
        )
        if 2 * largest_radius >= min(self.drum_radius, 0.9 * self.drum_length):
            raise ValueError("Particle diameter must fit through the drum inlet")
        if 2 * largest_radius >= 0.4 * self.drum_length:
            raise ValueError(
                "Particle diameter must be < 0.4 times the axial periodic length"
            )
        if self.n_particles < 1:
            raise ValueError("Drum charge must contain at least one particle")

    @property
    def drum_radius(self):
        return drum_dimensions(
            self.particle_box_len,
            self.aspect_ratio,
            length=self.length,
            radius=self.radius,
        )[0]

    @property
    def drum_length(self):
        return drum_dimensions(
            self.particle_box_len,
            self.aspect_ratio,
            length=self.length,
            radius=self.radius,
        )[1]

    @property
    def drum_volume(self):
        if self.particle_box_len is not None:
            return self.particle_box_len**3
        return math.pi * self.drum_radius**2 * self.drum_length

    @property
    def equivalent_box_len(self):
        return (
            self.particle_box_len
            if self.particle_box_len is not None
            else self.drum_volume ** (1 / 3)
        )

    @property
    def geometry_dir(self):
        return self.mesh_dir / geometry_name(
            self.particle_box_len,
            self.aspect_ratio,
            length=self.length,
            radius=self.radius,
        )

    @property
    def angular_velocity(self):
        return self.rpm * 2 * math.pi / 60

    @property
    def t_rotate(self):
        return self.n_revolutions * 60 / self.rpm

    @property
    def expected_particle_volume(self):
        distribution = (
            self.p_radius if isinstance(self.p_radius, dict) else {self.p_radius: 1.0}
        )
        return sum(
            4 / 3 * math.pi * radius**3 * weight
            for radius, weight in distribution.items()
        ) / sum(distribution.values())

    @property
    def n_particles(self):
        return round(
            self.drum_volume
            * self.fill_fraction
            * self.packing_fraction
            / self.expected_particle_volume
        )

    @property
    def charge_mass(self):
        return self.n_particles * self.expected_particle_volume * self.p_density

    @classmethod
    def from_dict(cls, data):
        """Read the flat settings.json written by the shared DOE runtime."""
        return cls(**data)

    @classmethod
    def from_json(cls, path, project_dir):
        """Read nested scalar JSON using the same common fields as uniax."""
        from .doe import _schema_for_settings

        data = json.loads(Path(path).read_text())
        values = field_values(data, _schema_for_settings(data["experiment_settings"]))
        shape = ShapeConfig.from_dict({"n_corners": 30, **data["shape"]})
        mapping = {
            "radius": "p_radius",
            "density": "p_density",
            "youngmod": "p_youngmod",
            "poisson": "p_poisson",
            "fric_rolling": "rolling_fric",
            "box_len": "particle_box_len",
            "drum_radius": "radius",
            "normal": "normal_force_model",
            "tangential": "tangential_force_model",
            "rolling": "rolling_model",
            "adhesion": "adhesion_model",
        }
        settings = {mapping.get(key, key): value for key, value in values.items()}
        settings.update(
            shape_name=shape.name,
            vert_ar=shape.vert_ar,
            horiz_ar=shape.horiz_ar,
            n_corners=shape.n_corners,
            sq_degree=shape.sq_degree,
            particle_path=shape.particle_path,
            smoothness=shape.smoothness,
        )
        for key in (
            "t_fill",
            "t_settle",
            "n_revolutions",
            "fill_fraction",
            "packing_fraction",
        ):
            if key in data["experiment_settings"]:
                settings[key] = data["experiment_settings"][key]
        return cls(project_dir=project_dir, **settings)


class GranularDrumSimulation(PyrockySimulation):
    """A rotating cylindrical shell with periodic boundaries at its axial ends."""

    def __init__(self, settings, filename="granular_drum.rocky"):
        super().__init__(filename, settings, "Rotating Granular Drum")

    def load_meshes(self):
        directory = self.settings.geometry_dir
        walls = self._study.ImportWall(
            str(directory / "drum_shell.stl"), import_scale=1.0, convert_yz=False
        )
        for index, wall in enumerate(walls):
            wall.SetName(f"Drum {index}")
            self._mesh[f"drum_{index}"] = wall
        inlet = self._study.ImportSurface(
            str(directory / "insert.stl"), import_scale=1.0, convert_yz=False
        )[0]
        inlet.SetName("Insert Inlet")
        self._mesh["insert_inlet"] = inlet

    def insertion_settings(self):
        inlet = self._study.CreateParticleInlet(
            self._ser(self._mesh["insert_inlet"]), self._ser(self._particle)
        )
        inlet.GetInputPropertiesList()[0].SetMassFlowRate(
            self.settings.charge_mass / self.settings.t_fill, "kg/s"
        )
        inlet.SetStartTime(0.0, "s")
        inlet.SetStopTime(self.settings.t_fill, "s")
        inlet.DisablePeriodic()

    def rotate_drum(self):
        frame = self._study.GetMotionFrameSource().NewFrame()
        frame.SetRelativePosition([0, 0, 0], "m")
        motion = frame.GetMotions().New()
        motion.SetType("Rotation")
        motion.GetTypeObject().SetInitialAngularVelocity(
            [0, 0, self.settings.angular_velocity], "rad/s"
        )
        start = self.settings.t_fill + self.settings.t_settle
        motion.SetStartTime(start, "s")
        motion.SetStopTime(start + self.settings.t_rotate, "s")
        for key, wall in self._mesh.items():
            if key != "insert_inlet":
                frame.ApplyTo(self._ser(wall))

    def set_domain_settings(self):
        domain = self._study.GetDomainSettings()
        domain.DisableUseBoundaryLimits()
        domain.DisablePeriodicAtGeometryLimits()
        domain.SetDomainType("CARTESIAN")
        domain.SetCartesianPeriodicDirections("Z")
        radius, length = self.settings.drum_radius, self.settings.drum_length
        domain.SetPeriodicLimitsMinCoordinates(
            [-1.1 * radius, -1.1 * radius, -length / 2]
        )
        domain.SetPeriodicLimitsMaxCoordinates([1.1 * radius, 1.1 * radius, length / 2])
        domain.SetCoordinateLimitsMinValues(
            [-1.1 * radius, -1.1 * radius, -0.55 * length]
        )
        domain.SetCoordinateLimitsMaxValues([1.1 * radius, 1.1 * radius, 0.55 * length])

    def load_modules(self):
        contacts = self._study.GetContactData()
        contacts.EnableCollectContactsData()
        if self.settings.adhesion_model != "none":
            contacts.EnableIncludeAdhesiveContacts()

    def post_process(self):
        """Save geometry, charge and final particle count alongside the Rocky project."""
        result = {
            "drum_radius": self.settings.drum_radius,
            "drum_length": self.settings.drum_length,
            "drum_volume": self.settings.drum_volume,
            "aspect_ratio": self.settings.drum_length / (2 * self.settings.drum_radius),
            "rpm": self.settings.rpm,
            "t_rotate": self.settings.t_rotate,
            "fill_fraction": self.settings.fill_fraction,
            "packing_fraction": self.settings.packing_fraction,
            "target_particle_count": self.settings.n_particles,
            "target_charge_mass": self.settings.charge_mass,
            "final_particle_count": int(
                self._study.GetParticles().GetNumberOfParticles(-1)
            ),
        }
        (self.settings.project_dir / "results.json").write_text(
            json.dumps(result, indent=2)
        )
        return result

    def execute(self):
        self.load_meshes()
        self.load_material_properties()
        self.load_interactions()
        self.gen_particle()
        self.sim_physics()
        self.insertion_settings()
        self.rotate_drum()
        self.set_domain_settings()
        self.load_modules()
        self.simulate(
            self.settings.t_fill + self.settings.t_settle + self.settings.t_rotate
        )
        return self.post_process()

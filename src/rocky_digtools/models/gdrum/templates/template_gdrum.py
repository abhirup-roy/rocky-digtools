"""Standalone Rocky PrePost drum script, rendered by Jinja2."""

import importlib.util
import json
import math
import os
from pathlib import Path
import sys

# Import the particle shapes module
spec = importlib.util.spec_from_file_location(
    "particles_shapes", {{SHAPES_MODULE_PATH | tojson}}
)
particle_shapes = importlib.util.module_from_spec(spec)
sys.modules["particles_shapes"] = particle_shapes
spec.loader.exec_module(particle_shapes)

box_len = {{L_BOX}}
aspect_ratio = {{ASPECT_RATIO}}
rpm = {{RPM}}
if not all(
    math.isfinite(value) and value > 0 for value in (box_len, aspect_ratio, rpm)
):
    raise ValueError("box_len, aspect_ratio and rpm must be finite and > 0")

# drum dimensions
radius = box_len / (2 * math.pi * aspect_ratio) ** (1 / 3)
length = 2 * radius * aspect_ratio

t_fill = 1.0
t_settle = 0.5
n_revolutions = 5.0
fill_fraction = 0.5
packing_fraction = 0.6

t_rotate = n_revolutions * 60 / rpm
duration = t_fill + t_settle + t_rotate

mesh_dir = Path.cwd().parent / "meshes_{{ L_BOX }}" / f"aspect_{float(aspect_ratio)}"
p_radius = particle_shapes.normalise_radius({{RADIUS_P}})
p_density = {{DENSITY_P}}
distribution = p_radius if isinstance(p_radius, dict) else {p_radius: 1.0}

particle_volume = sum(
    4 / 3 * math.pi * size**3 * weight for size, weight in distribution.items()
) / sum(distribution.values())

n_particles = round(box_len**3 * fill_fraction * packing_fraction / particle_volume)
charge_mass = n_particles * particle_volume * p_density
if n_particles < 1 or 2 * max(distribution) >= min(radius, 0.9 * length):
    raise ValueError(
        "Particles must fit through the inlet and the charge must contain at least one particle"
    )


def setup(filename="granular_drum.rocky") -> None:
    """Create the Rocky project and study."""
    global project, study

    project = app.CreateProject()
    project.SaveProject(str(Path.cwd() / filename))
    study = project.GetStudy()
    study.SetName("Rotating Granular Drum")


def load_meshes() -> None:
    """Import the closed drum walls and particle inlet surface."""
    global walls, inlet_surface

    walls = study.ImportWall(
        str(mesh_dir / "drum.stl"), import_scale=1.0, convert_yz=False
    )
    for index, wall in enumerate(walls):
        wall.SetName(f"Drum {index}")
    inlet_surface = study.ImportSurface(
        str(mesh_dir / "insert.stl"), import_scale=1.0, convert_yz=False
    )[0]
    inlet_surface.SetName("Insert Inlet")


def load_material_properties() -> None:
    """Create particle and wall materials and assign the drum material."""
    global particle_material, wall_material

    materials = study.GetMaterialCollection()
    particle_material = materials.AddSolidMaterial()
    particle_material.SetName("Particle Material")
    particle_material.SetDensity(p_density)
    particle_material.SetYoungsModulus({{YOUNGMOD_P}})
    particle_material.SetPoissonRatio({{POISSON_P}})
    particle_material.SetUseBulkDensity(False)
    wall_material = materials.AddSolidMaterial()
    wall_material.SetName("Wall Material")
    wall_material.SetDensity(2700)
    wall_material.SetYoungsModulus(1e9)
    wall_material.SetPoissonRatio(0.3)
    wall_material.SetUseBulkDensity(False)
    for wall in walls:
        wall.SetMaterial(wall_material)


def load_interactions() -> None:
    """Configure particle-particle and particle-wall interactions."""
    interactions = study.GetMaterialsInteractionCollection()
    for material, dynamic, static, restitution, stiffness, surface_energy in (
        (
            particle_material,
            {{DYNAMIC_FRICTION_PP}},
            {{STATIC_FRICTION_PP}},
            {{COR_PP}},
            {{TANG_STIFF_RATIO_PP}},
            {{SURF_EN_PP}},
        ),
        (
            wall_material,
            {{DYNAMIC_FRICTION_PW}},
            {{STATIC_FRICTION_PW}},
            {{COR_PW}},
            {{TANG_STIFF_RATIO_PW}},
            {{SURF_EN_PW}},
        ),
    ):
        interaction = interactions.GetMaterialsInteraction(particle_material, material)
        interaction.SetDynamicFriction(dynamic)
        interaction.SetStaticFriction(static)
        interaction.SetRestitutionCoefficient(restitution)
        if stiffness is not None:
            interaction.SetTangentialStiffnessRatio(stiffness)
        if {{ADH_MODEL | tojson}} == "JKR":
            if surface_energy <= 0:
                raise ValueError("JKR requires positive surface energies")
            interaction.SetSurfaceEnergy(surface_energy, "J/m2")


def gen_particle() -> None:
    """Create the particle and configure its shape and size distribution."""
    global particle

    shape_name = {{SHAPE | tojson}}
    if shape_name == "sphere":
        shape = particle_shapes.Sphere(radius=p_radius)
    elif shape_name == "polyhedron":
        shape = particle_shapes.Polyhedron(
            radius=p_radius,
            vert_ar={{VERT_AR}},
            horiz_ar={{HORIZ_AR}},
            n_corners={{N_CORNERS}},
            superquadric_degree={{SQ_DEGREE}},
        )
    elif shape_name == "sphero_cylinder":
        shape = particle_shapes.SpheroCylinder(radius=p_radius, vert_ar={{VERT_AR}})
    elif shape_name == "custom_polyhedron":
        shape = particle_shapes.CustomPolyhedron(
            stl_path={{PARTICLE_PATH | tojson}}, radius=p_radius
        )
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")
    particle = study.CreateParticle()
    particle.SetName("Particle")
    shape.particle2rocky(
        particle=particle,
        material=particle_material,
        rolling_friction={{ROLLING_FRICTION}},
    )


def sim_physics() -> None:
    """Configure contact force models and gravity."""
    physics = study.GetPhysics()
    physics.SetNormalForceModel({{NORMAL_MODEL | tojson}})
    physics.SetTangentialForceModel({{TANG_MODEL | tojson}})
    physics.SetAdhesionModel({{ADH_MODEL | tojson}})
    physics.SetRollingResistanceModel({{ROLLING_MODEL | tojson}})
    physics.SetGravityXDirection(0)
    physics.SetGravityYDirection(-9.81)
    physics.SetGravityZDirection(0)


def insertion_settings() -> None:
    """Set the mass flow and duration of the fill phase."""
    inlet = study.CreateParticleInlet(inlet_surface, particle)
    inlet.GetInputPropertiesList()[0].SetMassFlowRate(charge_mass / t_fill, "kg/s")
    inlet.SetStartTime(0.0, "s")
    inlet.SetStopTime(t_fill, "s")
    inlet.DisablePeriodic()


def rotate_drum() -> None:
    """Rotate the drum and both end caps after filling and settling."""
    frame = study.GetMotionFrameSource().NewFrame()
    frame.SetRelativePosition([0, 0, 0], "m")
    motion = frame.GetMotions().New()
    motion.SetType("Rotation")
    motion.GetTypeObject().SetInitialAngularVelocity(
        [0, 0, rpm * 2 * math.pi / 60], "rad/s"
    )
    motion.SetStartTime(t_fill + t_settle, "s")
    motion.SetStopTime(duration, "s")
    for wall in walls:
        frame.ApplyTo(wall)


def set_domain_settings() -> None:
    """Set nonperiodic coordinate limits around the drum."""
    domain = study.GetDomainSettings()
    domain.DisableUseBoundaryLimits()
    domain.DisablePeriodicAtGeometryLimits()
    domain.SetDomainType("CARTESIAN")
    domain.SetCartesianPeriodicDirections("NONE")
    domain.SetCoordinateLimitsMinValues([-1.1 * radius, -1.1 * radius, -0.55 * length])
    domain.SetCoordinateLimitsMaxValues([1.1 * radius, 1.1 * radius, 0.55 * length])


def load_modules() -> None:
    """Enable contact data collection."""
    contacts = study.GetContactData()
    contacts.EnableCollectContactsData()
    if {{ADH_MODEL | tojson}} != "none":
        contacts.EnableIncludeAdhesiveContacts()


def simulate() -> None:
    """Configure the solver and run until the rotating phase ends."""
    solver = study.GetSolver()
    processor = {{XPU}}
    solver.SetSimulationTarget(processor)
    if processor == "CPU":
        solver.SetNumberOfProcessors(
            int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
        )
    solver.SetTimestepModel("variable")
    solver.SetSimulationDuration(duration, "s")
    project.SaveProject()
    study.StartSimulation(non_blocking=True)
    while study.IsSimulating():
        study.RefreshResults()
        print(f"Simulation Progress: {study.GetProgress():.2f} %")


def post_process() -> dict:
    """Save the geometry, charge and final particle count."""
    results = {
        "drum_radius": radius,
        "drum_length": length,
        "drum_volume": box_len**3,
        "aspect_ratio": aspect_ratio,
        "rpm": rpm,
        "t_rotate": t_rotate,
        "fill_fraction": fill_fraction,
        "packing_fraction": packing_fraction,
        "target_particle_count": n_particles,
        "target_charge_mass": charge_mass,
        "final_particle_count": int(study.GetParticles().GetNumberOfParticles(-1)),
    }
    (Path.cwd() / "results.json").write_text(json.dumps(results, indent=2))
    return results


setup()
load_meshes()
load_material_properties()
load_interactions()
gen_particle()
sim_physics()
insertion_settings()
rotate_drum()
set_domain_settings()
load_modules()
simulate()
results = post_process()

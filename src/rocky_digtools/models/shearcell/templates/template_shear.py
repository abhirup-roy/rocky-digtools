"""
Script to run parallel shear test in Rocky DEM.

The parallel shear test designed to be a linear version of the Schulze shear test.
It allows for the configuration of various parameters such as particle properties,
adhesion models, force models, and simulation settings. The script also includes
functions to validate the parameters, export them to a JSON file, and run the simulation.

This script is provided as-is and the author does not take any responsibility for its use. Please use it at your own risk and ensure that you have the necessary permissions
and licenses to run Rocky DEM simulations.
"""

import os
import sys
import json
import importlib
from warnings import warn
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from typing import Optional

INSERTIONS = True  # Whether to use insertions or not
T_FILL = 0.5  # Time to fill the box with particles (s)


# Import particle shapes using importlib
shapes_spec = importlib.util.spec_from_file_location(
    'particles_shapes', os.path.abspath('../../../particles_shapes.py'))
if not shapes_spec:
    raise ImportError("Could not find the particles_shapes.py file.")
particle_shapes = importlib.util.module_from_spec(shapes_spec)
sys.modules['particles_shapes'] = particle_shapes  # Add to sys.modules
shapes_spec.loader.exec_module(particle_shapes)

# Set the current working directory as the project directory
project_dir: str = os.getcwd()
preshear_project_name: str = 'parallel_shear_pre.rocky'  # Change this if needed
preshear_path: str = os.path.join(
    project_dir,
    preshear_project_name
)

outputs_dir: str = os.path.join(
    project_dir,
    'pyoutputs'  # Change this if needed
)
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)

# For pyrocky API TODO: Add pyrocky implementation
# bb_rocky_path = '/rds/bear-apps/2023a/EL8-ice/software/ANSYS_Rocky/2024R2.0/bin/Rocky'
# vm_rocky_path = '/home/rocky-vm/ansys_inc/v242/rocky/bin/Rocky'

PROCESSOR = {{XPU}}
assert PROCESSOR in ['CPU', 'GPU', 'MULTI_GPU']
LOC = '{{LOC}}'  # 'bb-cpu', 'az-gpu', or 'custom'

def get_params(params):
    """
    Get the parameters for the parallel shear test.
    Exports to JSON file if dict is passed.
    """

    # While global variables are not recommended, they are used here for simplicity.
    # It helps deal with the 'quirks' of the Rocky API.
    global params_dict

    if isinstance(params, str) and params.endswith('.json'):
        params_dict = json.loads(params)
    elif isinstance(params, dict):
        params_dict = params
    else:
        raise ValueError("params must be a JSON string or a dictionary.")

    valid_keys = [
        'p_radius', 'p_density', 'p_youngmod',
        'p_poisson', 'rolling_model', 'pp_dynamic_friction',
        'pp_static_friction', 'pp_tangential_stiffness_ratio',
        'rolling_friction', 'pp_cor',
        'pw_dynamic_friction', 'pw_static_friction',
        'pw_tangential_stiffness_ratio', 'pw_cor',
        'normal_force_model', 'tangential_force_model',
        'adhesion', 'particle_box_len', 't_settle',
        't_compression', 'sigma_pre', 'n_shear_points',
        'processor', 'n_procs', 'neighbour_search',
        't_shear', 'shear_vel', 'shape', 'mesh_metrics'
    ]

    # Find the keys that are not in the list of valid keys
    invalid_keys = set(params_dict.keys()) - set(valid_keys)
    if invalid_keys:
        raise ValueError(
            f"Invalid parameter(s): {', '.join(invalid_keys)}. Expected one of {valid_keys}.")

    # Check if the values are compatible with Rocky API
    if 'rolling_model' in params_dict.keys():
        assert params_dict['rolling_model'] in [
            'type_1', 'type_3', 'none', 'custom']
        
        if 'rolling_friction' not in params_dict \
        and params_dict['rolling_model'] == 'none':
            params_dict['rolling_friction'] = 0.

    if 'normal_force_model' in params_dict.keys():
        assert params_dict['normal_force_model'] in [
            'linear_hysteresis', 'linear_elastic_viscous', 'damped_hertzian', 'custom']

    if 'tangential_force_model' in params_dict.keys():
        assert params_dict['tangential_force_model'] in [
            'elastic_coulomb', 'coulomb_limit', 'mindlin_deresiewicz', 'custom'
        ]

    if 'adhesion' in params_dict.keys():
        assert isinstance(params_dict['adhesion'], dict), \
            "If 'adhesion' is specified, it must be a dictionary."
    else:
        params_dict['adhesion'] = {
            "adhesion_model": "none"
        }

    match params_dict['adhesion']['adhesion_model']:
        case 'none' | None:
            pass

        case 'constant':
            assert 'adhesive_distance' in params_dict['adhesion'].keys(
            ), "If adhesion_model is 'constant', 'adhesive_distance' must be specified."
            assert 'force_fraction' in params_dict['adhesion'].keys(
            ), "If adhesion_model is 'constant', 'force_fraction' must be specified."

        case 'linear':
            assert 'adhesive_distance' in params_dict['adhesion'].keys(
            ), "If adhesion_model is 'linear', 'adhesive_distance' must be specified."
            assert 'stiffness_fraction' in params_dict['adhesion'].keys(
            ), "If adhesion_model is 'linear', 'stiffness_fraction' must be specified."

        case 'JKR':
            assert all(key in params_dict['adhesion'] for key in [
                'pp_surface_energy', 'pw_surface_energy'
            ]), \
                "If adhesion_model is 'JKR', 'pp_surface_energy' and 'pw_surface_energy' must be specified."
            assert params_dict['normal_force_model'] == 'damped_hertzian', \
                "If adhesion_model is 'JKR', normal_force_model must be 'damped_hertzian'."

    if 'neighbour_search' in params_dict.keys():
        assert params_dict['neighbour_search'] in [
            None, 'BVH', 'RegularGrid', 'SparseGrid'
        ]

    if isinstance(params, dict):
        # Save the parameters to a JSON file
        with open(os.path.join(project_dir, 'params.json'), 'w') as f:
            json.dump(params_dict, f, indent=4)


def setup_rocky(filename: str = preshear_project_name):
    """Create a new Rocky project"""

    global project, study

    project = app.CreateProject()  # type: ignore
    project.SaveProject(
        os.path.join(project_dir, filename)
    )
    study = project.GetStudy()
    study.SetName('Parallel Shear Test')


def import_meshes(
    meshdir: Optional[str] = 'meshes',
    bottom_stl_path: Optional[str] = 'bottomwall.stl',
    top_stl_path: Optional[str] = 'topwall.stl',
):
    """
    Import meshes from gmsh in meshes directory.

    Parameters
    ----------
    meshdir : str
        The directory where the meshes are located.
    bottom_stl_path : str
        The path to the bottom (shearing) wall STL file.
    top_stl_path : str
        The path to the top (compressing) wall STL file.
    """

    global study, walls

    # Use absolute paths for the STL files (quirk of Rocky API)
    top_path_abs = os.path.abspath(
        os.path.join(
            project_dir,
            meshdir,
            top_stl_path
        )
    )
    bottom_path_abs = os.path.abspath(
        os.path.join(
            project_dir,
            meshdir,
            bottom_stl_path
        )
    )

    # Check if the files exist
    if not os.path.isfile(top_path_abs):
        raise FileNotFoundError(
            f"File {top_path_abs} does not exist."
        )
    if not os.path.isfile(bottom_path_abs):
        raise FileNotFoundError(
            f"File {bottom_path_abs} does not exist."
        )

    # Import meshes
    top_wall = study.ImportWall(
        top_path_abs,
        import_scale=1.0,
        convert_yz=False)[0]
    top_wall.SetName('Compression Wall 1')

    vol_top = params_dict['mesh_metrics']['volume']
    cog_top = params_dict['mesh_metrics']['cog']
    pm_inert_top = params_dict['mesh_metrics']['pmoment_inertia']

    top_wall.SetPrincipalMomentOfInertia(pm_inert_top)
    top_wall.SetGravityCenter(cog_top)
    top_wall.SetBoundaryMass(vol_top * 2700)  # Assuming wall density of 2700 kg/m3

    if INSERTIONS:
        top_wall.SetEnableTime(T_FILL + 0.25)

    bottom_wall = study.ImportWall(
        bottom_path_abs,
        import_scale=1.0,
        convert_yz=False)[0]
    bottom_wall.SetName('Compression Wall 2')

    if INSERTIONS:
        insert_path = os.path.abspath(os.path.abspath(
            os.path.join(
                project_dir,
                meshdir,
                'insert.stl'
            )
        ))
        if not os.path.isfile(insert_path):
            raise FileNotFoundError(
                f"File {insert_path} does not exist."
            )

        insert_plane = study.ImportSurface(
            insert_path,
            import_scale=1.0,
            convert_yz=False
        )[0]

    # Save the walls in a global variable
    walls = [top_wall, bottom_wall, insert_plane] if INSERTIONS else [top_wall, bottom_wall]


def simulation_physics():
    """
    Set the simulation physics for the study.
    """

    global study, params_dict

    physics = study.GetPhysics()
    physics.SetNormalForceModel(params_dict['normal_force_model'])
    physics.SetTangentialForceModel(params_dict['tangential_force_model'])
    physics.SetAdhesionModel(params_dict['adhesion']['adhesion_model'])

    # Set gravity
    physics.SetGravityXDirection(0)
    physics.SetGravityYDirection(-9.81)
    physics.SetGravityZDirection(0)


def specify_materials():
    """
    Specify the materials for the particles and walls from params
    """

    global study, walls, params_dict, materials

    p_density = params_dict['p_density']
    p_youngmod = params_dict['p_youngmod']
    p_poisson = params_dict['p_poisson']

    material_collection = study.GetMaterialCollection()

    particle_mat = material_collection.AddSolidMaterial()
    particle_mat.SetName("Particle Material")
    particle_mat.SetDensity(p_density, 'kg/m3')
    particle_mat.SetYoungsModulus(p_youngmod, 'Pa')
    particle_mat.SetPoissonRatio(p_poisson)
    particle_mat.SetUseBulkDensity(False)

    # Wall material properties
    # Can be adjusted as needed, but wall materials don't matter as much
    wall_mat = material_collection.AddSolidMaterial()
    wall_mat.SetName("Wall Material")
    wall_mat.SetDensity(2700, 'kg/m3')
    wall_mat.SetYoungsModulus(5e6, 'Pa')
    wall_mat.SetPoissonRatio(0.3)
    wall_mat.SetUseBulkDensity(False)

    # Set the material for the meshes
    for wall in walls[:2]:
        wall.SetMaterial(wall_mat)

    materials = {
        'particle_mat': particle_mat,
        'wall_mat': wall_mat
    }


def material_interactions():
    """
    Set the material interactions for the particle-particle and
    particle-wall interactions.
    """

    global study, materials, params_dict

    interaction_collection = study.GetMaterialsInteractionCollection()
    pp_interaction = interaction_collection.GetMaterialsInteraction(
        materials['particle_mat'],
        materials['particle_mat']
    )
    pw_interaction = interaction_collection.GetMaterialsInteraction(
        materials['particle_mat'],
        materials['wall_mat']
    )

    pp_interaction.SetRestitutionCoefficient(
        params_dict['pp_cor']
    )
    pp_interaction.SetStaticFriction(
        params_dict['pp_static_friction']
    )
    pp_interaction.SetDynamicFriction(
        params_dict['pp_dynamic_friction']
    )
    if params_dict['rolling_model'] != 'none':
        pp_interaction.SetRollingFriction(
            params_dict['rolling_friction']
        )

    # Set the contact laws for the particle-wall interaction
    pw_interaction.SetRestitutionCoefficient(
        params_dict['pw_cor']
    )
    pw_interaction.SetStaticFriction(
        params_dict['pw_static_friction']
    )
    pw_interaction.SetDynamicFriction(
        params_dict['pw_dynamic_friction']
    )
    if params_dict['rolling_model'] != 'none':
        pw_interaction.SetRollingFriction(
            params_dict['pw_rolling_friction']
        )

    if params_dict['pp_tangential_stiffness_ratio'] is not None:
        pp_interaction.SetTangentialStiffnessRatio(
            params_dict['pp_tangential_stiffness_ratio']
        )
    if params_dict['pw_tangential_stiffness_ratio'] is not None:
        pw_interaction.SetTangentialStiffnessRatio(
            params_dict['pw_tangential_stiffness_ratio']
        )

    # Set the adhesion values
    adhesion_model = params_dict['adhesion']['adhesion_model']
    match adhesion_model:
        case 'none':
            pass
        case 'constant':
            pp_interaction.SetAdhesiveDistance(
                params_dict['adhesion']['adhesive_distance'], 'm'
            )
            pp_interaction.SetAdhesiveFraction(
                params_dict['adhesion']['force_fraction']
            )
        case 'linear':
            pp_interaction.SetAdhesiveDistance(
                params_dict['adhesion']['adhesive_distance'], 'm'
            )
            pp_interaction.SetAdhesiveFraction(
                params_dict['adhesion']['stiffness_fraction']
            )
        case 'JKR':
            pp_interaction.SetSurfaceEnergy(
                params_dict['adhesion']['pp_surface_energy'], 'J/m2'
            )
            pw_interaction.SetSurfaceEnergy(
                params_dict['adhesion']['pw_surface_energy'], 'J/m2'
            )


def insertions():
    """
    Create a volumetric inlet for the particles.
    """

    global study, particle, params_dict, walls

    fill_box_vol = params_dict['particle_box_len']**3
    radii = params_dict['p_radius']
    if isinstance(radii, dict):
        particle_vol = sum((4 / 3) * np.pi * r**3 * p for r, p in radii.items()) / sum(radii.values())
    else:
        particle_vol = 4 / 3 * np.pi * radii**3
    n_particles = fill_box_vol * 0.5/ particle_vol
    mass_particles = n_particles * params_dict['p_density'] * particle_vol

    if INSERTIONS:
        flowr = mass_particles / T_FILL  # kg/s

        particle_inlet = study.CreateParticleInlet(walls[2], particle)

        input_property_lst = particle_inlet.GetInputPropertiesList()
        input_property_lst[0].SetMassFlowRate(flowr, "kg/s")

        particle_inlet.SetStartTime(0)
        particle_inlet.SetStopTime(T_FILL)
        particle_inlet.DisablePeriodic()
    else:
        study.CreateVolumetricInlet(
            particle=particle,
            name='Volumetric Inlet',
            mass=mass_particles,
            seed_coordinates=[0, 0, 0],
            use_geometries_to_compute=False,
            box_center=[0, 0, 0],
            box_dimensions=[
                params_dict['particle_box_len'],
                params_dict['particle_box_len'],
                params_dict['particle_box_len']
            ],
        )



def gen_particle():
    
    global particle, study, params_dict
    
    particle = study.CreateParticle()
    match params_dict['shape']['shape_name']:
        case 'sphere':
            shape_obj = particle_shapes.Sphere(
                radius=params_dict['p_radius']
            )
        case 'polyhedron':
            shape_obj = particle_shapes.Polyhedron(
                radius=params_dict['p_radius'],
                vert_ar=params_dict['shape']['vert_ar'],
                horiz_ar=params_dict['shape']['horiz_ar'],
                n_corners=int(params_dict['shape']['n_corners']),
                superquadric_degree=params_dict['shape']['sq_degree']
            )
        case 'sphero_cylinder':
            shape_obj = particle_shapes.SpheroCylinder(
                radius=params_dict['p_radius'],
                vert_ar=params_dict['shape']['vert_ar']
            )
        case 'custom_polyhedron':
            shape_obj = particle_shapes.CustomPolyhedron(
                radius=params_dict['p_radius'],
                stl_path= params_dict['shape']['stl_path']
            )
        case _:
            raise ValueError(
                f"Unknown shape type: {params_dict['shape']['shape_name']}. "
                "Supported shapes are: 'sphere', 'sphero_cylinder', 'polyhedron', 'custom_polyhedron'."
            )
        
    shape_obj.particle2rocky(
        particle=particle,
        material=materials['particle_mat'],
        rolling_friction=params_dict['rolling_friction']
    )

def animate_walls():
    """
    Animate the walls for the parallel shear test.
    """

    global study, walls, params_dict

    frame_source = study.GetMotionFrameSource()
    topwall_frame = frame_source.NewFrame()
    # topwall_frame.SetEnableFbmLinearLimits(True)
    # topwall_frame.SetFbmMinLinearLimits(
    #     [-1, 0, -1]
    # )
    # topwall_frame.SetFbmMaxLinearLimits(
    #     [1, params_dict['particle_box_len']*1.2, 1]
    # )

    topwall_motions = topwall_frame.GetMotions()

    # Handling the free body motion
    freebody_motion = topwall_motions.New()
    freebody_motion.SetType('Free Body Translation')
    freebody = freebody_motion.GetTypeObject()
    freebody.SetFreeMotionDirection('y')
    if INSERTIONS:
        freebody_motion.SetStartTime(T_FILL + 0.25)
    else:
        freebody_motion.SetStartTime(0)

    freebody_motion.SetStopTime(
        sum([
            params_dict['t_settle'],
            params_dict['t_compression'],
            params_dict['t_shear']
        ])
    )

    # Compression motion for the top wall
    wall_mass = walls[0].GetBoundaryMass()
    force_magnitude = (params_dict['sigma_pre'] - wall_mass * 9.81) \
        * params_dict['particle_box_len']**2
    force_motion = topwall_motions.New()
    force_motion.SetType('Additional Force')

    add_force = force_motion.GetTypeObject()
    add_force.SetForceValue([0, -force_magnitude, 0], 'N')
    if INSERTIONS:
        force_motion.SetStartTime(T_FILL + 0.35)
    else:
        force_motion.SetStartTime(params_dict['t_settle'])
        
    force_motion.SetStopTime(
        sum([
            params_dict['t_settle'],
            params_dict['t_compression'],
            params_dict['t_shear']
        ])
    )

    topwall_frame.ApplyTo(walls[0])

    # Shearing motion for the bottom wall
    bottomwall_frame = frame_source.NewFrame()
    bottomwall_motions = bottomwall_frame.GetMotions()

    shearing_motion_bottom = bottomwall_motions.New()
    shearing_motion_bottom.SetType('Translation')
    translation_bottom = shearing_motion_bottom.GetTypeObject()
    translation_bottom.SetInput('fixed_velocity')
    translation_bottom.SetVelocity(
        [0, 0, params_dict['shear_vel']],
        'm/s'
    )

    shearing_motion_bottom.SetStartTime(
        sum([
            params_dict['t_settle'],
            params_dict['t_compression']
        ])
    )
    shearing_motion_bottom.SetStopTime(
        sum([
            params_dict['t_settle'],
            params_dict['t_compression'],
            params_dict['t_shear']
        ])
    )
    bottomwall_frame.ApplyTo(walls[1])


def config_domain():
    """
    Configure the domain settings and periodic conditions for the simulation.
    """

    global study, params_dict
     
    # Set periodic conditions and domain type
    domain_settings = study.GetDomainSettings()
    domain_settings.SetCartesianPeriodicDirections('XZ')
    domain_settings.SetDomainType('CARTESIAN')

    # Set the coordinate limits based on the particle box length
    particle_box_len = params_dict['particle_box_len']

    domain_settings.DisableUseBoundaryLimits()
    domain_settings.SetCoordinateLimitsMinValues([
        (-particle_box_len / 2) * 1.5,
        (-particle_box_len / 2) * 1.5,
        (-particle_box_len / 2) * 1.5
    ])
    domain_settings.SetCoordinateLimitsMaxValues([
        (particle_box_len / 2) * 1.5,
        (particle_box_len / 2) * 1.5,
        (particle_box_len / 2) * 1.5
    ])

    domain_settings.DisablePeriodicAtGeometryLimits()
    domain_settings.SetPeriodicLimitsMinCoordinates([
        -particle_box_len / 2,
        -np.inf,
        -particle_box_len / 2
    ])
    domain_settings.SetPeriodicLimitsMaxCoordinates([
        particle_box_len / 2,
        np.inf,
        particle_box_len / 2
    ])


def load_modules():
    """
    Load the Boundary Collision Statistics module
    and enable the intensity calculation.
    This module is used to compute the power input for the shearing wall
    """

    global study

    module_collection = study.GetModuleCollection()
    bcs = module_collection.GetModule('Boundary Collision Statistics')
    bcs.EnableModule()
    bcs.SetModuleProperty('Intensities', value=True)

    contacts_data = study.GetContactData()
    contacts_data.EnableCollectContactsData()
    if params_dict['adhesion']['adhesion_model'] != 'none':
        contacts_data.EnableIncludeAdhesiveContacts()


def _select_processor(solver, processor: str) -> None:
    """
    Handle the selection of the processor for the simulation.
    Based on the PROCESSOR variable, it sets the simulation target.
    Writes a warning to a file if GPU is not available and switches to CPU.

    **Parameters:**
    - `solver`: The solver object from the Rocky study.
    - `processor`: The processor to use for the simulation ('GPU' or 'CPU').
    """
    global params_dict
    if processor == 'GPU':
        if processor not in solver.GetValidSimulationTargetValues():
            warning_path = os.path.join(project_dir, 'warnings.txt')
            write_mode = 'w' if os.path.exists(warning_path) else 'a'
            with open(warning_path, write_mode) as f:
                f.write('GPU was not available - switching to CPU')
            solver.SetSimulationTarget('CPU')
        else:
            solver.SetSimulationTarget('GPU')

    elif processor == 'CPU':
        solver.SetSimulationTarget('CPU')
        nprocs = max(
            params_dict['n_procs'],
            int(os.environ.get('SLURM_CPUS_ON_NODE', 0))
        )
        solver.SetNumberOfProcessors(nprocs)
    
    params_dict['processor'] = processor
    if processor == 'CPU':
        params_dict['n_procs'] = nprocs
    
    with open(os.path.join(project_dir, 'params.json'), 'w') as f:
        json.dump(params_dict, f, indent=4)


def simulate(autotimestep: bool = True, timestep=None) -> None:
    """
    Run the simulation for the parallel shear test.
    """

    global study, project, params_dict

    solver = study.GetSolver()

    processor = PROCESSOR
    _select_processor(solver, processor)

    runtime = sum([
        params_dict['t_settle'],
        params_dict['t_compression'],
        params_dict['t_shear']
    ])
    solver.SetSimulationDuration(runtime, 's')

    if not autotimestep:
        if not timestep:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(1e-6, 's')
        else:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(timestep, 's')
    project.SaveProject()  # Save before run

    study.StartSimulation(skip_summary=True)
    while study.IsSimulating():
        study.RefreshResults()
        print(f"Simulation Progress: {study.GetProgress():.2f} %", flush=True)
    project.SaveProject()  # Save results


def continue_sim(filename: str):
    """
    This function is used to resume a simulation that was previously saved.
    It opens the project file, checks if the simulation can be resumed,
    and if so, resumes the simulation.
    If the simulation cannot be resumed, it moves to post-processing.

    Parameters
    ----------
    filename : str
        The name of the project file to open. This should be the path to the .rocky file.
        If the file is not found, it will raise a FileNotFoundError.
    """

    global study, project, walls

    project = app.OpenProject(filename)
    study = project.GetStudy()

    if study.CanResumeSimulation():
        study.ResumeSimulation()
        project.SaveProject()
    else:
        warn("Simulation cannot be resumed. Moving to post-processing.")

    geometry_collection = study.GetGeometryCollection()

    top_wall = geometry_collection.GetGeometry('Compression Wall 1')
    bottom_wall = geometry_collection.GetGeometry('Compression Wall 2')

    walls = [top_wall, bottom_wall]


def comp_shearstress(
        plot=True,
        window_size: int = 5,
        dump: bool = True
) -> float:
    """
    Compute the shear stress from the simulation results.
    Saves to a plots, tau and shearstress if dump is True.
    If dump is False, returns the average shear stress.

    Parameters
    ----------
    plot : bool, optional
        If True, plots the shear stress and saves it to a file.
        Defaults to True.
    window_size : int, optional
        The window size for the Savitzky-Golay filter.
        Defaults to 5.
    dump : bool, optional
        If True, saves the shear stress and sigma values to files.
        If False, returns the average shear stress.
        Defaults to True.
    """

    global params_dict, walls

    box_len = params_dict['particle_box_len']
    t_shear = params_dict['t_shear']
    shear_vel = params_dict['shear_vel']

    # Get the bottom wall
    geometry_collection = study.GetGeometryCollection()
    bottom_wall = geometry_collection.GetGeometry('Compression Wall 2')

    time_arr, power_lst = bottom_wall.GetNumpyCurve('Power')
    power_arr = np.array(power_lst)
    shear_arr = power_arr / (box_len**2 * shear_vel)

    shear_mask = np.where(time_arr >= t_shear)[0]
    shear_arr_masked = shear_arr[shear_mask]

    # Compute the shear stress
    shear_peaks_idx = signal.find_peaks(shear_arr)[0]
    shear_peaks = shear_arr[shear_peaks_idx]

    if len(shear_peaks) >= 5:
        tau_avg = shear_peaks[-5:].mean().item()
    elif len(shear_peaks) > 0:
        tau_avg = shear_peaks.mean().item()
    else:
        tau_avg = shear_arr_masked.mean().item()

    if plot:
        plt.plot(time_arr, shear_arr, label='Shear Stress')
        plt.axhline(
            y=tau_avg,
            color='r',
            linestyle='--',
            label='Average Shear Stress')
        if len(shear_peaks) > 0:
            plt.plot(
                time_arr[shear_peaks_idx],
                shear_peaks,
                'x',
                label='Shear Peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Shear Stress (Pa)')
        plt.legend()

        plt.savefig(
            os.path.join(
                outputs_dir,
                'shear_stress_preshear.png'
            )
        )
        plt.close()

    if dump:
        n_points = params_dict['n_shear_points']
        sigma_pre = params_dict['sigma_pre']
        sigma_arr = np.linspace(
            sigma_pre, 0, n_points, endpoint=False
        )
        tau_arr = np.zeros_like(sigma_arr)
        tau_arr[0] = tau_avg

        np.save(
            os.path.join(outputs_dir, 'sigma.npy'),
            sigma_arr
        )
        np.save(
            os.path.join(outputs_dir, 'shear_stresses.npy'),
            tau_arr
        )

    else:
        return tau_avg


def _new_sim_settings(
        sigma: float,
        particle_box_len: float,
        t_compression: float,
        shear_vel: float,
        t_shear: float,
        runtime: float):
    """
    Helper function to set the simulation settings
    for shear point cases.
    """

    project = app.GetProject()   # type: ignore
    study = project.GetStudy()

    geometry_collection = study.GetGeometryCollection()
    top_wall = geometry_collection.GetGeometry('Compression Wall 1')
    bottom_wall = geometry_collection.GetGeometry('Compression Wall 2')

    motion_frame_source = study.GetMotionFrameSource()

    topwall_frame = motion_frame_source.GetMotionFrame('Frame <01>')
    topwall_motions = topwall_frame.GetMotions()

    # Change the top wall compression
    release_motion = topwall_motions.New()
    release_motion.SetType('Translation')
    release_dist = 1e-5  # 10 um release distance
    t_release = 0.5
    release_translation = release_motion.GetTypeObject()
    release_translation.SetInput('fixed_velocity')
    release_translation.SetVelocity(
        [0, release_dist / t_release, 0], 'm/s'
    )
    release_motion.SetStartTime(0)
    release_motion.SetStopTime(t_release)
    sim_time = t_release

    force_magnitude = sigma * particle_box_len**2
    force_motion = topwall_motions.New()
    force_motion.SetType('Additional Force')
    add_force = force_motion.GetTypeObject()
    add_force.SetForceValue([0, -force_magnitude, 0], 'N')
    force_motion.SetStartTime(sim_time)
    force_motion.SetStopTime(sim_time + t_compression + t_shear)

    # Handling the free body motion
    freebody_motion = topwall_motions.New()
    freebody_motion.SetType('Free Body Translation')
    freebody = freebody_motion.GetTypeObject()
    freebody.SetFreeMotionDirection('y')
    freebody_motion.SetStartTime(sim_time)
    topwall_frame.ApplyTo(top_wall)

    sim_time += t_compression
    # Shearing motion for the bottom wall
    bottomwall_frame = motion_frame_source.GetMotionFrame('Frame <02>')
    bottomwall_motions = bottomwall_frame.GetMotions()
    shearing_motion_bottom = bottomwall_motions.New()
    shearing_motion_bottom.SetType('Translation')
    translation_bottom = shearing_motion_bottom.GetTypeObject()
    translation_bottom.SetInput('fixed_velocity')
    translation_bottom.SetVelocity(
        [0, 0, shear_vel], 'm/s'
    )
    shearing_motion_bottom.SetStartTime(sim_time)
    shearing_motion_bottom.SetStopTime(sim_time + t_shear)
    bottomwall_frame.ApplyTo(bottom_wall)

    # Set the simulation duration
    solver = study.GetSolver()
    solver.SetSimulationDuration(sim_time + t_shear)
    project.SaveProject()
    project.CloseProject(check_save_state=False)


def copy_script(target_dir: str, sigma: float):
    """
    Copy the script to the target directory
    and rename it to include the sigma value.
    This is useful for running multiple simulations in parallel

    Parameters
    ----------
    target_dir : str
        The directory where the script will be copied.
    sigma : float
        The sigma value to be included in the script name.
        This is used to differentiate between different simulations.
        The sigma value is expected to be in Pascals (Pa).
    """

    og_filename = "script_newcases.py"
    script_path = os.path.join(
        project_dir,
        og_filename
    )

    target_path = os.path.join(
        target_dir,
        f"script_newcases_{sigma/1000}kpa.py"
    )

    with open(script_path, 'r', encoding='utf-8') as f:
        script = f.read()
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(script)


def create_new_cases():
    """
    Create new cases for each sigma value.
    """
    global params_dict, project, script_paths

    # Get sigma values
    sigma_arr = np.load(
        os.path.join(
            outputs_dir,
            'sigma.npy'
        )
    )

    # Tracking the script paths and working directories
    script_paths = []
    sigma_cnt = 1

    while sigma_cnt < len(sigma_arr):
        # Make a new directory for each sigma value

        sigma = sigma_arr[sigma_cnt]
        new_dir = os.path.join(
            project_dir,
            f'sigma_{sigma/1000}kpa'
        )
        os.makedirs(
            new_dir,
            exist_ok=True
        )
        # Save restart file for each sigma value
        restart_filename = os.path.join(
            new_dir,
            f'parallel_shear_{sigma/1000}kpa.rocky'
        )
        new_script_path = os.path.join(
            new_dir,
            f'script_newcases_{sigma/1000}kpa.py'
        )

        script_paths.append(
            os.path.relpath(
                new_script_path)
        )

        project.SaveProjectForRestart(
            filename=restart_filename,
            timestep_or_index=-1)
        _new_sim_settings(
            sigma=sigma,
            particle_box_len=params_dict['particle_box_len'],
            t_compression=params_dict['t_compression'],
            shear_vel=params_dict['shear_vel'],
            t_shear=params_dict['t_shear'],
            runtime=sum([
                params_dict['t_settle'],
                params_dict['t_compression'],
                params_dict['t_shear']
            ])
        )
        if not os.path.isfile(preshear_path):
            raise FileNotFoundError(
                f"File {preshear_project_name} does not exist"
            )

        # Copy the script to the new directory
        copy_script(new_dir, sigma)

        project = app.OpenProject(preshear_path)  # type: ignore
        sigma_cnt += 1


def slurm_job(
        job_name: str = "parallel_shear",
        n_array: int = None,
        cpus_per_task: int = None,
        n_days: int = 10
):
    """
    Create a SLURM job script for the simulation.
    This script will run the Rocky simulations in parallel
    using the SLURM workload manager.

    Parameters
    ----------
    job_name : str, optional
        The name of the SLURM job. Defaults to "parallel_shear".
    n_array : int, optional
        The number of array jobs to run. If None, it will be set to the length
        of the script_paths list.
    cpus_per_task : int, optional
        The number of CPUs per task. If None, it will be set to the value in
        params_dict['n_procs'].
    n_days : int, optional
        The number of days for the job to run. Defaults to 10 days.
    """

    global params_dict, script_paths

    params_dict
    if not cpus_per_task:
        cpus_per_task = params_dict['n_procs']
    if not n_array:
        n_array = len(script_paths)

    # Use Jinja2 raw blocks to prevent template parsing of bash variables
    if LOC == 'bb-cpu':
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --array=1-{n_array}
#SBATCH --nodes=1
#SBATCH --time={n_days}-0
#SBATCH --qos=bbdefault
#SBATCH --mail-type=ALL
#SBATCH --account=windowcr-astrazeneca-abhi

set -e

module purge; module load bluebear
module load bear-apps/2023a
module load ANSYS_Rocky/2024R2.0

scripts=(
{' '.join([f'"{script}"' for script in script_paths])}
)

script_path=${% raw %}{{scripts[$SLURM_ARRAY_TASK_ID-1]}}{% endraw %}
script_dir=$(dirname "$script_path")
script_file=$(basename "$script_path")

echo "Selected script path is: $script_path"
echo "Attempting to change into directory: $script_dir"

cd "$script_dir" || {% raw %}{{ echo "FATAL: Could not change to directory. Exiting."; exit 1; }}{% endraw %}
Rocky --script "$script_file" --headless >> rocky.log
"""
    elif LOC == 'az-gpu':
        slurm_script = f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{n_array}
#SBATCH --time={n_days}-0
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name={job_name}
#SBATCH -p long-gpu

set -e

ml rocky/25.2.0

scripts=(
{' '.join([f'"{script}"' for script in script_paths])}
)

script_path=${% raw %}{{scripts[$SLURM_ARRAY_TASK_ID-1]}}{% endraw %}
script_dir=$(dirname "$script_path")
script_file=$(basename "$script_path")

echo "Selected script path is: $script_path"
echo "Attempting to change into directory: $script_dir"

cd "$script_dir" || {% raw %}{{ echo "FATAL: Could not change to directory. Exiting."; exit 1; }}{% endraw %}
Rocky --script "$script_file" --headless >> rocky.log
"""

    #  Write the SLURM script to a file
    slurm_script_path = os.path.join(project_dir, "run_sims.sh")
    with open(slurm_script_path, "w") as f:
        f.write(slurm_script)

    # Launch using sbatch
    try:
        subprocess.run(
            ['sbatch', slurm_script_path], check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")



# MAIN EXECUTION
# ----------------

params = {
    "p_radius": {{RADIUS_P}},
    "p_density": {{DENSITY_P}},
    "p_youngmod": {{YOUNGMOD_P}},
    "p_poisson": {{POISSON_P}},
    "rolling_model": {{ROLLING_MODEL}},
    "pp_dynamic_friction": {{DYNAMIC_FRICTION_PP}},
    "pp_static_friction": {{STATIC_FRICTION_PP}},
    "pp_cor": {{COR_PP}},
    "pp_tangential_stiffness_ratio": {{TANG_STIFF_RATIO_PP}},
    "pw_dynamic_friction": {{DYNAMIC_FRICTION_PW}},
    "pw_static_friction": {{STATIC_FRICTION_PW}},
    "pw_cor": {{COR_PW}},
    "pw_tangential_stiffness_ratio": {{TANG_STIFF_RATIO_PW}},
    "normal_force_model": {{NORMAL_MODEL}},
    "tangential_force_model": {{TANG_MODEL}},
    "adhesion": {
        "adhesion_model": "{{ADH_MODEL}}",
        "pp_surface_energy": {{SURF_EN_PP}},
        "pw_surface_energy": {{SURF_EN_PW}}
    },
    "particle_box_len": {{L_BOX}},
    "t_settle": {{T_SETTLE}},
    "t_compression": {{T_COMPRESSION}},
    "sigma_pre": {{SIGMA_PRE}},
    "n_shear_points": {{N_SHEAR_POINTS}},
    "n_procs": {{NPROCS}},
    "neighbour_search": {{NEIGHBOUR_SEARCH}},
    "t_shear": {{T_SHEAR}},
    "shear_vel": {{SHEAR_VEL}},
    "shape": {{SHAPE_DICT}},
    "mesh_metrics": {{MESH_METRICS_DICT}} 
}

get_params(params)

meshdir = f"../meshes_{params['particle_box_len']}"

if os.path.isfile(preshear_path):
    continue_sim(preshear_path)
else:
    print("File does not exist, creating a new project.",
          flush=True)
    setup_rocky()
    import_meshes(meshdir=meshdir)
    specify_materials()
    material_interactions()
    gen_particle()
    simulation_physics()
    insertions()
    animate_walls()
    config_domain()
    load_modules()
    print("Simulation setup complete.", flush=True)
    simulate()

comp_shearstress()
create_new_cases()
slurm_job(
    job_name="parallel_shear",
    n_array=len(script_paths),
    n_days=5
)

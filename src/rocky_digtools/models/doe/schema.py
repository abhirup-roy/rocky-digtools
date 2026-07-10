"""Generic Design-of-Experiments parameter schema and iteration engine.

Defines the parameter model shared by DEM sweep/OFAT experiments across
``rocky_digtools`` models: particle properties, particle-particle /
particle-wall interactions, contact models, and particle shape. Each model
(uniaxial compression, shear cell, ...) declares its own extension to this
common set — e.g. compression pressure for uniaxial compression — via
:class:`ParamSchema`, so :func:`iter_params` and :func:`iter_ofat` do not
need to know about model-specific fields.
"""

from __future__ import annotations

import itertools
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from ...particles_shapes import normalise_radius

# Fields common to every model's base JSON configuration, in canonical order.
# A model's extra `experiment_settings` fields (see ParamSchema) are inserted
# immediately after "box_len" and before the contact-model fields.
_COMMON_HEAD_FIELDS = (
    "radius",
    "density",
    "poisson",
    "youngmod",
    "fric_dyn_pp",
    "fric_stat_pp",
    "fric_rolling_pp",
    "cor_pp",
    "tang_stiff_ratio_pp",
    "surf_en_pp",
    "fric_dyn_pw",
    "fric_stat_pw",
    "cor_pw",
    "tang_stiff_ratio_pw",
    "surf_en_pw",
    "box_len",
)
_COMMON_TAIL_FIELDS = ("normal", "tangential", "rolling", "adhesion")

# Nested JSON key path for each common field in the base configuration.
_COMMON_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "radius": ("particle_properties", "radius"),
    "density": ("particle_properties", "density"),
    "poisson": ("particle_properties", "poisson"),
    "youngmod": ("particle_properties", "youngmod"),
    "fric_dyn_pp": ("interactions", "pp", "fric_dyn"),
    "fric_stat_pp": ("interactions", "pp", "fric_stat"),
    "fric_rolling_pp": ("interactions", "pp", "fric_rolling"),
    "cor_pp": ("interactions", "pp", "cor"),
    "tang_stiff_ratio_pp": ("interactions", "pp", "tang_stiff_ratio"),
    "surf_en_pp": ("interactions", "pp", "surf_en"),
    "fric_dyn_pw": ("interactions", "pw", "fric_dyn"),
    "fric_stat_pw": ("interactions", "pw", "fric_stat"),
    "cor_pw": ("interactions", "pw", "cor"),
    "tang_stiff_ratio_pw": ("interactions", "pw", "tang_stiff_ratio"),
    "surf_en_pw": ("interactions", "pw", "surf_en"),
    "box_len": ("experiment_settings", "box_len"),
    "normal": ("contact_model", "normal"),
    "tangential": ("contact_model", "tangential"),
    "rolling": ("contact_model", "rolling"),
    "adhesion": ("contact_model", "adhesion"),
}

_COMMON_DEFAULTS = {
    "surf_en_pp": 0.0,
    "surf_en_pw": 0.0,
    "tang_stiff_ratio_pp": None,
    "tang_stiff_ratio_pw": None,
}


def field_paths(schema: ParamSchema) -> dict[str, tuple[str, ...]]:
    """Map each schema field name to its nested JSON key path."""
    return {
        **_COMMON_FIELD_PATHS,
        **{f: ("experiment_settings", f) for f in schema.extra_experiment_fields},
    }


def get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Look up a value in nested dicts by key path."""
    for key in path:
        data = data[key]
    return data


def field_values(data: dict[str, Any], schema: ParamSchema) -> dict[str, Any]:
    """Read schema values, applying defaults for optional common fields."""
    values = {}
    for name, path in field_paths(schema).items():
        try:
            values[name] = get_nested(data, path)
        except KeyError:
            if name not in _COMMON_DEFAULTS:
                raise
            values[name] = _COMMON_DEFAULTS[name]
    return values


COMMON_RANGES: dict[str, tuple[float, Optional[float]]] = {
    "radius": (0, None),
    "density": (0, None),
    "poisson": (0, 0.5),
    "youngmod": (0, None),
    "fric_dyn_pp": (0, None),
    "fric_stat_pp": (0, None),
    "fric_rolling_pp": (0, None),
    "cor_pp": (0, 1),
    "tang_stiff_ratio_pp": (0, None),
    "surf_en_pp": (0, None),
    "fric_dyn_pw": (0, None),
    "fric_stat_pw": (0, None),
    "cor_pw": (0, 1),
    "tang_stiff_ratio_pw": (0, None),
    "surf_en_pw": (0, None),
    "box_len": (0, None),
    "vert_ar": (0, None),
    "horiz_ar": (0, None),
    "n_corners": (10, None),
    "sq_degree": (2, None),
}


@dataclass
class ShapeConfig:
    """Configuration for a particle shape in simulations.

    Attributes:
        name: Shape identifier (e.g. ``"sphere"``, ``"polyhedron"``).
        vert_ar: Vertical aspect ratio.
        horiz_ar: Horizontal aspect ratio.
        n_corners: Number of corners for polyhedral shapes.
        sq_degree: Superquadric degree.
        particle_path: File path to an STL for custom polyhedra.
        smoothness: Surface smoothness parameter.
    """

    name: str = "sphere"
    vert_ar: float = 1.0
    horiz_ar: float = 1.0
    n_corners: int = 6
    sq_degree: float = 2.0
    particle_path: str = ""
    smoothness: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> ShapeConfig:
        """Create a ShapeConfig from a dictionary.

        Missing keys are filled with class defaults.

        Args:
            d: Dictionary of shape configuration values.

        Returns:
            A new ``ShapeConfig`` instance.
        """
        return cls(
            name=d.get("name", "sphere"),
            vert_ar=d.get("vert_ar", 1.0),
            horiz_ar=d.get("horiz_ar", 1.0),
            n_corners=d.get("n_corners", 6),
            sq_degree=d.get("sq_degree", 2.0),
            particle_path=d.get("particle_path", ""),
            smoothness=d.get("smoothness", 0.5),
        )


@dataclass
class SimParams:
    """Typed representation of the parameters common to all DEM models.

    Model-specific parameters (e.g. compression pressure for uniaxial
    compression) are held in :attr:`extra`, keyed by the field names declared
    in that model's :class:`ParamSchema`.

    Attributes:
        radius: Particle radius in metres.
        density: Particle density in kg/m³.
        poisson: Poisson's ratio.
        youngmod: Young's modulus in Pa.
        fric_dyn_pp: Dynamic friction coefficient (particle-particle).
        fric_stat_pp: Static friction coefficient (particle-particle).
        fric_rolling_pp: Rolling friction coefficient (particle-particle).
        cor_pp: Coefficient of restitution (particle-particle).
        fric_dyn_pw: Dynamic friction coefficient (particle-wall).
        fric_stat_pw: Static friction coefficient (particle-wall).
        cor_pw: Coefficient of restitution (particle-wall).
        box_len: Length of the simulation box in metres.
        normal: Normal contact force model name.
        tangential: Tangential contact force model name.
        rolling: Rolling resistance model name.
        adhesion: Adhesion model name.
        tang_stiff_ratio_pp: Particle-particle tangential stiffness ratio.
        tang_stiff_ratio_pw: Particle-wall tangential stiffness ratio.
        surf_en_pp: Particle-particle surface energy in J/m².
        surf_en_pw: Particle-wall surface energy in J/m².
        shape: Particle shape configuration.
        extra: Model-specific parameter values, keyed by field name.
    """

    radius: float | dict[float, float]
    density: float
    poisson: float
    youngmod: float
    fric_dyn_pp: float
    fric_stat_pp: float
    fric_rolling_pp: float
    cor_pp: float
    fric_dyn_pw: float
    fric_stat_pw: float
    cor_pw: float
    box_len: float
    normal: str
    tangential: str
    rolling: str
    adhesion: str
    tang_stiff_ratio_pp: Optional[float] = None
    tang_stiff_ratio_pw: Optional[float] = None
    surf_en_pp: float = 0.0
    surf_en_pw: float = 0.0
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.radius = normalise_radius(self.radius)

    def __repr__(self) -> str:
        extra_repr = f", extra={self.extra!r}" if self.extra else ""
        return (
            f"SimParams(\n"
            f"    r={self.radius!r}, ρ={self.density:.3g}, ν={self.poisson:.3g}, E={self.youngmod:.3g},\n"
            f"    μ_pp={self.fric_dyn_pp:.3g}/{self.fric_stat_pp:.3g}, μ_pw={self.fric_dyn_pw:.3g}/{self.fric_stat_pw:.3g},\n"
            f"    e_pp={self.cor_pp:.3g}, e_pw={self.cor_pw:.3g},\n"
            f"    L={self.box_len:.3g}, shape={self.shape.name!r}{extra_repr}\n"
            f")\n"
        )


@dataclass(frozen=True)
class ParamSchema:
    """Declares a model's extension to the common DOE parameter set.

    Attributes:
        extra_experiment_fields: Names of additional ``experiment_settings`` JSON
            keys (beyond ``box_len``) that this model varies, e.g.
            ``("p_compress",)`` for uniaxial compression.
        extra_ranges: Validation ranges (``(lower, upper)``, ``upper=None``
            means unbounded) for the extra fields, used by OFAT bounds
            checking.
    """

    extra_experiment_fields: tuple[str, ...] = ()
    extra_ranges: dict[str, tuple[float, Optional[float]]] = field(default_factory=dict)

    @property
    def fields(self) -> tuple[str, ...]:
        """All parameter field names in canonical product/tuple order."""
        return _COMMON_HEAD_FIELDS + self.extra_experiment_fields + _COMMON_TAIL_FIELDS

    @property
    def ranges(self) -> dict[str, tuple[float, Optional[float]]]:
        """Validation ranges for all OFAT-tunable fields."""
        return {**COMMON_RANGES, **self.extra_ranges}


def _load_json(json_path: str) -> OrderedDict:
    with open(json_path, "r") as f_params:
        return json.load(f_params, object_pairs_hook=OrderedDict)


def _field_sources(params: OrderedDict, schema: ParamSchema) -> dict[str, Any]:
    """Map each schema field name to its (possibly list-valued) JSON source."""
    sources = field_values(params, schema)
    for name in _COMMON_DEFAULTS:
        if not isinstance(sources[name], list):
            sources[name] = [sources[name]]
    # A radius distribution is one parameter value, not a sweep iterable.
    if isinstance(sources["radius"], dict):
        sources["radius"] = [sources["radius"]]
    return sources


def _split_common_extra(
    schema: ParamSchema, values: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a field-name -> value mapping into common and extra kwargs."""
    common = {
        k: v for k, v in values.items() if k not in schema.extra_experiment_fields
    }
    extra = {k: v for k, v in values.items() if k in schema.extra_experiment_fields}
    return common, extra


def iter_params(json_path: str, schema: ParamSchema) -> list[SimParams]:
    """Read a sweep JSON configuration and expand all parameter combinations.

    Args:
        json_path: Path to the JSON configuration file defining parameter
            ranges for the sweep.
        schema: The model's :class:`ParamSchema`, declaring any parameter
            fields beyond the common set.

    Returns:
        List of :class:`SimParams` instances, one per parameter combination.
    """
    params = _load_json(json_path)

    shape_list = params["shape"]
    if not isinstance(shape_list, list):
        shape_list = [shape_list]

    field_sources = _field_sources(params, schema)
    ordered_fields = schema.fields

    combinations = itertools.product(
        *(field_sources[f] for f in ordered_fields), shape_list
    )

    result = []
    for combo in combinations:
        values = dict(zip(ordered_fields, combo[:-1]))
        shape_dict = combo[-1]
        common, extra = _split_common_extra(schema, values)
        result.append(
            SimParams(**common, shape=ShapeConfig.from_dict(shape_dict), extra=extra)
        )
    return result


def iter_ofat(
    json_path: str,
    ofat_values: dict[str, list | str],
    n_points: int,
    schema: ParamSchema,
) -> tuple[pd.DataFrame, dict]:
    """Compute all OFAT experiment points from a base configuration.

    Reads the base parameter values from a JSON file and generates an
    experiment matrix where each factor is varied independently while all
    others are held at a designated level.

    Args:
        json_path: Path to the JSON configuration file with base parameters.
        ofat_values: Dictionary specifying the OFAT design. Must contain:

            - ``"parameters"``: list of parameter names to vary.
            - ``"test_range"``: list of ``(lower, upper)`` tuples for each parameter.
            - ``"hold_values"``: list of hold strategies — ``"h"`` (high),
              ``"l"`` (low), or ``"m"`` (mid) — for the baseline of each parameter.

        n_points: Number of evenly-spaced levels to generate for each factor.
        schema: The model's :class:`ParamSchema`, declaring any parameter
            fields beyond the common set.

    Returns:
        A tuple ``(experiments_df, base_dict)`` where:

        - ``experiments_df`` is a :class:`~pandas.DataFrame` with one row per
            experiment.
        - ``base_dict`` is a dictionary of parameters that remain constant
            across all experiments.

    Raises:
        ValueError: If the OFAT specification is invalid, parameters are
            unrecognised, ranges are out of bounds, or list values appear in
            base parameters.
    """
    params = _load_json(json_path)

    if isinstance(params["shape"], list):
        raise ValueError("Shape parameters should be a single object, not a list.")

    ofat_base_valid = field_values(params, schema)
    ofat_base_valid.update(
        shape=params["shape"]["name"],
        vert_ar=params["shape"]["vert_ar"],
        horiz_ar=params["shape"]["horiz_ar"],
        n_corners=params["shape"]["n_corners"],
        sq_degree=params["shape"]["sq_degree"],
        particle_path=params["shape"].get("particle_path", ""),
        smoothness=params["shape"].get("smoothness", 0.5),
    )

    for name, value in ofat_base_valid.items():
        if isinstance(value, list):
            raise ValueError(
                f"Parameter values should not be lists. Use a single value for {name!r}."
            )

    if not isinstance(ofat_values, dict):
        raise ValueError("OFAT values must be provided as a dictionary.")

    if set(ofat_values.keys()) != {"parameters", "test_range", "hold_values"}:
        raise ValueError(
            "OFAT values must contain 'parameters', 'test_range', and 'hold_values' keys."
        )

    if not all(isinstance(ofat_values[key], list) for key in ofat_values):
        raise ValueError("OFAT parameters, test_range, and hold_values must be lists.")
    if not (
        len(ofat_values["parameters"])
        == len(ofat_values["test_range"])
        == len(ofat_values["hold_values"])
    ):
        raise ValueError("Mismatched lengths in OFAT values.")

    if len(set(ofat_values["parameters"])) != len(ofat_values["parameters"]):
        raise ValueError("OFAT parameters must be unique.")
    if not set(ofat_values["parameters"]).issubset(ofat_base_valid.keys()):
        raise ValueError(
            f"Invalid OFAT parameters. Allowed parameters are: {list(ofat_base_valid.keys())}"
        )

    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer of at least 2.")
    range_valid = schema.ranges

    # Single pass: validate each factor's baseline and test range, then build
    # its levels and hold point.
    levels = {}
    for k, test_range, hold_value in zip(
        ofat_values["parameters"],
        ofat_values["test_range"],
        ofat_values["hold_values"],
    ):
        if k not in range_valid:
            raise ValueError(
                f"OFAT parameter '{k}' is categorical and cannot be ranged."
            )
        if not isinstance(test_range, (tuple, list)) or len(test_range) != 2:
            raise ValueError(
                f"Test range for parameter '{k}' must be a (min, max) pair."
            )
        lb, ub = range_valid[k]
        ub = ub if ub is not None else float("inf")

        if not np.isfinite(ofat_base_valid[k]) or not (lb <= ofat_base_valid[k] <= ub):
            raise ValueError(
                f"Base parameter '{k}' with value {ofat_base_valid[k]} is out of range ({lb}, {ub})."
            )

        if hold_value not in ("h", "l", "m"):
            raise ValueError(
                f"Hold value '{hold_value}' for parameter '{k}' is not valid. Use 'h', 'l', or 'm'."
            )

        lb_i, ub_i = test_range
        if not np.isfinite([lb_i, ub_i]).all() or lb_i >= ub_i:
            raise ValueError(
                f"Invalid test range for parameter '{k}': ({lb_i}, {ub_i})"
            )
        if not (lb <= lb_i <= ub and lb <= ub_i <= ub):
            raise ValueError(
                f"Test range for parameter '{k}' with values ({lb_i}, {ub_i}) is out of bounds ({lb}, {ub})."
            )

        dtype = int if k == "n_corners" else float
        levels_i = np.linspace(lb_i, ub_i, n_points, dtype=dtype)
        if dtype is int:
            levels_i = np.unique(levels_i)
        hold_i = {
            "h": levels_i[-1],
            "l": levels_i[0],
            "m": levels_i[(levels_i.size - 1) // 2],
        }[hold_value]
        levels[k] = {"levels": levels_i, "hold": hold_i}

    baseline = {param: v["hold"] for param, v in levels.items()}
    experiments = [baseline.copy()]

    for factor, v in levels.items():
        for level in v["levels"]:
            if level != baseline[factor]:
                experiment = baseline.copy()
                experiment[factor] = level
                experiments.append(experiment)

    experiments_df = pd.DataFrame(experiments)

    ofat_vars = set(experiments_df.columns)
    base_vars = set(ofat_base_valid.keys())

    keys_to_drop = base_vars & ofat_vars
    for k in keys_to_drop:
        del ofat_base_valid[k]

    return experiments_df, ofat_base_valid


def get_unique_box_lens(params_list: list[SimParams]) -> set[float]:
    """Get unique box lengths from a list of :class:`SimParams`.

    Args:
        params_list: List of simulation parameter instances.

    Returns:
        Set of unique ``box_len`` values.
    """
    return {p.box_len for p in params_list}

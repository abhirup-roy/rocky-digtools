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

# Fields common to every model's base JSON configuration, in canonical order.
# A model's extra `experim_settings` fields (see ParamSchema) are inserted
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
    "fric_dyn_pw",
    "fric_stat_pw",
    "cor_pw",
    "box_len",
)
_COMMON_TAIL_FIELDS = ("normal", "tangential", "rolling", "adhesion")

COMMON_RANGES: dict[str, tuple[float, Optional[float]]] = {
    "fric_dyn_pp": (0, None),
    "fric_stat_pp": (0, None),
    "fric_rolling_pp": (0, None),
    "cor_pp": (0, 1),
    "fric_dyn_pw": (0, None),
    "fric_stat_pw": (0, None),
    "cor_pw": (0, 1),
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
        shape: Particle shape configuration.
        extra: Model-specific parameter values, keyed by field name.
    """

    radius: float
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
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        extra_repr = f", extra={self.extra!r}" if self.extra else ""
        return (
            f"SimParams(\n"
            f"    r={self.radius:.3g}, ρ={self.density:.3g}, ν={self.poisson:.3g}, E={self.youngmod:.3g},\n"
            f"    μ_pp={self.fric_dyn_pp:.3g}/{self.fric_stat_pp:.3g}, μ_pw={self.fric_dyn_pw:.3g}/{self.fric_stat_pw:.3g},\n"
            f"    e_pp={self.cor_pp:.3g}, e_pw={self.cor_pw:.3g},\n"
            f"    L={self.box_len:.3g}, shape={self.shape.name!r}{extra_repr}\n"
            f")\n"
        )


@dataclass(frozen=True)
class ParamSchema:
    """Declares a model's extension to the common DOE parameter set.

    Attributes:
        extra_experim_fields: Names of additional ``experim_settings`` JSON
            keys (beyond ``box_len``) that this model varies, e.g.
            ``("p_compress",)`` for uniaxial compression.
        extra_ranges: Validation ranges (``(lower, upper)``, ``upper=None``
            means unbounded) for the extra fields, used by OFAT bounds
            checking.
    """

    extra_experim_fields: tuple[str, ...] = ()
    extra_ranges: dict[str, tuple[float, Optional[float]]] = field(default_factory=dict)

    @property
    def fields(self) -> tuple[str, ...]:
        """All parameter field names in canonical product/tuple order."""
        return _COMMON_HEAD_FIELDS + self.extra_experim_fields + _COMMON_TAIL_FIELDS

    @property
    def ranges(self) -> dict[str, tuple[float, Optional[float]]]:
        """Validation ranges for all OFAT-tunable fields."""
        return {**COMMON_RANGES, **self.extra_ranges}


def _load_json(json_path: str) -> OrderedDict:
    with open(json_path, "r") as f_params:
        return json.load(f_params, object_pairs_hook=OrderedDict)


def _field_sources(params: OrderedDict, schema: ParamSchema) -> dict[str, Any]:
    """Map each schema field name to its (possibly list-valued) JSON source."""
    sources = {
        "radius": params["particle_properties"]["radius"],
        "density": params["particle_properties"]["density"],
        "poisson": params["particle_properties"]["poisson"],
        "youngmod": params["particle_properties"]["youngmod"],
        "fric_dyn_pp": params["inseractions"]["pp"]["fric_dyn"],
        "fric_stat_pp": params["inseractions"]["pp"]["fric_stat"],
        "fric_rolling_pp": params["inseractions"]["pp"]["fric_rolling"],
        "cor_pp": params["inseractions"]["pp"]["cor"],
        "fric_dyn_pw": params["inseractions"]["pw"]["fric_dyn"],
        "fric_stat_pw": params["inseractions"]["pw"]["fric_stat"],
        "cor_pw": params["inseractions"]["pw"]["cor"],
        "box_len": params["experim_settings"]["box_len"],
        "normal": params["contact_model"]["normal"],
        "tangential": params["contact_model"]["tangential"],
        "rolling": params["contact_model"]["rolling"],
        "adhesion": params["contact_model"]["adhesion"],
    }
    for extra_field in schema.extra_experim_fields:
        sources[extra_field] = params["experim_settings"][extra_field]
    return sources


def _split_common_extra(
    schema: ParamSchema, values: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a field-name -> value mapping into common and extra kwargs."""
    common = {k: v for k, v in values.items() if k not in schema.extra_experim_fields}
    extra = {k: v for k, v in values.items() if k in schema.extra_experim_fields}
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

    shape = params["shape"]
    if isinstance(shape, list):
        raise ValueError("Shape parameters should be a single object, not a list.")

    # Validate no list values in base parameters
    base_scalars = [
        params["particle_properties"]["radius"],
        params["particle_properties"]["density"],
        params["particle_properties"]["poisson"],
        params["particle_properties"]["youngmod"],
        params["inseractions"]["pp"]["fric_dyn"],
        params["inseractions"]["pp"]["fric_stat"],
        params["inseractions"]["pp"]["fric_rolling"],
        params["inseractions"]["pp"]["cor"],
        params["inseractions"]["pw"]["fric_dyn"],
        params["inseractions"]["pw"]["fric_stat"],
        params["inseractions"]["pw"]["cor"],
        params["experim_settings"]["box_len"],
        params["contact_model"]["normal"],
        params["contact_model"]["tangential"],
        params["contact_model"]["rolling"],
        params["contact_model"]["adhesion"],
        shape,
    ]
    for extra_field in schema.extra_experim_fields:
        base_scalars.append(params["experim_settings"][extra_field])

    for p in base_scalars:
        if isinstance(p, list):
            raise ValueError(
                f"Parameter values should not be lists. Use a single value for {p}."
            )

    if not isinstance(ofat_values, dict):
        raise ValueError("OFAT values must be provided as a dictionary.")

    ofat_dict_check = set(ofat_values.keys()) == set(
        ["parameters", "test_range", "hold_values"]
    )
    if not ofat_dict_check:
        raise ValueError(
            "OFAT values must contain 'parameters', 'test_range', and 'hold_values' keys."
        )

    ofat_base_valid = {
        "radius": params["particle_properties"]["radius"],
        "density": params["particle_properties"]["density"],
        "poisson": params["particle_properties"]["poisson"],
        "youngmod": params["particle_properties"]["youngmod"],
        "fric_dyn_pp": params["inseractions"]["pp"]["fric_dyn"],
        "fric_stat_pp": params["inseractions"]["pp"]["fric_stat"],
        "fric_rolling_pp": params["inseractions"]["pp"]["fric_rolling"],
        "cor_pp": params["inseractions"]["pp"]["cor"],
        "fric_dyn_pw": params["inseractions"]["pw"]["fric_dyn"],
        "fric_stat_pw": params["inseractions"]["pw"]["fric_stat"],
        "cor_pw": params["inseractions"]["pw"]["cor"],
        "box_len": params["experim_settings"]["box_len"],
        "normal": params["contact_model"]["normal"],
        "tangential": params["contact_model"]["tangential"],
        "rolling": params["contact_model"]["rolling"],
        "adhesion": params["contact_model"]["adhesion"],
        "shape": params["shape"]["name"],
        "vert_ar": params["shape"]["vert_ar"],
        "horiz_ar": params["shape"]["horiz_ar"],
        "n_corners": params["shape"]["n_corners"],
        "sq_degree": params["shape"]["sq_degree"],
    }
    for extra_field in schema.extra_experim_fields:
        ofat_base_valid[extra_field] = params["experim_settings"][extra_field]

    if not set(ofat_values["parameters"]).issubset(set(ofat_base_valid.keys())):
        raise ValueError(
            f"Invalid OFAT parameters. Allowed parameters are: {list(ofat_base_valid.keys())}"
        )

    range_valid = schema.ranges

    for k in ofat_values["parameters"]:
        lb, ub = range_valid[k]
        ub = ub if ub is not None else float("inf")
        if k not in ofat_base_valid:
            raise ValueError(f"Parameter '{k}' is not in the base parameters.")
        if not (lb <= ofat_base_valid[k] <= ub):
            raise ValueError(
                f"Base parameter '{k}' with value {ofat_base_valid[k]} is out of range ({lb}, {ub})."
            )

        param_idx = ofat_values["parameters"].index(k)
        test_range = ofat_values["test_range"][param_idx]
        hold_value = ofat_values["hold_values"][param_idx]

        if hold_value not in ["h", "l", "m"]:
            raise ValueError(
                f"Hold value '{hold_value}' for parameter '{k}' is not valid. Use 'h', 'l', or 'm'."
            )

        lb_i, ub_i = test_range
        if lb_i >= ub_i:
            raise ValueError(
                f"Invalid test range for parameter '{k}': ({lb_i}, {ub_i})"
            )
        elif not (lb <= lb_i <= ub and lb <= ub_i <= ub):  # type: ignore
            raise ValueError(
                f"Test range for parameter '{k}' with values ({lb_i}, {ub_i}) is out of bounds ({lb}, {ub})."
            )

    if len(ofat_values["parameters"]) != len(ofat_values["test_range"]) or len(
        ofat_values["hold_values"]
    ) != len(ofat_values["parameters"]):
        raise ValueError("Mismatched lengths in OFAT values.")

    levels = {}
    for i, rng in enumerate(ofat_values["test_range"]):
        lb, ub = rng
        if lb >= ub:
            raise ValueError(
                f"Invalid range for parameter '{ofat_values['parameters'][i]}': ({lb}, {ub})"
            )

        dtype = int if ofat_values["parameters"][i] == "n_corners" else float
        levels_i = np.linspace(lb, ub, n_points, dtype=dtype)
        if ofat_values["hold_values"][i] == "h":
            hold_i = levels_i[-1]
        elif ofat_values["hold_values"][i] == "l":
            hold_i = levels_i[0]
        elif ofat_values["hold_values"][i] == "m":
            hold_i = levels_i[(levels_i.size - 1) // 2]
        else:
            raise ValueError(
                f"Invalid hold value for parameter '{ofat_values['parameters'][i]}':\
                            {ofat_values['hold_values'][i]}. Select from 'h', 'l', 'm'."
            )
        levels[ofat_values["parameters"][i]] = {"levels": levels_i, "hold": hold_i}

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

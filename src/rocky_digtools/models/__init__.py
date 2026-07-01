"""Model-specific packages for rocky_digtools.

Each submodule implements a particular DEM test/experiment (e.g. uniaxial
compression, shear cell) on top of the general case-setup and Rocky API
utilities provided by the top-level package, and on the shared DOE
sweep/OFAT engine in :mod:`rocky_digtools.models.doe`.
"""

from . import doe
from . import uniax
from . import shearcell

__all__ = ["doe", "uniax", "shearcell"]
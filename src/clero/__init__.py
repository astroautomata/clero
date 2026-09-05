"""the CLimate Emulator for ROcky exoplanets.

CLERO takes a hypothetical planet and returns its 3D steady-state climate as 53 fields on a
32×64 latitude-longitude grid. It targets tidally locked ocean-covered rocky planets in or
near the habitable zone. See the [README](https://github.com/edstevenson/clero#readme) for
an introduction, [SCOPE.md](https://github.com/edstevenson/clero/blob/main/SCOPE.md) for the
range of validity and [UNCERTAINTY.md](https://github.com/edstevenson/clero/blob/main/UNCERTAINTY.md)
for how to use the predictive distribution.

`EARTH`, `M_EARTH` and `TRAPPIST1E` are ready-made planet inputs; `CORE_DOMAIN` and
`EXTENDED_DOMAIN` hold the valid input ranges.
"""

from importlib.metadata import version

from . import climate_analysis
from .domain import CORE_DOMAIN, EXTENDED_DOMAIN
from .inference import Emulator
from .orbit import orbital_period
from .presets import EARTH, M_EARTH, TRAPPIST1E

__version__ = version("clero")

__all__ = [  # order is the order on the documentation page
    "Emulator",
    "climate_analysis",
    "EARTH",
    "M_EARTH",
    "TRAPPIST1E",
    "CORE_DOMAIN",
    "EXTENDED_DOMAIN",
    "orbital_period",
]

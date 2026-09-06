"""the **CL**imate **E**mulator for **RO**cky exoplanets.

CLERO takes a hypothetical planet and returns its 3D steady-state climate as 53 fields on a
32×64 latitude-longitude grid. It targets tidally locked ocean-covered rocky planets in or
near the habitable zone. See the [README](https://github.com/astroautomata/clero#readme) for
an introduction.
"""

from importlib.metadata import version

from . import climate_analysis, inputs
from .inference import Emulator
from .inputs import CORE_DOMAIN, EARTH, EXTENDED_DOMAIN, M_EARTH, TRAPPIST1E, orbital_period

__version__ = version("clero")

__all__ = [  # order is the order on the documentation page
    "Emulator",
    "climate_analysis",
    "inputs",
]

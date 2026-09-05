"""CLERO — inference-only climate emulator for tidally locked ocean-covered rocky planets.

Two public surfaces:

- ``Emulator`` — turn planet parameters into predicted climate fields
  (``predict`` / ``sample``, with physical and model-space transforms).
- ``clero.climate_analysis`` — turn prediction dicts into science-facing summaries
  (global means, vertical profiles, maps); no pandas / xarray needed.

``EARTH``, ``M_EARTH`` and ``TRAPPIST1E`` are ready-made input dicts to start from. Inputs use
astro units: radius in Earth radii, P0 in bar. ``CORE_DOMAIN`` / ``EXTENDED_DOMAIN``
give the validity ranges from ``SCOPE.md``.
"""

from importlib.metadata import version

from . import climate_analysis
from .domain import CORE_DOMAIN, EXTENDED_DOMAIN
from .inference import Emulator
from .orbit import orbital_period
from .presets import EARTH, M_EARTH, TRAPPIST1E

__version__ = version("clero")

__all__ = [
    "CORE_DOMAIN",
    "EARTH",
    "EXTENDED_DOMAIN",
    "M_EARTH",
    "TRAPPIST1E",
    "Emulator",
    "__version__",
    "climate_analysis",
    "orbital_period",
]

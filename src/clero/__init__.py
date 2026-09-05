"""CLERO: the CLimate Emulator for ROcky exoplanets.

CLERO takes a hypothetical planet (eight parameters plus a target GCM) and returns its
3D steady-state climate as 53 fields on a 32×64 latitude–longitude grid. It targets
tidally locked ocean-covered rocky planets in or near the habitable zone.

```python
from clero import Emulator, TRAPPIST1E

emu = Emulator()
inputs = {**TRAPPIST1E, "P0": 1.0, "CO2": 4e-4, "CH4": 0.0, "GCM": "um"}
climate = emu.predict(inputs)                        # best point estimate, dict of (32, 64) fields
draws = emu.sample(inputs, n_samples=100, seed=0)    # 100 draws from the climate distribution
```

Start with the [README](https://github.com/edstevenson/clero#readme) for inputs,
outputs and install. [SCOPE.md](https://github.com/edstevenson/clero/blob/main/SCOPE.md)
gives the range of planets CLERO is valid for, and
[UNCERTAINTY.md](https://github.com/edstevenson/clero/blob/main/UNCERTAINTY.md) explains
model space, predictive variance and samples. Worked examples are in
[demos/](https://github.com/edstevenson/clero/tree/main/demos).

This page documents `Emulator` and the top-level helpers; the summary, profile,
map and diagnostic functions are in `clero.climate_analysis`.
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

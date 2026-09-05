"""Empirical stellar relations and Kepler's law, matching sampling/priors/orbit.py in XCE."""

import numpy as np
from numpy.typing import ArrayLike


def orbital_period(F_star: ArrayLike, T_star: ArrayLike) -> np.ndarray:
    """Estimate orbital period (days) from flux (W/m²) and stellar temperature (K).

    Uses Cassisi & Salaris (2019) + Duric (2004) below 3300 K, Mann (2013)
    below 4800 K, and Moya (2018) above. These empirical main-sequence relations
    are approximate and piecewise, with jumps at the boundaries. Prefer a measured
    period when available. Set P_rot to this estimate only under tidal locking.
    Inputs broadcast; scalar inputs return a zero-dimensional array.
    """
    flux, temperature = np.broadcast_arrays(np.asarray(F_star, dtype=float), np.asarray(T_star, dtype=float))
    mass, luminosity = np.full_like(temperature, np.nan), np.full_like(temperature, np.nan)
    cool = temperature < 3300
    medium = (temperature >= 3300) & (temperature < 4800)
    warm = temperature >= 4800

    luminosity[cool] = (0.763 * temperature[cool] / 5777 - 0.224) ** 2 * (temperature[cool] / 5777) ** 4
    mass[cool] = (luminosity[cool] / 0.23) ** (1 / 2.3)
    mass[medium] = np.polyval((2.65e-10, -3.488e-6, 1.544e-2, -22.297), temperature[medium])
    luminosity[medium] = np.polyval((2.95e-11, -2.49e-7, 7.40e-4, -0.781), temperature[medium])
    mass[warm] = -0.964 + 3.475e-4 * temperature[warm]
    luminosity[warm] = 10 ** ((np.log10(mass[warm]) + 0.0008) / 0.2227)
    return np.asarray(365.25 * (luminosity * 1361 / flux) ** 0.75 / np.sqrt(mass))

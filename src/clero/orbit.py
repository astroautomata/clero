"""Orbital period from stellar flux and temperature, via empirical main-sequence relations and Kepler's law."""

import numpy as np
from numpy.typing import ArrayLike


def orbital_period(F_star: ArrayLike, T_star: ArrayLike) -> np.ndarray:
    """Estimate the orbital period in days from instellation (W/m²) and stellar temperature (K).

    For a tidally locked planet this is also its rotation period, so it can be used as
    `P_rot` when no measured period is available (prefer a measured one when it is).
    Stellar mass and luminosity come from empirical main-sequence relations: Cassisi &
    Salaris (2019) with Duric (2004) below 3300 K, Mann et al. (2013) from 3300 to 4800 K,
    and Moya et al. (2018) above; the semi-major axis then follows from the flux and the
    period from Kepler's third law. The relations are approximate and piecewise, with
    small jumps at the regime boundaries. Accepts scalars or arrays.
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

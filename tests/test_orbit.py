import numpy as np
from numpy.testing import assert_allclose

from clero import orbital_period


def test_orbital_period_reference():
    # Reference values from the ThousandWorlds orbital prior (the relations described in the paper), including regime boundaries.
    assert_allclose(
        orbital_period(1361, [2600, 3300, 4000, 4800, 5777]),
        [5.03438215, 25.29746998, 72.02134640, 134.32582144, 415.26496203],
        rtol=1e-8,
    )
    assert orbital_period(1361, 2600).shape == ()


def test_orbital_period_flux_scaling():
    temperatures = np.array([2600, 4000, 5777])
    periods = orbital_period(np.array([[1361], [5444]]), temperatures)
    assert periods.shape == (2, 3)
    assert_allclose(periods[1], periods[0] * 4 ** -0.75)

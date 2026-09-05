"""Input domains from SCOPE.md (single source of truth; the SCOPE.md table is tested against these).

``CORE_DOMAIN`` is where the training simulations are densest and CLERO was tuned;
``EXTENDED_DOMAIN`` is the full training-set extent. Units match the ``Emulator`` inputs.
"""

CORE_DOMAIN = {
    "radius": (0.7, 1.75),
    "gravity": (6.0, 17.0),
    "P_rot": (1.0, 200.0),
    "P0": (0.5, 5.0),
    "CO2": (0.0, 1.0),
    "CH4": (0.0, 0.05),
    "F_star": (500.0, 1500.0),
    "T_star": (2500.0, 5800.0),
}

EXTENDED_DOMAIN = {
    "radius": (0.26, 2.76),
    "gravity": (4.7, 20.0),
    "P_rot": (0.25, 220.0),
    "P0": (0.1, 12.0),
    "CO2": (0.0, 1.0),
    "CH4": (0.0, 0.05),
    "F_star": (400.0, 3100.0),
    "T_star": (2500.0, 5800.0),
}

"""The input ranges from SCOPE.md, as `{input: (low, high)}` dicts in the units used by `Emulator`.

`CORE_DOMAIN` is where the training simulations are densest and CLERO is most reliable;
`EXTENDED_DOMAIN` is the full extent of the training set. Inputs outside the extended
domain emit a `UserWarning`.
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
"""Where the training simulations are densest and CLERO is most reliable: `{input: (low, high)}`."""

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
"""The full extent of the training set; inputs outside it emit a `UserWarning`: `{input: (low, high)}`."""

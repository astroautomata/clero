"""Ready-made planet inputs and the valid input ranges.

`EARTH`, `M_EARTH` and `TRAPPIST1E` are complete or near-complete input dicts to start from;
override what you need:

    from clero import Emulator, EARTH, M_EARTH, TRAPPIST1E

    emu = Emulator()
    climate = emu.predict({**EARTH, "CO2": 1e-3})                  # Earth with 1000 ppm CO2
    climate = emu.predict(M_EARTH)                                  # the same planet around an M dwarf
    climate = emu.predict({**TRAPPIST1E, "P0": 2.0, "CO2": 4e-4, "CH4": 0.0})  # TRAPPIST-1e with a 2 bar atmosphere

`CORE_DOMAIN` and `EXTENDED_DOMAIN` are the input ranges from SCOPE.md as `{input: (low, high)}`
dicts, in the units used by `Emulator`. All of these are also importable directly from `clero`.
"""

# Earth around the Sun (treated as a tidally locked aquaplanet). Its 365 d rotation period lies
# beyond the extended domain's 220 d bound (SCOPE.md), so predictions from EARTH emit a UserWarning.
EARTH = {
    "T_star": 5777.0,  # K
    "F_star": 1361.0,  # W/m^2
    "radius": 1.0,  # Earth radii
    "gravity": 9.807,  # m/s^2
    "P_rot": 365.25,  # days (tidally locked => = orbital period)
    "P0": 1.0,  # bar
    "CO2": 4.0e-4,  # volume fraction
    "CH4": 0.0,  # volume fraction
}
"""Earth around the Sun, as a tidally locked aquaplanet (its 365 d rotation period is beyond the extended domain, so it warns)."""

# Earth-like planet around a 2600 K M dwarf at Earth's instellation. The rotation period is the
# tidally locked orbital period from ThousandWorlds' stellar mass-luminosity relations (Kepler's law).
M_EARTH = {**EARTH, "T_star": 2600.0, "P_rot": 5.03}
"""Earth-like planet around a 2600 K M dwarf at Earth's instellation, with the tidally locked 5 d orbital period."""

# TRAPPIST-1e bulk and stellar parameters (Grimm+2018). Its atmosphere is observationally
# unconstrained, so P0/CO2/CH4 are intentionally left out.
TRAPPIST1E = {
    "T_star": 2600.0,  # K
    "F_star": 900.0,  # W/m^2
    "radius": 0.91,  # Earth radii
    "gravity": 9.12,  # m/s^2
    "P_rot": 6.1,  # days 
}
"""TRAPPIST-1e bulk and stellar parameters (Grimm et al. 2018); supply `P0`, `CO2` and `CH4` yourself."""

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

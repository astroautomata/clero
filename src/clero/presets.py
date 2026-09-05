"""Ready-made planet inputs: `EARTH`, `M_EARTH` (Earth around a 2600 K M dwarf) and `TRAPPIST1E`.

Use one as a starting point and override what you need:

    from clero import Emulator, EARTH, M_EARTH, TRAPPIST1E

    emu = Emulator()
    climate = emu.predict({**EARTH, "CO2": 1e-3})                  # Earth with 1000 ppm CO2
    climate = emu.predict(M_EARTH)                                  # the same planet around an M dwarf
    climate = emu.predict({**TRAPPIST1E, "P0": 2.0, "CO2": 4e-4, "CH4": 0.0})  # TRAPPIST-1e with a 2 bar atmosphere

`TRAPPIST1E` sets only the observationally constrained bulk and stellar parameters; its
atmosphere (`P0`, `CO2`, `CH4`) is unknown and must be supplied.
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
"""Earth-like planet around a 2600 K M dwarf at Earth's instellation, with the tidally locked 5 d orbital period."""

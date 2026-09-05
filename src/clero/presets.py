"""Presets for Earth, an Earth around an M dwarf, and TRAPPIST-1e. Use as a starting point and override what you need, e.g.:

    from clero import Emulator, EARTH, M_EARTH, TRAPPIST1E
    emu = Emulator()
    mean = emu.predict({**EARTH, "F_star": 3000})  # Earth-like ocean-covered planet around a red dwarf
    mean = emu.predict({**TRAPPIST1E,
                        "P0": 2.0,  # bar
                        "CO2": EARTH["CO2"],
                        "CH4": 1.0e-4})     
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

# Scope of validity

CLERO is restricted to tidally locked ocean-covered rocky planets. Do not use CLERO for dry planets, or asynchronous rotators, it will just be confused and predict a tidally locked ocean-covered rocky planet anyway!

Within that class, CLERO was tuned based on its performance over the **core domain**, where the training simulations are densest. 
The wider **extended domain** is a superset of the core that reflects the full training-set extent. CLERO can give reasonable predictions throughout the extended domain, but reliability degrades in sparser regions.
There's no training data outside the extended domain; CLERO is extrapolating there and may be less reliable.


| parameter             | core domain | extended domain                              |
| --------------------- | ----------- | -------------------------------------------- |
| radius / Earth radii  | 0.7 – 1.75  | 0.26 – 2.76                                  |
| gravity / m/s²        | 6.0 – 17.0  | 4.7 – 20.0                                   |
| P_rot / days          | 1 – 200     | 0.25 – 220                                   |
| P0 / bar              | 0.5 – 5     | 0.1 – 12                                     |
| CO2 / volume fraction | 0 – 1       | 0 – 1                                        |
| CH4 / volume fraction | 0 – 0.05    | 0 – 0.05                                     |
| F_star / W/m²         | 500 – 1500  | 400 – 3100                                   |
| T_star / K            | 2500 – 5800 | 2500 – 5800                                  |
| GCM                   | UM, ExoCAM  | UM, ExoCAM, ExoCAM-pre2022, ExoPlaSim, LFRic |


These bounds are available programmatically as `clero.CORE_DOMAIN` and `clero.EXTENDED_DOMAIN` (dicts of `(low, high)` per input).
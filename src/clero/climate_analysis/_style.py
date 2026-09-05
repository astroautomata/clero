"""Figure font sizing for climate-analysis plots.

Each plotting function is wrapped in ``_styled``, which applies the sizes via
``matplotlib.rc_context`` for the duration of the call, so the user's global
``rcParams`` are never altered. Adjust ``_BASE_FONTSIZE`` to rescale all figures.
"""
from __future__ import annotations

import functools

_BASE_FONTSIZE = 13.0


def _styled(plot_fn):
    @functools.wraps(plot_fn)
    def wrapper(*args, **kwargs):
        import matplotlib

        s = _BASE_FONTSIZE
        rc = {
            "font.size": s,
            "axes.titlesize": s,
            "axes.labelsize": s,  # axis labels + colorbar labels
            "xtick.labelsize": 0.85 * s,
            "ytick.labelsize": 0.85 * s,
            "legend.fontsize": 0.85 * s,
            "figure.labelsize": s,  # supxlabel / supylabel
        }
        with matplotlib.rc_context(rc):
            return plot_fn(*args, **kwargs)

    return wrapper

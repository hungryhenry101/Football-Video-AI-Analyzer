"""Manual pitch calibration bootstrap (vendored subset of the PitchCal project).

The annotator half lives here: ``geometry`` (pitch template + pinhole camera),
``solve`` (clicks/lines -> homography), ``anchors`` (JSON format),
``segments``/``linemask`` (white-paint evidence) and ``annotate`` (the
interactive GUI).  ``tracking`` (the LK/SIFT ground tracker) is also here
because ``core.pitch_tracker`` runs it as the default runtime backend.  The
standalone BEV renderer and HTML report are not vendored; ``main.py`` renders
the BEV.

World frame here is **z up**.  ``core.anchor_bootstrap`` converts anchors and
per-frame homographies to this project's z-down projection.
"""

__version__ = "0.1.0"


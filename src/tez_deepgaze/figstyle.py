"""Canvas sizes that leave figure text readable once the figure is on the page.

A figure drawn ten inches wide and placed in the thesis's 5.7-inch text block is
reduced to 57 % on paper, and its text shrinks with it, so 10 pt labels print at
under 6 pt against 11 pt body text. ``canvas`` sizes a figure from the width the
thesis gives it rather than from the drawing, so ``fontsize=10`` measures 8 pt
on paper. The figure keeps the proportions it was drawn with; only the canvas
shrinks, and ``tight_layout`` re-flows the panels for it. dpi is set so that
whatever the canvas, the result prints at about 300 pixels per printed inch.

Pixel dimensions are therefore not fixed. A figure drawn much wider than the
space it prints in carries more pixels per printed inch than a printer reads,
and the excess is discarded when the page is rendered; a figure drawn near the
size it prints at keeps every pixel.
"""
from __future__ import annotations

# \textwidth and \textheight of thesis-tex/main.tex, in inches. Printed by
# typearea into main.log; re-read them there if the class options or the
# binding correction move.
TEXTWIDTH_IN = 413.86 / 72.27
TEXTHEIGHT_IN = 623.00 / 72.27
# matplotlib's default font size, and what it should measure once printed.
BASE_PT = 10.0
TARGET_PT = 8.0
# Resolution of the placed figure, in pixels per inch of paper.
PRINT_PPI = 300


def canvas(aspect: tuple[float, float], page_frac: float = 1.0,
           page_height_frac: float | None = None) -> dict:
    """``figsize`` and ``dpi`` for a figure the thesis places at ``page_frac``.

    ``aspect`` is the width and height the figure was drawn with; only their
    ratio is used. ``page_frac`` is the fraction of ``\\textwidth`` the
    ``\\includegraphics`` in the chapter asks for. ``page_height_frac`` is the
    fraction of ``\\textheight`` it caps the figure at, for the one figure the
    thesis gives a height as well; a figure taller than the cap is narrowed
    until it fits, the way ``keepaspectratio`` narrows it on the page.
    """
    width = page_frac * TEXTWIDTH_IN * BASE_PT / TARGET_PT
    height = width * aspect[1] / aspect[0]
    if page_height_frac is not None:
        cap = page_height_frac * TEXTHEIGHT_IN * BASE_PT / TARGET_PT
        if height > cap:
            width *= cap / height
            height = cap
    return {
        "figsize": (width, height),
        "dpi": PRINT_PPI * TARGET_PT / BASE_PT,
    }

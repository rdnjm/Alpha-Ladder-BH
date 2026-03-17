"""Formatting utilities for the Alpha Ladder BH dashboard.

Provides consistent number formatting, color coding, and display helpers
matching the main Alpha Ladder dashboard style.
"""

from __future__ import annotations


def fmt_decimal(val: float, sig_figs: int = 6) -> str:
    """Format a number with the specified number of significant figures.

    Parameters
    ----------
    val : float
        The value to format.
    sig_figs : int
        Number of significant figures (default 6).

    Returns
    -------
    str
        Formatted string representation.
    """
    if val == 0:
        return "0"
    fmt = f"{{:.{sig_figs}g}}"
    return fmt.format(val)


def fmt_percent(val: float, decimals: int = 4) -> str:
    """Format a value as a percentage string.

    Parameters
    ----------
    val : float
        The fractional value (e.g. 0.05 for 5%).
    decimals : int
        Decimal places in the percentage (default 4).

    Returns
    -------
    str
        Formatted percentage string, e.g. "5.0000%".
    """
    fmt = f"{{:.{decimals}f}}%"
    return fmt.format(val * 100)


def fmt_sigma(sigma: float) -> str:
    """Format a sigma tension value with color markup for Streamlit.

    Parameters
    ----------
    sigma : float
        The tension in units of sigma (standard deviations).

    Returns
    -------
    str
        Markdown-safe string with color span indicating severity.
        Green for < 2 sigma, orange for 2-5, red for > 5.
    """
    abs_sigma = abs(sigma)
    if abs_sigma < 2:
        color = "#4caf50"
    elif abs_sigma < 5:
        color = "#ff9800"
    else:
        color = "#f44336"
    sign = "+" if sigma > 0 else ""
    return f'<span style="color:{color};font-weight:bold">{sign}{sigma:.1f} sigma</span>'


def color_by_quality(err_pct: float) -> str:
    """Return a CSS color string based on the magnitude of a percentage error.

    Parameters
    ----------
    err_pct : float
        The error as a percentage (e.g. 0.5 for 0.5%).

    Returns
    -------
    str
        Hex color code: green for < 1%, orange for 1-10%, red for > 10%.
    """
    abs_err = abs(err_pct)
    if abs_err < 1:
        return "#4caf50"
    elif abs_err < 10:
        return "#ff9800"
    else:
        return "#f44336"

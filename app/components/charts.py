"""
Plotly chart builders for the Alpha Ladder BH dashboard.

10 chart functions covering Gibbons-Maeda black hole observables:
temperature, entropy, QNMs, shadows, EHT constraints, ISCO/efficiency,
Hawking spectrum, greybody factors, effects scaling, and constraint summary.

All charts use a consistent dark theme matching the Streamlit dashboard.
"""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLOR_GM_BLUE = "#60a5fa"
COLOR_PURPLE = "#a78bfa"
COLOR_AMBER = "#f59e0b"
COLOR_GREEN = "#34d399"
COLOR_RED = "#f87171"

PALETTE = [COLOR_GM_BLUE, COLOR_PURPLE, COLOR_AMBER, COLOR_GREEN, COLOR_RED]

# ---------------------------------------------------------------------------
# Dark theme constants
# ---------------------------------------------------------------------------
PAPER_BG = "#0e1117"
PLOT_BG = "#1a1d23"
GRID_COLOR = "#2e3440"
FONT_COLOR = "#e0e0e0"
FONT_FAMILY = "Fira Mono"


# ---------------------------------------------------------------------------
# Helper: base layout dict
# ---------------------------------------------------------------------------

def _base_layout(
    title: str,
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> dict[str, Any]:
    """Return a standard Plotly layout dict with dark theme settings."""
    return dict(
        title=dict(text=title, font=dict(size=16, color=FONT_COLOR)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color=FONT_COLOR, size=12),
        xaxis=dict(
            title=xaxis_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=FONT_COLOR, size=11),
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
        ),
    )


def _empty_figure(message: str = "No data available") -> go.Figure:
    """Return an empty figure with a centred annotation."""
    fig = go.Figure()
    layout = _base_layout(message, "", "")
    layout["xaxis"]["visible"] = False
    layout["yaxis"]["visible"] = False
    fig.update_layout(**layout)
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=FONT_COLOR, family=FONT_FAMILY),
    )
    return fig


def _safe_list(data: Any, key: str | None = None) -> list[dict]:
    """Extract a list of dicts from *data*, tolerating None/empty."""
    if data is None:
        return []
    if key is not None:
        data = data.get(key, []) if isinstance(data, dict) else []
    if not isinstance(data, list):
        return []
    return data


# ---------------------------------------------------------------------------
# 1. Temperature profile
# ---------------------------------------------------------------------------

def temperature_profile_chart(profile_data: Any) -> go.Figure:
    """Hawking temperature ratio vs charge ratio Q/Q_ext."""
    rows = _safe_list(profile_data)
    if not rows:
        return _empty_figure("No temperature data available")

    x = [d.get("qm_ratio", 0) for d in rows]
    y = [d.get("T_ratio", 0) for d in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name="T_H / T_Schwarz",
        line=dict(color=COLOR_GM_BLUE, width=2),
    ))

    # Schwarzschild reference
    fig.add_hline(
        y=1.0, line_dash="dash", line_color=COLOR_PURPLE, line_width=1,
        annotation_text="Schwarzschild",
        annotation_font=dict(color=COLOR_PURPLE, size=10),
    )

    # Annotate extremal point (last point where qm_ratio is closest to 1)
    extremal_idx = max(range(len(x)), key=lambda i: x[i])
    fig.add_annotation(
        x=x[extremal_idx],
        y=y[extremal_idx],
        text="Finite T at extremality",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLOR_AMBER,
        font=dict(color=COLOR_AMBER, size=10),
        ax=-40,
        ay=-30,
    )

    layout = _base_layout(
        "Hawking Temperature vs Charge Ratio",
        xaxis_title="Q / Q_ext",
        yaxis_title="T_H / T_Schwarzschild",
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 2. Entropy profile
# ---------------------------------------------------------------------------

def entropy_profile_chart(profile_data: Any) -> go.Figure:
    """Entropy correction ratio(s) vs charge ratio."""
    rows = _safe_list(profile_data)
    if not rows:
        return _empty_figure("No entropy data available")

    x = [d.get("qm_ratio", 0) for d in rows]
    y_same_rp = [d.get("S_ratio", 0) for d in rows]
    y_same_m = [d.get("S_over_S_schwarz_same_mass", None) for d in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_same_rp,
        mode="lines",
        name="S / S_Schwarz (same r+)",
        line=dict(color=COLOR_GM_BLUE, width=2),
    ))

    # Second line only if data present
    if any(v is not None for v in y_same_m):
        fig.add_trace(go.Scatter(
            x=x,
            y=[v if v is not None else float("nan") for v in y_same_m],
            mode="lines",
            name="S / S_Schwarz (same M)",
            line=dict(color=COLOR_PURPLE, width=2),
        ))

    fig.add_hline(
        y=1.0, line_dash="dash", line_color=COLOR_AMBER, line_width=1,
        annotation_text="Schwarzschild",
        annotation_font=dict(color=COLOR_AMBER, size=10),
    )

    layout = _base_layout(
        "Entropy Correction vs Charge Ratio",
        xaxis_title="Q / Q_ext",
        yaxis_title="S / S_Schwarzschild",
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 3. QNM spectrum
# ---------------------------------------------------------------------------

def qnm_spectrum_chart(spectrum_data: Any) -> go.Figure:
    """Scatter plot of QNM frequencies coloured by angular index l.

    Accepts the dict returned by compute_qnm_spectrum(), which has
    ``modes`` as a dict keyed by ``"l=2"``, ``"l=3"``, etc.  Each value
    is a dict with ``"GM"`` and ``"Schwarzschild"`` sub-dicts containing
    ``f_Hz``, ``tau_ms``, ``omega_R``, ``omega_I``, etc.
    """
    if not spectrum_data or not isinstance(spectrum_data, dict):
        return _empty_figure("No QNM data available")

    modes = spectrum_data.get("modes", {})
    if not modes or not isinstance(modes, dict):
        return _empty_figure("No QNM data available")

    symbols = ["circle", "diamond", "square", "cross", "triangle-up",
               "hexagon", "star", "pentagon"]

    fig = go.Figure()
    for idx, (l_key, mode_dict) in enumerate(sorted(modes.items())):
        if not isinstance(mode_dict, dict):
            continue
        gm = mode_dict.get("GM", {})
        schwarz = mode_dict.get("Schwarzschild", {})
        color = PALETTE[idx % len(PALETTE)]
        symbol = symbols[idx % len(symbols)]

        # GM point
        f_gm = abs(gm.get("f_Hz", 0))
        tau_gm = abs(gm.get("tau_ms", 0))
        if f_gm > 0:
            fig.add_trace(go.Scatter(
                x=[f_gm], y=[1.0 / (tau_gm * 1e-3) if tau_gm > 0 else 0],
                mode="markers",
                name=f"{l_key} (GM)",
                marker=dict(color=color, symbol=symbol, size=10,
                            line=dict(width=1, color=FONT_COLOR)),
            ))

        # Schwarzschild point for comparison
        f_s = abs(schwarz.get("f_Hz", 0))
        tau_s = abs(schwarz.get("tau_ms", 0))
        if f_s > 0:
            fig.add_trace(go.Scatter(
                x=[f_s], y=[1.0 / (tau_s * 1e-3) if tau_s > 0 else 0],
                mode="markers",
                name=f"{l_key} (Schwarz)",
                marker=dict(color=color, symbol=symbol, size=10,
                            opacity=0.4,
                            line=dict(width=1, color=FONT_COLOR)),
            ))

    if not fig.data:
        return _empty_figure("No QNM data available")

    layout = _base_layout(
        "QNM Spectrum (Gibbons-Maeda)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Damping rate (1/s)",
    )
    layout["xaxis"]["type"] = "log"
    layout["yaxis"]["type"] = "log"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 4. Shadow scan
# ---------------------------------------------------------------------------

def shadow_scan_chart(scan_data: Any, comparison_data: Any) -> go.Figure:
    """Shadow size deviation (%) vs charge for GM and RN."""
    scan_rows = _safe_list(scan_data, key="scan")
    comp_rows = _safe_list(comparison_data)
    if not scan_rows and not comp_rows:
        return _empty_figure("No shadow data available")

    fig = go.Figure()

    # Prefer comparison_data if it has both GM and RN
    if comp_rows:
        x = [d.get("q", 0) for d in comp_rows]
        y_gm = [d.get("delta_gm_percent", 0) for d in comp_rows]
        y_rn = [d.get("delta_rn_percent", 0) for d in comp_rows]
        fig.add_trace(go.Scatter(
            x=x, y=y_gm, mode="lines",
            name="Gibbons-Maeda",
            line=dict(color=COLOR_GM_BLUE, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_rn, mode="lines",
            name="Reissner-Nordstrom",
            line=dict(color=COLOR_PURPLE, width=2),
        ))
    elif scan_rows:
        x = [d.get("q", 0) for d in scan_rows]
        y = [d.get("delta_theta_percent", 0) for d in scan_rows]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            name="Gibbons-Maeda",
            line=dict(color=COLOR_GM_BLUE, width=2),
        ))

    layout = _base_layout(
        "Shadow Size vs Charge: GM vs RN",
        xaxis_title="q = Q / M",
        yaxis_title="delta_b (% vs Schwarzschild)",
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 5. EHT constraint
# ---------------------------------------------------------------------------

def eht_constraint_chart(eht_data: Any) -> go.Figure:
    """Horizontal bar chart of EHT charge constraints for Sgr A* and M87*."""
    if not eht_data or not isinstance(eht_data, dict):
        return _empty_figure("No EHT constraint data available")

    sources: list[str] = []
    q1: list[float] = []
    q2: list[float] = []

    for key, label in [("sgra", "Sgr A*"), ("m87", "M87*")]:
        entry = eht_data.get(key)
        if entry and isinstance(entry, dict):
            sources.append(label)
            q1.append(entry.get("q_max_1sigma", 0))
            q2.append(entry.get("q_max_2sigma", 0))

    if not sources:
        return _empty_figure("No EHT constraint data available")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sources, x=q1, orientation="h",
        name="1-sigma",
        marker_color=COLOR_AMBER,
    ))
    fig.add_trace(go.Bar(
        y=sources, x=q2, orientation="h",
        name="2-sigma",
        marker_color=COLOR_RED,
    ))

    layout = _base_layout(
        "EHT Charge Constraints",
        xaxis_title="q_max",
        yaxis_title="",
    )
    layout["barmode"] = "group"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 6. ISCO scan (two subplots)
# ---------------------------------------------------------------------------

def isco_scan_chart(scan_data: Any, comparison_data: Any) -> go.Figure:
    """Two-panel plot: r_ISCO/M and radiative efficiency vs charge."""
    comp_rows = _safe_list(comparison_data)
    scan_rows = _safe_list(scan_data)
    rows = comp_rows if comp_rows else scan_rows
    if not rows:
        return _empty_figure("No ISCO data available")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("r_ISCO / M vs q", "Efficiency (%) vs q"),
        horizontal_spacing=0.12,
    )

    # Keys differ depending on source
    q = [d.get("q", 0) for d in rows]

    # Left panel: r_ISCO / M
    r_gm = [d.get("r_isco_gm_over_M", d.get("r_isco_over_M", None)) for d in rows]
    r_rn = [d.get("r_isco_rn_over_M", None) for d in rows]

    if any(v is not None for v in r_gm):
        fig.add_trace(go.Scatter(
            x=q, y=[v if v is not None else float("nan") for v in r_gm],
            mode="lines", name="GM",
            line=dict(color=COLOR_GM_BLUE, width=2),
        ), row=1, col=1)
    if any(v is not None for v in r_rn):
        fig.add_trace(go.Scatter(
            x=q, y=[v if v is not None else float("nan") for v in r_rn],
            mode="lines", name="RN",
            line=dict(color=COLOR_PURPLE, width=2),
        ), row=1, col=1)

    # Schwarzschild reference r_ISCO = 6M
    fig.add_hline(y=6.0, line_dash="dash", line_color=COLOR_AMBER,
                  line_width=1, row=1, col=1)

    # Right panel: efficiency
    eta_gm = [d.get("eta_gm_percent", d.get("eta_percent", None)) for d in rows]
    eta_rn = [d.get("eta_rn_percent", None) for d in rows]

    if any(v is not None for v in eta_gm):
        fig.add_trace(go.Scatter(
            x=q, y=[v if v is not None else float("nan") for v in eta_gm],
            mode="lines", name="GM",
            line=dict(color=COLOR_GM_BLUE, width=2),
            showlegend=False,
        ), row=1, col=2)
    if any(v is not None for v in eta_rn):
        fig.add_trace(go.Scatter(
            x=q, y=[v if v is not None else float("nan") for v in eta_rn],
            mode="lines", name="RN",
            line=dict(color=COLOR_PURPLE, width=2),
            showlegend=False,
        ), row=1, col=2)

    # Schwarzschild efficiency ~ 5.72%
    fig.add_hline(y=5.72, line_dash="dash", line_color=COLOR_AMBER,
                  line_width=1, row=1, col=2)

    # Apply theme
    base = _base_layout("ISCO & Efficiency vs Charge", "", "")
    fig.update_layout(
        title=base["title"],
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=base["font"],
        margin=base["margin"],
        legend=base["legend"],
    )
    for axis_key in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{
            axis_key: dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        })
    fig.update_xaxes(title_text="q = Q / M", row=1, col=1)
    fig.update_xaxes(title_text="q = Q / M", row=1, col=2)
    fig.update_yaxes(title_text="r_ISCO / M", row=1, col=1)
    fig.update_yaxes(title_text="eta (%)", row=1, col=2)

    return fig


# ---------------------------------------------------------------------------
# 7. Hawking emission spectrum
# ---------------------------------------------------------------------------

def hawking_spectrum_chart(spectrum_data: Any) -> go.Figure:
    """Hawking emission spectrum with and without greybody factors."""
    rows = _safe_list(spectrum_data, key="spectrum")
    if not rows:
        return _empty_figure("No Hawking spectrum data available")

    omega = [d.get("omega", 0) for d in rows]
    total = [d.get("dN_dt_domega", d.get("dn_dt_domega_total", 0)) for d in rows]
    bb = [d.get("dN_dt_domega_blackbody", d.get("dn_dt_domega_blackbody", 0)) for d in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=omega, y=total,
        mode="lines", name="With greybody",
        line=dict(color=COLOR_GREEN, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=omega, y=bb,
        mode="lines", name="Blackbody",
        line=dict(color=COLOR_PURPLE, width=2, dash="dash"),
    ))

    layout = _base_layout(
        "Hawking Emission Spectrum",
        xaxis_title="omega (geometrized)",
        yaxis_title="dN / dt / domega",
    )
    layout["yaxis"]["type"] = "log"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 8. Greybody scan
# ---------------------------------------------------------------------------

def greybody_scan_chart(scan_data: Any) -> go.Figure:
    """Greybody factor at peak vs charge ratio for l=2."""
    rows = _safe_list(scan_data, key="scan")
    if not rows:
        # Fallback: try the data directly as a list
        rows = _safe_list(scan_data)
    if not rows:
        return _empty_figure("No greybody data available")

    q = [d.get("q", 0) for d in rows]
    gamma = [d.get("Gamma_at_barrier", d.get("Gamma_peak", d.get("gamma_peak", 0))) for d in rows]

    gamma_thermal = [d.get("Gamma_at_thermal", None) for d in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=q, y=gamma,
        mode="lines",
        name="Gamma at barrier (l=2)",
        line=dict(color=COLOR_GM_BLUE, width=2),
    ))

    if any(v is not None and v > 0 for v in gamma_thermal):
        fig.add_trace(go.Scatter(
            x=q,
            y=[v if v is not None else float("nan") for v in gamma_thermal],
            mode="lines",
            name="Gamma at thermal peak (l=2)",
            line=dict(color=COLOR_PURPLE, width=2, dash="dash"),
        ))

    # Reference line at Gamma = 0.5 (definition of barrier top)
    fig.add_hline(
        y=0.5, line_dash="dot", line_color=COLOR_AMBER, line_width=1,
        annotation_text="Gamma = 0.5",
        annotation_font=dict(color=COLOR_AMBER, size=10),
    )

    layout = _base_layout(
        "Greybody Factor vs Charge (l=2)",
        xaxis_title="q = Q / Q_ext",
        yaxis_title="Gamma",
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 9. Effects scaling (log-log)
# ---------------------------------------------------------------------------

def effects_scaling_chart(effects_data: Any) -> go.Figure:
    """Log-log plot of observable deviations (ppm) vs charge ratio."""
    rows = _safe_list(effects_data, key="effects")
    if not rows:
        # Fallback: try the data directly as a list
        rows = _safe_list(effects_data)
    if not rows:
        return _empty_figure("No effects scaling data available")

    q = [d.get("q", 0) for d in rows]

    observable_keys = [
        ("delta_T_ppm", "Temperature", COLOR_GM_BLUE),
        ("delta_shadow_ppm", "Shadow", COLOR_PURPLE),
        ("delta_isco_ppm", "ISCO", COLOR_AMBER),
        ("delta_qnm_ppm", "QNM", COLOR_GREEN),
    ]

    fig = go.Figure()
    for key, label, color in observable_keys:
        vals = [d.get(key, None) for d in rows]
        if not any(v is not None for v in vals):
            continue
        y = [abs(v) if v is not None else float("nan") for v in vals]
        fig.add_trace(go.Scatter(
            x=q, y=y,
            mode="lines", name=label,
            line=dict(color=color, width=2),
        ))

    # Detection threshold at 1 ppm
    fig.add_hline(
        y=1.0, line_dash="dot", line_color=COLOR_RED, line_width=1,
        annotation_text="1 ppm threshold",
        annotation_font=dict(color=COLOR_RED, size=10),
    )

    # Realistic BH charge band at q ~ 1e-9
    fig.add_vrect(
        x0=5e-10, x1=5e-9,
        fillcolor=COLOR_AMBER, opacity=0.12,
        line_width=0,
        annotation_text="Realistic q",
        annotation_font=dict(color=COLOR_AMBER, size=10),
        annotation_position="top left",
    )

    layout = _base_layout(
        "Observable Deviations vs Charge Ratio",
        xaxis_title="q = Q / M",
        yaxis_title="Deviation (ppm)",
    )
    layout["xaxis"]["type"] = "log"
    layout["yaxis"]["type"] = "log"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 10. Constraint summary
# ---------------------------------------------------------------------------

def constraint_summary_chart(summary_data: Any) -> go.Figure:
    """Horizontal bar chart of charge constraints by source and mechanism."""
    # Try both 'constraints' and 'rows' keys for compatibility
    rows = _safe_list(summary_data, key="constraints")
    if not rows:
        rows = _safe_list(summary_data, key="rows")
    if not rows:
        return _empty_figure("No constraint data available")

    # Filter to rows that have a numeric q_max
    filtered = []
    for d in rows:
        q = d.get("q_max")
        if q is not None and isinstance(q, (int, float)) and q > 0:
            filtered.append(d)
    if not filtered:
        return _empty_figure("No constraint data with valid q_max")

    sources = [d.get("source", "?") for d in filtered]
    q_max = [d.get("q_max", 0) for d in filtered]

    # Use log10 for display
    log_q = []
    for v in q_max:
        try:
            log_q.append(math.log10(v) if v > 0 else -25)
        except (ValueError, TypeError):
            log_q.append(-25)

    # Assign colours: blue for observations, red for physics limits
    bar_colors = []
    for d in filtered:
        source = d.get("source", "").lower()
        if any(k in source for k in ["wald", "schwinger"]):
            bar_colors.append(COLOR_RED)
        elif "cassini" in source:
            bar_colors.append(COLOR_AMBER)
        elif "dark" in source:
            bar_colors.append(COLOR_GREEN)
        else:
            bar_colors.append(COLOR_GM_BLUE)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sources,
        x=log_q,
        orientation="h",
        marker_color=bar_colors,
        text=[f"q = {v:.2e}" for v in q_max],
        textposition="auto",
        textfont=dict(color=FONT_COLOR, size=10),
    ))

    # Vertical line at q = 1 (extremal) => log10(1) = 0
    fig.add_vline(
        x=0, line_dash="dash", line_color=COLOR_RED, line_width=1,
        annotation_text="Extremal (q=1)",
        annotation_font=dict(color=COLOR_RED, size=10),
    )

    layout = _base_layout(
        "Charge Constraints by Source",
        xaxis_title="log10(q_max)",
        yaxis_title="",
    )
    fig.update_layout(**layout)
    return fig

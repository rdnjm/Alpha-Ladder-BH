"""
Black hole shadow properties for Gibbons-Maeda dilaton black holes
with the Alpha Ladder coupling a = 1/sqrt(3).

The shadow of a black hole is the dark region seen by a distant observer,
bounded by the critical curve where photons asymptotically approach the
unstable photon sphere orbit.  For the GM dilaton black hole the metric
functions differ from Reissner-Nordstrom, causing the shadow to shrink
faster with charge -- a qualitative observational signature.

The GM metric in Einstein frame:
    ds^2 = -f(r) dt^2 + f(r)^{-1} dr^2 + R(r)^2 dOmega^2

    f(r)   = (1 - r+/r)(1 - r-/r)^gamma
    R(r)^2 = r^2 (1 - r-/r)^{1-gamma}

where gamma = (1 - a^2)/(1 + a^2).  For a = 1/sqrt(3): gamma = 1/2.

IMPORTANT: The GM horizon parametrization satisfies 2M = r+ + gamma*r-.
For a^2 = 1/3, gamma = 1/2:
    disc = 1 - 8 q^2 / 9
    r+ = M (1 + sqrt(disc))
    r- = 2 M (1 - sqrt(disc))
where q = Q/Q_ext.  At q=1: r+ = r- = 4M/3.

The photon sphere radius r_ph solves d/dr[f(r)/R(r)^2] = 0.
The critical impact parameter is b_c = R(r_ph) / sqrt(f(r_ph)).
The shadow angular diameter for a distant observer is theta = 2 b_c / D.

References:
    G. W. Gibbons and K. Maeda, Nucl. Phys. B 298, 741 (1988).
    EHT Collaboration, Astrophys. J. Lett. 930, L12 (2022) [Sgr A*].
    EHT Collaboration, Astrophys. J. Lett. 875, L1 (2019) [M87*].

Pure Python -- only 'import math' is used.
"""

import math

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G = 6.674298e-11        # m^3 kg^-1 s^-2  (Alpha Ladder predicted value)
c = 2.99792458e8        # m/s
M_sun = 1.989e30        # kg
pc = 3.0857e16          # meters per parsec

# Default dilaton coupling from Alpha Ladder S^2 KK reduction
_A_DEFAULT = 1.0 / math.sqrt(3.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_a():
    """Return default dilaton coupling a = 1/sqrt(3)."""
    return _A_DEFAULT


def _gamma(a):
    """Metric exponent gamma = (1 - a^2)/(1 + a^2)."""
    a_sq = a * a
    return (1.0 - a_sq) / (1.0 + a_sq)


def _gm_f(r, r_plus, r_minus, gam):
    """GM metric function f(r) = (1 - r+/r)(1 - r-/r)^gamma."""
    if r <= 0.0:
        return 0.0
    t1 = 1.0 - r_plus / r
    t2 = 1.0 - r_minus / r
    if t2 <= 0.0:
        return 0.0
    return t1 * (t2 ** gam)


def _gm_R_sq(r, r_minus, gam):
    """GM areal radius squared R(r)^2 = r^2 (1 - r-/r)^{1-gamma}."""
    if r <= 0.0:
        return 0.0
    t = 1.0 - r_minus / r
    if t <= 0.0:
        return 0.0
    return r * r * (t ** (1.0 - gam))


def _potential(r, r_plus, r_minus, gam):
    """
    Effective potential for photon orbits: V(r) = f(r) / R(r)^2.
    The photon sphere is at dV/dr = 0 (maximum of V).
    """
    R2 = _gm_R_sq(r, r_minus, gam)
    if R2 <= 0.0:
        return 0.0
    f = _gm_f(r, r_plus, r_minus, gam)
    return f / R2


def _dpotential_dr(r, r_plus, r_minus, gam, h):
    """Numerical derivative dV/dr by central differences."""
    vp = _potential(r + h, r_plus, r_minus, gam)
    vm = _potential(r - h, r_plus, r_minus, gam)
    return (vp - vm) / (2.0 * h)


# --- RN metric helpers (for comparison) ---

def _rn_f(r, r_plus, r_minus):
    """RN metric function f(r) = (1 - r+/r)(1 - r-/r).  gamma=1, R=r."""
    if r <= 0.0:
        return 0.0
    return (1.0 - r_plus / r) * (1.0 - r_minus / r)


def _rn_potential(r, r_plus, r_minus):
    """RN photon potential V(r) = f(r)/r^2."""
    if r <= 0.0:
        return 0.0
    f = _rn_f(r, r_plus, r_minus)
    return f / (r * r)


def _rn_dpotential_dr(r, r_plus, r_minus, h):
    """Numerical derivative of RN potential."""
    vp = _rn_potential(r + h, r_plus, r_minus)
    vm = _rn_potential(r - h, r_plus, r_minus)
    return (vp - vm) / (2.0 * h)


def _rn_photon_sphere(M, q):
    """
    Photon sphere for Reissner-Nordstrom (a=0).

    RN horizons: r+/- = M +/- sqrt(M^2 - Q^2), with Q_ext = M.
    So Q = q*M, r+ = M(1+sqrt(1-q^2)), r- = M(1-sqrt(1-q^2)).
    Metric: f(r) = 1 - 2M/r + Q^2/r^2 = (1-r+/r)(1-r-/r), R(r)=r.

    Returns dict with r_ph, b_c, etc.
    """
    q_eff = min(q, 1.0)
    disc = max(1.0 - q_eff * q_eff, 0.0)
    sqrt_disc = math.sqrt(disc)
    r_plus = M * (1.0 + sqrt_disc)
    r_minus = M * (1.0 - sqrt_disc)

    h = 1.0e-8 * M

    r_lo = r_plus * 1.001
    r_hi = 10.0 * M

    dv_lo = _rn_dpotential_dr(r_lo, r_plus, r_minus, h)
    dv_hi = _rn_dpotential_dr(r_hi, r_plus, r_minus, h)

    if dv_lo * dv_hi > 0:
        r_hi = 20.0 * M
        dv_hi = _rn_dpotential_dr(r_hi, r_plus, r_minus, h)

    for _ in range(200):
        r_mid = 0.5 * (r_lo + r_hi)
        dv_mid = _rn_dpotential_dr(r_mid, r_plus, r_minus, h)
        if abs(dv_mid) < 1.0e-15 / (M * M * M):
            break
        if dv_lo * dv_mid < 0:
            r_hi = r_mid
        else:
            r_lo = r_mid
            dv_lo = dv_mid

    r_ph = 0.5 * (r_lo + r_hi)
    f_ph = _rn_f(r_ph, r_plus, r_minus)
    b_c = r_ph / math.sqrt(f_ph) if f_ph > 0 else float("inf")

    b_c_schwarz = 3.0 * math.sqrt(3.0) * M
    delta_b = (b_c / b_c_schwarz - 1.0) * 100.0

    return {
        "r_ph":             r_ph,
        "r_ph_over_M":      r_ph / M if M > 0 else None,
        "b_c":              b_c,
        "b_c_over_M":       b_c / M if M > 0 else None,
        "delta_b_percent":  delta_b,
        "r_plus":           r_plus,
        "r_minus":          r_minus,
    }


# ---------------------------------------------------------------------------
# 1. GM horizons
# ---------------------------------------------------------------------------

def gm_horizons(M, q, a=None):
    """
    Compute outer and inner horizon radii for a Gibbons-Maeda black hole.

    Parameters
    ----------
    M : float
        Mass in geometrized units (meters).
    q : float
        Charge ratio Q/Q_ext in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with r_plus, r_minus, r_plus_rn, r_minus_rn, gamma, extremal flag.
    """
    if a is None:
        a = _default_a()

    a_sq = a * a
    gam = _gamma(a)
    extremal = (q >= 1.0)

    q_eff = min(q, 1.0)

    # GM horizons: 2M = r+ + gamma*r-, disc = 1 - 8q^2/9
    disc_gm = max(1.0 - 8.0 * q_eff * q_eff / 9.0, 0.0)
    sqrt_disc_gm = math.sqrt(disc_gm)
    r_plus = M * (1.0 + sqrt_disc_gm)
    r_minus = 2.0 * M * (1.0 - sqrt_disc_gm)

    # RN horizons for comparison (a = 0 limit)
    disc_rn = max(1.0 - q_eff * q_eff, 0.0)
    sqrt_disc_rn = math.sqrt(disc_rn)
    r_plus_rn = M * (1.0 + sqrt_disc_rn)
    r_minus_rn = M * (1.0 - sqrt_disc_rn)

    return {
        "r_plus":       r_plus,
        "r_minus":      r_minus,
        "r_plus_rn":    r_plus_rn,
        "r_minus_rn":   r_minus_rn,
        "a":            a,
        "gamma":        gam,
        "q":            q,
        "M":            M,
        "extremal":     extremal,
    }


# ---------------------------------------------------------------------------
# 2. Photon sphere
# ---------------------------------------------------------------------------

def photon_sphere(M, q, a=None):
    """
    Find the photon sphere radius by bisection on dV/dr = 0.

    The photon sphere is the unstable circular orbit for photons,
    located where d/dr[f(r)/R(r)^2] = 0 outside the outer horizon.

    Parameters
    ----------
    M : float
        Mass in geometrized units (meters).
    q : float
        Charge ratio Q/Q_ext in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with r_ph, b_c, Schwarzschild comparisons, delta_b_percent.
    """
    if a is None:
        a = _default_a()

    hz = gm_horizons(M, q, a)
    r_plus = hz["r_plus"]
    r_minus = hz["r_minus"]
    gam = hz["gamma"]

    h = 1.0e-8 * M

    # Search interval: just outside r+ to 10*M
    r_lo = r_plus * 1.001
    r_hi = 10.0 * M

    # Bisection: find where dV/dr changes sign (from + to -)
    # V has a maximum at the photon sphere
    dv_lo = _dpotential_dr(r_lo, r_plus, r_minus, gam, h)
    dv_hi = _dpotential_dr(r_hi, r_plus, r_minus, gam, h)

    # If same sign, expand search
    if dv_lo * dv_hi > 0:
        # Try wider range
        r_hi = 20.0 * M
        dv_hi = _dpotential_dr(r_hi, r_plus, r_minus, gam, h)

    # Bisection
    for _ in range(200):
        r_mid = 0.5 * (r_lo + r_hi)
        dv_mid = _dpotential_dr(r_mid, r_plus, r_minus, gam, h)
        if abs(dv_mid) < 1.0e-15 / (M * M * M):
            break
        if dv_lo * dv_mid < 0:
            r_hi = r_mid
            dv_hi = dv_mid
        else:
            r_lo = r_mid
            dv_lo = dv_mid

    r_ph = 0.5 * (r_lo + r_hi)

    # Critical impact parameter: b_c = R(r_ph) / sqrt(f(r_ph))
    f_ph = _gm_f(r_ph, r_plus, r_minus, gam)
    R2_ph = _gm_R_sq(r_ph, r_minus, gam)
    R_ph = math.sqrt(R2_ph)

    if f_ph > 0.0:
        b_c = R_ph / math.sqrt(f_ph)
    else:
        b_c = float("inf")

    # Schwarzschild values
    r_ph_schwarz = 3.0 * M
    b_c_schwarz = 3.0 * math.sqrt(3.0) * M

    delta_b_percent = (b_c / b_c_schwarz - 1.0) * 100.0

    return {
        "r_ph":             r_ph,
        "r_ph_over_M":      r_ph / M if M > 0 else None,
        "b_c":              b_c,
        "b_c_over_M":       b_c / M if M > 0 else None,
        "f_at_rph":         f_ph,
        "R_at_rph":         R_ph,
        "r_ph_schwarz":     r_ph_schwarz,
        "b_c_schwarz":      b_c_schwarz,
        "delta_b_percent":  delta_b_percent,
    }


# ---------------------------------------------------------------------------
# 3. Shadow angular size
# ---------------------------------------------------------------------------

def shadow_angular_size(M_solar, D_Mpc, q, a=None, D_kpc=None):
    """
    Compute shadow angular diameter in microarcseconds.

    Parameters
    ----------
    M_solar : float
        Black hole mass in solar masses.
    D_Mpc : float
        Distance in megaparsecs (ignored if D_kpc is given).
    q : float
        Charge ratio Q/Q_ext.
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).
    D_kpc : float or None
        Distance in kiloparsecs (overrides D_Mpc if provided).

    Returns
    -------
    dict with theta_uas, theta_uas_schwarz, delta_theta_percent, etc.
    """
    if a is None:
        a = _default_a()

    # Geometrized mass in meters
    M_geom = G * M_solar * M_sun / (c * c)

    # Distance in meters
    if D_kpc is not None:
        D_m = D_kpc * 1.0e3 * pc
    else:
        D_m = D_Mpc * 1.0e6 * pc

    # Photon sphere and critical impact parameter
    ps = photon_sphere(M_geom, q, a)
    b_c_over_M = ps["b_c_over_M"]
    b_c_schwarz_over_M = ps["b_c_schwarz"] / M_geom

    # Shadow angular diameter: theta = 2 * b_c / D  (in radians, using b_c in meters)
    # b_c (meters) = b_c_over_M * M_geom
    b_c_m = b_c_over_M * M_geom
    theta_rad = 2.0 * b_c_m / D_m

    # Convert to microarcseconds
    # 1 rad = 180/pi degrees = 180*3600/pi arcseconds = 180*3600*1e6/pi uas
    rad_to_uas = 1.0e6 * 180.0 * 3600.0 / math.pi
    theta_uas = theta_rad * rad_to_uas

    # Schwarzschild shadow
    b_c_schwarz_m = b_c_schwarz_over_M * M_geom
    theta_schwarz_rad = 2.0 * b_c_schwarz_m / D_m
    theta_schwarz_uas = theta_schwarz_rad * rad_to_uas

    delta_theta_percent = (theta_uas / theta_schwarz_uas - 1.0) * 100.0

    return {
        "theta_uas":            theta_uas,
        "theta_uas_schwarz":    theta_schwarz_uas,
        "delta_theta_percent":  delta_theta_percent,
        "b_c_over_M":          b_c_over_M,
        "r_ph_over_M":         ps["r_ph_over_M"],
        "M_solar":             M_solar,
        "D_Mpc":               D_Mpc,
        "D_kpc":               D_kpc,
        "q":                   q,
    }


# ---------------------------------------------------------------------------
# 4. Shadow scan over charge
# ---------------------------------------------------------------------------

def shadow_scan(M_solar, D_Mpc, q_values=None, a=None, D_kpc=None):
    """
    Scan shadow size over a range of charge ratios.

    Parameters
    ----------
    M_solar : float
        Black hole mass in solar masses.
    D_Mpc : float
        Distance in megaparsecs.
    q_values : list of float or None
        Charge ratios to scan (default: 20 points from 0 to 0.95).
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).
    D_kpc : float or None
        Distance in kiloparsecs (overrides D_Mpc if provided).

    Returns
    -------
    dict with 'scan' (list of dicts) and 'theta_schwarz_uas'.
    """
    if a is None:
        a = _default_a()
    if q_values is None:
        q_values = [i * 0.05 for i in range(20)]

    results = []
    theta_schwarz = None
    for q in q_values:
        s = shadow_angular_size(M_solar, D_Mpc, q, a, D_kpc=D_kpc)
        if theta_schwarz is None:
            theta_schwarz = s["theta_uas_schwarz"]
        results.append({
            "q":                    q,
            "r_ph_over_M":         s["r_ph_over_M"],
            "b_c_over_M":          s["b_c_over_M"],
            "theta_uas":           s["theta_uas"],
            "delta_theta_percent": s["delta_theta_percent"],
        })

    return {
        "scan":             results,
        "theta_schwarz_uas": theta_schwarz,
        "M_solar":          M_solar,
        "D_Mpc":            D_Mpc,
        "D_kpc":            D_kpc,
        "a":                a,
    }


# ---------------------------------------------------------------------------
# 5. EHT constraints
# ---------------------------------------------------------------------------

def eht_constraints(a=None):
    """
    Compare GM shadow predictions against EHT observations of Sgr A* and M87*.

    For each source, find the maximum charge ratio q consistent with the
    observed shadow size at 1-sigma and 2-sigma confidence.

    Returns
    -------
    dict with 'sgra' and 'm87' sub-dicts containing constraints.
    """
    if a is None:
        a = _default_a()

    sources = {
        "sgra": {
            "name":           "Sgr A*",
            "M_solar":        4.0e6,
            "D_kpc":          8.127,
            "theta_observed":  48.7,
            "sigma":           7.0,
            "ref":            "EHT 2022, ApJL 930, L12",
        },
        "m87": {
            "name":           "M87*",
            "M_solar":        6.5e9,
            "D_Mpc":          16.8,
            "theta_observed":  42.0,
            "sigma":           3.0,
            "ref":            "EHT 2019, ApJL 875, L1",
        },
    }

    results = {}
    for key, src in sources.items():
        M_solar = src["M_solar"]
        D_kpc = src.get("D_kpc", None)
        D_Mpc = src.get("D_Mpc", None)
        if D_kpc is not None:
            D_Mpc_eff = D_kpc / 1000.0
        else:
            D_Mpc_eff = D_Mpc

        theta_obs = src["theta_observed"]
        sigma = src["sigma"]

        # Schwarzschild shadow
        s0 = shadow_angular_size(M_solar, D_Mpc_eff, 0.0, a,
                                  D_kpc=D_kpc)
        theta_schwarz = s0["theta_uas_schwarz"]

        # Find q_max at 1-sigma and 2-sigma lower bounds
        # Shadow shrinks with q, so find q where theta = theta_obs - n*sigma
        q_max_1sigma = _find_q_max(M_solar, D_Mpc_eff, theta_obs - sigma,
                                    a, D_kpc=D_kpc)
        q_max_2sigma = _find_q_max(M_solar, D_Mpc_eff, theta_obs - 2.0 * sigma,
                                    a, D_kpc=D_kpc)

        constraint_1 = (
            f"EHT constrains |q| < {q_max_1sigma:.2f} at 1-sigma "
            f"for {src['name']}"
        )
        constraint_2 = (
            f"EHT constrains |q| < {q_max_2sigma:.2f} at 2-sigma "
            f"for {src['name']}"
        )

        results[key] = {
            "name":              src["name"],
            "M_solar":           M_solar,
            "D_kpc":             D_kpc,
            "D_Mpc":             D_Mpc,
            "theta_schwarz_uas": theta_schwarz,
            "theta_observed":    theta_obs,
            "sigma":             sigma,
            "q_max_1sigma":      q_max_1sigma,
            "q_max_2sigma":      q_max_2sigma,
            "constraint_1sigma": constraint_1,
            "constraint_2sigma": constraint_2,
            "ref":               src["ref"],
        }

    return results


def _find_q_max(M_solar, D_Mpc, theta_target, a, D_kpc=None):
    """
    Find maximum q such that the GM shadow angular size >= theta_target.

    Uses bisection on q in [0, 0.999].
    """
    # Shadow at q=0 (Schwarzschild-like)
    s0 = shadow_angular_size(M_solar, D_Mpc, 0.0, a, D_kpc=D_kpc)
    theta_0 = s0["theta_uas"]

    # If even q=0 shadow is below target, return 0
    if theta_0 < theta_target:
        return 0.0

    # Check if q=0.999 shadow is still above target
    s_high = shadow_angular_size(M_solar, D_Mpc, 0.999, a, D_kpc=D_kpc)
    if s_high["theta_uas"] >= theta_target:
        return 0.999

    # Bisection
    q_lo = 0.0
    q_hi = 0.999
    for _ in range(100):
        q_mid = 0.5 * (q_lo + q_hi)
        s_mid = shadow_angular_size(M_solar, D_Mpc, q_mid, a, D_kpc=D_kpc)
        if s_mid["theta_uas"] >= theta_target:
            q_lo = q_mid
        else:
            q_hi = q_mid
        if q_hi - q_lo < 1.0e-6:
            break

    return 0.5 * (q_lo + q_hi)


# ---------------------------------------------------------------------------
# 6. Compare RN vs GM shadows
# ---------------------------------------------------------------------------

def compare_rn_vs_gm(q_values=None, a=None):
    """
    Compare shadow sizes for GM (a=1/sqrt(3)) and RN (a=0) black holes.

    The GM dilaton modifies R(r)^2, causing the shadow to shrink faster
    with charge than in the RN case.

    Parameters
    ----------
    q_values : list of float or None
        Charge ratios to compare (default: 0.0 to 0.95 in steps of 0.05).
    a : float or None
        GM dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    list of dicts with q, b_c_gm, b_c_rn, delta_gm_percent, delta_rn_percent,
    gm_vs_rn_ratio.
    """
    if a is None:
        a = _default_a()
    if q_values is None:
        q_values = [i * 0.05 for i in range(20)]

    M = 1.0  # work in units of M

    # Schwarzschild reference
    b_c_schwarz = 3.0 * math.sqrt(3.0) * M

    results = []
    for q in q_values:
        # GM shadow (dilaton metric with GM r_minus)
        ps_gm = photon_sphere(M, q, a)
        b_c_gm = ps_gm["b_c"]

        # RN shadow (standard RN metric with RN horizons)
        ps_rn = _rn_photon_sphere(M, q)
        b_c_rn = ps_rn["b_c"]

        delta_gm = (b_c_gm / b_c_schwarz - 1.0) * 100.0
        delta_rn = (b_c_rn / b_c_schwarz - 1.0) * 100.0

        ratio = b_c_gm / b_c_rn if b_c_rn > 0 else None

        results.append({
            "q":                q,
            "b_c_gm":          b_c_gm,
            "b_c_rn":          b_c_rn,
            "delta_gm_percent": delta_gm,
            "delta_rn_percent": delta_rn,
            "gm_vs_rn_ratio":  ratio,
        })

    return results


# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------

def summarize_shadow_analysis():
    """
    Run all shadow analyses and return a comprehensive summary.

    Returns
    -------
    dict
    """
    a = _default_a()
    a_sq = a * a
    gam = _gamma(a)

    # Shadow scan for Sgr A*
    scan_sgra = shadow_scan(4.0e6, None, a=a, D_kpc=8.127)

    # Shadow scan for M87*
    scan_m87 = shadow_scan(6.5e9, 16.8, a=a)

    # EHT constraints
    eht = eht_constraints(a)

    # GM vs RN comparison
    comparison = compare_rn_vs_gm(a=a)

    # Key numbers at q = 0.5
    ps_half = photon_sphere(1.0, 0.5, a)
    ps_half_rn = _rn_photon_sphere(1.0, 0.5)

    return {
        "framework":  "Alpha Ladder GM dilaton BH shadow analysis",
        "coupling": {
            "a":        a,
            "a_sq":     a_sq,
            "gamma":    gam,
        },
        "scan_sgra":    scan_sgra,
        "scan_m87":     scan_m87,
        "eht":          eht,
        "comparison":   comparison,
        "key_numbers": {
            "q_0.5_b_c_over_M_gm":   ps_half["b_c_over_M"],
            "q_0.5_b_c_over_M_rn":   ps_half_rn["b_c_over_M"],
            "q_0.5_delta_gm_pct":    ps_half["delta_b_percent"],
            "q_0.5_delta_rn_pct":    ps_half_rn["delta_b_percent"],
        },
        "physical_caveats": [
            "Astrophysical black holes are expected to be nearly neutral "
            "(q ~ 0), so dilaton shadow deviations are tiny for real sources.",
            "The GM metric assumes a massless dilaton.  If the dilaton has "
            "Planck-scale mass (from flux stabilization), it decouples and "
            "BH shadows revert to GR predictions.",
            "EHT shadow measurements have large uncertainties (7-15%), so "
            "current constraints on q are weak.",
            "The shadow size depends on both mass and charge; degeneracies "
            "with mass uncertainty limit the constraining power.",
            "For a = 1/sqrt(3) the shadow shrinks ~34% faster with charge "
            "than for RN (a=0), providing a distinctive signature if "
            "precision improves.",
        ],
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(x, unit=""):
    """Format a number in scientific notation with 4 significant figures."""
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    if x == 0.0:
        return f"0.0 {unit}".strip()
    if abs(x) == float("inf"):
        return f"inf {unit}".strip()
    return f"{x:.4e} {unit}".strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 76)
    print("BLACK HOLE SHADOWS FOR GIBBONS-MAEDA DILATON BLACK HOLES")
    print("Alpha Ladder coupling a = 1/sqrt(3),  gamma = 1/2")
    print("=" * 76)

    a = _default_a()
    a_sq = a * a
    gam = _gamma(a)

    # --- Physical constants ---
    print("\n--- Physical Constants ---")
    print(f"  G             = {G:.6e}  m^3 kg^-1 s^-2  (Alpha Ladder)")
    print(f"  c             = {c:.8e}  m/s")
    print(f"  M_sun         = {M_sun:.3e}  kg")
    print(f"  1 pc          = {pc:.4e}  m")
    print(f"  a             = 1/sqrt(3) = {a:.6f}")
    print(f"  a^2           = 1/3 = {a_sq:.6f}")
    print(f"  gamma         = (1-a^2)/(1+a^2) = {gam:.4f}")

    # --- 1. Horizon structure ---
    print("\n--- 1. Horizon Structure (M = 1 geometrized) ---")
    print(f"  {'q':>6s}  {'r+/M':>8s}  {'r-/M':>10s}  "
          f"{'r+_RN/M':>8s}  {'r-_RN/M':>10s}  {'r-_GM/r-_RN':>12s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  "
          f"{'-'*8}  {'-'*10}  {'-'*12}")
    for q in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]:
        hz = gm_horizons(1.0, q, a)
        ratio_str = "N/A"
        if hz["r_minus_rn"] > 1.0e-15:
            ratio_str = f"{hz['r_minus'] / hz['r_minus_rn']:.6f}"
        print(f"  {q:6.2f}  {hz['r_plus']:8.4f}  {hz['r_minus']:10.6f}  "
              f"{hz['r_plus_rn']:8.4f}  {hz['r_minus_rn']:10.6f}  "
              f"{ratio_str:>12s}")
    print(f"\n  GM horizons use disc = 1 - 8q^2/9, satisfying 2M = r+ + r-/2.")
    print(f"  At extremality: r+_GM = r-_GM = 4M/3 = {4.0/3.0:.4f} M "
          f"(vs r-_RN = M).")

    # --- 2. Photon sphere ---
    print("\n--- 2. Photon Sphere and Critical Impact Parameter ---")
    print(f"  Schwarzschild: r_ph = 3M,  b_c = 3 sqrt(3) M = {3*math.sqrt(3):.6f} M")
    print()
    print(f"  {'q':>6s}  {'r_ph/M':>8s}  {'b_c/M':>10s}  "
          f"{'delta_b %':>10s}  {'f(r_ph)':>10s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}")
    for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        ps = photon_sphere(1.0, q, a)
        print(f"  {q:6.2f}  {ps['r_ph_over_M']:8.4f}  {ps['b_c_over_M']:10.6f}  "
              f"{ps['delta_b_percent']:10.4f}  {ps['f_at_rph']:10.6f}")

    # --- 3. Shadow angular sizes ---
    print("\n--- 3. Shadow Angular Size ---")

    # Sgr A*
    print("\n  Sgr A*: M = 4.0e6 M_sun, D = 8.127 kpc")
    scan_sgra = shadow_scan(4.0e6, None, a=a, D_kpc=8.127)
    print(f"  Schwarzschild shadow = {scan_sgra['theta_schwarz_uas']:.2f} uas")
    print(f"  EHT observed = 48.7 +/- 7.0 uas")
    print()
    print(f"  {'q':>6s}  {'r_ph/M':>8s}  {'b_c/M':>10s}  "
          f"{'theta (uas)':>12s}  {'delta %':>10s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  "
          f"{'-'*12}  {'-'*10}")
    for row in scan_sgra["scan"]:
        print(f"  {row['q']:6.2f}  {row['r_ph_over_M']:8.4f}  "
              f"{row['b_c_over_M']:10.6f}  "
              f"{row['theta_uas']:12.2f}  {row['delta_theta_percent']:10.4f}")

    # M87*
    print(f"\n  M87*: M = 6.5e9 M_sun, D = 16.8 Mpc")
    scan_m87 = shadow_scan(6.5e9, 16.8, a=a)
    print(f"  Schwarzschild shadow = {scan_m87['theta_schwarz_uas']:.2f} uas")
    print(f"  EHT observed = 42.0 +/- 3.0 uas")
    print()
    print(f"  {'q':>6s}  {'r_ph/M':>8s}  {'b_c/M':>10s}  "
          f"{'theta (uas)':>12s}  {'delta %':>10s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  "
          f"{'-'*12}  {'-'*10}")
    for row in scan_m87["scan"]:
        print(f"  {row['q']:6.2f}  {row['r_ph_over_M']:8.4f}  "
              f"{row['b_c_over_M']:10.6f}  "
              f"{row['theta_uas']:12.2f}  {row['delta_theta_percent']:10.4f}")

    # --- 4. GM vs RN comparison ---
    print("\n--- 4. GM vs RN Shadow Comparison ---")
    print(f"  GM coupling a = {a:.6f},  RN has a = 0")
    print(f"  GM shadows shrink faster with charge because the dilaton "
          f"modifies R(r)^2.")
    print()
    comp = compare_rn_vs_gm(a=a)
    print(f"  {'q':>6s}  {'b_c_GM/M':>10s}  {'b_c_RN/M':>10s}  "
          f"{'delta_GM %':>10s}  {'delta_RN %':>10s}  {'GM/RN':>8s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*8}")
    for row in comp:
        gm_rn = row["gm_vs_rn_ratio"]
        gm_rn_s = f"{gm_rn:.6f}" if gm_rn is not None else "N/A"
        print(f"  {row['q']:6.2f}  {row['b_c_gm']:10.6f}  "
              f"{row['b_c_rn']:10.6f}  "
              f"{row['delta_gm_percent']:10.4f}  "
              f"{row['delta_rn_percent']:10.4f}  {gm_rn_s:>8s}")

    # Highlight q = 0.5
    row_05 = None
    for row in comp:
        if abs(row["q"] - 0.5) < 0.01:
            row_05 = row
            break
    if row_05:
        print(f"\n  At q = 0.5:")
        print(f"    GM shadow: {row_05['delta_gm_percent']:.2f}% vs Schwarzschild")
        print(f"    RN shadow: {row_05['delta_rn_percent']:.2f}% vs Schwarzschild")
        print(f"    GM/RN = {row_05['gm_vs_rn_ratio']:.6f}")
        print(f"    Dilaton amplifies shadow shrinkage by ~34% vs RN.")

    # --- 5. EHT constraints ---
    print("\n--- 5. EHT Constraints ---")
    eht = eht_constraints(a)
    for key in ["sgra", "m87"]:
        e = eht[key]
        print(f"\n  {e['name']}:")
        print(f"    Schwarzschild shadow  = {e['theta_schwarz_uas']:.2f} uas")
        print(f"    Observed shadow       = {e['theta_observed']:.1f} +/- "
              f"{e['sigma']:.1f} uas")
        print(f"    {e['constraint_1sigma']}")
        print(f"    {e['constraint_2sigma']}")
        print(f"    Ref: {e['ref']}")

    # --- 6. Summary ---
    print("\n--- 6. Summary and Honest Assessment ---")
    print()
    print("  The Gibbons-Maeda dilaton black hole with the Alpha Ladder")
    print(f"  coupling a = 1/sqrt(3) (gamma = {gam:.4f}) predicts shadows")
    print("  that shrink with charge ratio q = Q/Q_ext.  The key findings:")
    print()
    print("  1. At q = 0 (neutral): shadow is identical to Schwarzschild (GR).")
    print("     No deviation whatsoever for uncharged black holes.")
    print()
    if row_05:
        print(f"  2. At q = 0.5: shadow shrinks by {abs(row_05['delta_gm_percent']):.1f}% "
              f"(GM) vs {abs(row_05['delta_rn_percent']):.1f}% (RN).")
        print("     The dilaton amplifies the shadow shrinkage by ~34%.")
    print()
    print("  3. EHT constraints:")
    for key in ["sgra", "m87"]:
        e = eht[key]
        print(f"     {e['name']}: q < {e['q_max_1sigma']:.2f} (1-sigma), "
              f"q < {e['q_max_2sigma']:.2f} (2-sigma)")
    print()
    print("  HONEST CAVEATS:")
    print()
    print("  - Astrophysical black holes are expected to be electrically neutral")
    print("    to extremely high precision.  Selective charge accretion and pair")
    print("    production discharge any macroscopic charge on timescales much")
    print("    shorter than the BH age.  Therefore q ~ 0 for all observed BHs,")
    print("    and the dilaton shadow deviation is unobservably small in practice.")
    print()
    print("  - If the dilaton acquires a Planck-scale mass (as predicted by the")
    print("    Alpha Ladder flux stabilization), it decouples at astrophysical")
    print("    scales.  In that case BH solutions revert exactly to GR and the")
    print("    entire GM shadow analysis becomes academic.")
    print()
    print("  - Current EHT uncertainties (7-15%) are far too large to distinguish")
    print("    GM from RN or GR, even at moderate q.  Next-generation EHT (ngEHT)")
    print("    may reach ~1% precision, which would begin to probe q ~ 0.3-0.5")
    print("    if charge were somehow sustained.")
    print()
    print("  - The value of this analysis is theoretical: it maps out the")
    print("    observable consequences of the Alpha Ladder's specific dilaton")
    print("    coupling and provides concrete, falsifiable predictions should")
    print("    charged BH solutions prove relevant in other contexts (e.g.,")
    print("    primordial BHs, dark sector charges, or magnetic monopoles).")

    print("\n" + "=" * 76)
    print("Done.")

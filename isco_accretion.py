"""
Innermost stable circular orbit (ISCO) and accretion efficiency for
Gibbons-Maeda dilaton black holes with the Alpha Ladder coupling
a = 1/sqrt(3).

For a test particle orbiting in the equatorial plane of a static,
spherically symmetric spacetime

    ds^2 = -f(r) dt^2 + f(r)^{-1} dr^2 + R(r)^2 dOmega^2

the conserved specific energy E = f dt/dtau and angular momentum
L = R^2 dphi/dtau yield the radial equation

    (dr/dtau)^2 = E^2 - V_eff(r, L)

where V_eff(r, L) = f(r) (1 + L^2 / R(r)^2).

Circular orbits require dV_eff/dr = 0 (determines L^2 as a function
of r), and the ISCO additionally requires d^2V_eff/dr^2 = 0 (marginal
stability).  The accretion efficiency is eta = 1 - E_ISCO, the fraction
of rest-mass energy radiated as matter spirals from infinity to the ISCO.

The GM metric:
    f(r)   = (1 - r+/r)(1 - r-/r)^gamma
    R(r)^2 = r^2 (1 - r-/r)^{1-gamma}

where gamma = (1 - a^2)/(1 + a^2).  For a = 1/sqrt(3): gamma = 1/2.

IMPORTANT: The GM horizon parametrization satisfies 2M = r+ + gamma*r-.
For a^2 = 1/3, gamma = 1/2:
    disc = 1 - 8 q^2 / 9
    r+ = M (1 + sqrt(disc))
    r- = 2 M (1 - sqrt(disc))
where q = Q/Q_ext in [0, 1].  At q=1: r+ = r- = 4M/3.

References:
    G. W. Gibbons and K. Maeda, Nucl. Phys. B 298, 741 (1988).
    S. Chandrasekhar, The Mathematical Theory of Black Holes (1983).

Pure Python -- only 'import math' is used.
"""

import math

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G = 6.674298e-11        # m^3 kg^-1 s^-2  (Alpha Ladder predicted value)
c = 2.99792458e8        # m/s
M_sun = 1.989e30        # kg
L_sun = 3.828e26        # W
year = 3.156e7          # seconds
m_p = 1.6726219e-27     # kg  (proton mass)
sigma_T = 6.6524587e-29 # m^2 (Thomson cross section)

# Default dilaton coupling from Alpha Ladder S^2 KK reduction
_A_DEFAULT = 1.0 / math.sqrt(3.0)


# ---------------------------------------------------------------------------
# Internal helpers -- metric functions
# ---------------------------------------------------------------------------

def _default_a():
    """Return default dilaton coupling a = 1/sqrt(3)."""
    return _A_DEFAULT


def _gamma(a):
    """Metric exponent gamma = (1 - a^2)/(1 + a^2)."""
    a_sq = a * a
    return (1.0 - a_sq) / (1.0 + a_sq)


def _horizons(M, q, a):
    """
    Compute GM horizon radii satisfying 2M = r+ + gamma*r-.

    For a^2=1/3, gamma=1/2:
        disc = 1 - 8q^2/9
        r+ = M(1 + sqrt(disc)),  r- = 2M(1 - sqrt(disc))

    Returns (r_plus, r_minus, gamma).
    """
    a_sq = a * a
    gam = _gamma(a)
    q_eff = min(q, 1.0)
    disc = max(1.0 - 8.0 * q_eff * q_eff / 9.0, 0.0)
    sqrt_disc = math.sqrt(disc)
    r_plus = M * (1.0 + sqrt_disc)
    r_minus = 2.0 * M * (1.0 - sqrt_disc)
    return r_plus, r_minus, gam


def _rn_horizons(M, q):
    """
    Compute Reissner-Nordstrom horizon radii (a=0 limit).

    For RN: r+/- = M(1 +/- sqrt(1 - q^2)) with Q_ext = M.
    Returns (r_plus, r_minus).
    """
    q_eff = min(q, 1.0)
    disc = max(1.0 - q_eff * q_eff, 0.0)
    sqrt_disc = math.sqrt(disc)
    r_plus = M * (1.0 + sqrt_disc)
    r_minus = M * (1.0 - sqrt_disc)
    return r_plus, r_minus


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


def _rn_f(r, r_plus, r_minus):
    """RN metric function f(r) = (1 - r+/r)(1 - r-/r).  R(r)=r."""
    if r <= 0.0:
        return 0.0
    return (1.0 - r_plus / r) * (1.0 - r_minus / r)


# ---------------------------------------------------------------------------
# Effective potential and its derivatives (numerical)
# ---------------------------------------------------------------------------

def _veff(r, L_sq, r_plus, r_minus, gam, is_rn=False):
    """
    Effective potential V_eff(r, L) = f(r) * (1 + L^2 / R(r)^2).

    For RN: f = (1-r+/r)(1-r-/r), R(r)^2 = r^2.
    For GM: f and R as above.
    """
    if is_rn:
        f = _rn_f(r, r_plus, r_minus)
        R2 = r * r
    else:
        f = _gm_f(r, r_plus, r_minus, gam)
        R2 = _gm_R_sq(r, r_minus, gam)
    if R2 <= 0.0:
        return 0.0
    return f * (1.0 + L_sq / R2)


def _dveff_dr(r, L_sq, r_plus, r_minus, gam, h, is_rn=False):
    """Numerical first derivative of V_eff by central differences."""
    vp = _veff(r + h, L_sq, r_plus, r_minus, gam, is_rn)
    vm = _veff(r - h, L_sq, r_plus, r_minus, gam, is_rn)
    return (vp - vm) / (2.0 * h)


def _d2veff_dr2(r, L_sq, r_plus, r_minus, gam, h, is_rn=False):
    """Numerical second derivative of V_eff by central differences."""
    vp = _veff(r + h, L_sq, r_plus, r_minus, gam, is_rn)
    v0 = _veff(r, L_sq, r_plus, r_minus, gam, is_rn)
    vm = _veff(r - h, L_sq, r_plus, r_minus, gam, is_rn)
    return (vp - 2.0 * v0 + vm) / (h * h)


# ---------------------------------------------------------------------------
# Circular orbit angular momentum from dV_eff/dr = 0
# ---------------------------------------------------------------------------

def _step_size(M):
    """
    Choose a finite-difference step size appropriate for the mass scale.

    For numerical derivatives of smooth metric functions, h ~ 1e-6 * M
    gives good accuracy without floating-point cancellation.
    """
    return 1.0e-6 * M


def _circular_L_sq(r, r_plus, r_minus, gam, h, is_rn=False):
    """
    Compute L^2 for a circular orbit at radius r.

    From dV_eff/dr = 0 with V_eff = f(1 + L^2/R^2):
        f'(1 + L^2/R^2) + f * L^2 * d(1/R^2)/dr = 0

    Rearranged:
        L^2 = -f' / (f' / R^2 + f * d(1/R^2)/dr)

    All derivatives computed numerically by central differences.
    """
    if is_rn:
        f = _rn_f(r, r_plus, r_minus)
        fp = (_rn_f(r + h, r_plus, r_minus) - _rn_f(r - h, r_plus, r_minus)) / (2.0 * h)
        R2 = r * r
        inv_R2 = 1.0 / R2
        inv_R2_p = 1.0 / ((r + h) * (r + h))
        inv_R2_m = 1.0 / ((r - h) * (r - h))
    else:
        f = _gm_f(r, r_plus, r_minus, gam)
        fp = (_gm_f(r + h, r_plus, r_minus, gam) - _gm_f(r - h, r_plus, r_minus, gam)) / (2.0 * h)
        R2 = _gm_R_sq(r, r_minus, gam)
        if R2 <= 0.0:
            return None
        inv_R2 = 1.0 / R2
        R2p = _gm_R_sq(r + h, r_minus, gam)
        R2m = _gm_R_sq(r - h, r_minus, gam)
        inv_R2_p = 1.0 / R2p if R2p > 0 else 0.0
        inv_R2_m = 1.0 / R2m if R2m > 0 else 0.0

    d_inv_R2 = (inv_R2_p - inv_R2_m) / (2.0 * h)

    denom = fp * inv_R2 + f * d_inv_R2

    # Use a relative threshold: if |denom| is negligibly small compared
    # to the individual terms, we cannot reliably compute L^2.
    scale = abs(fp * inv_R2) + abs(f * d_inv_R2)
    if scale == 0.0 or abs(denom) < 1.0e-12 * scale:
        return None

    L_sq = -fp / denom
    if L_sq < 0.0:
        return None
    return L_sq


# ---------------------------------------------------------------------------
# ISCO finder: scan + bisection
# ---------------------------------------------------------------------------

def _E_sq_circular(r, r_plus, r_minus, gam, h, is_rn=False):
    """
    Compute E^2 for a circular orbit at radius r.

    Returns E^2 or None if no circular orbit exists at this radius.
    The ISCO is at the minimum of this function along the family of
    circular orbits.
    """
    L_sq = _circular_L_sq(r, r_plus, r_minus, gam, h, is_rn)
    if L_sq is None:
        return None

    if is_rn:
        f = _rn_f(r, r_plus, r_minus)
        R2 = r * r
    else:
        f = _gm_f(r, r_plus, r_minus, gam)
        R2 = _gm_R_sq(r, r_minus, gam)

    if R2 <= 0.0:
        return None
    return f * (1.0 + L_sq / R2)


def _find_isco_radius(r_plus, r_minus, gam, M, is_rn=False):
    """
    Find the ISCO radius by minimizing E^2(r) along the family of
    circular orbits.

    The ISCO is the innermost stable circular orbit, which corresponds
    to the minimum of E^2(r) for r > r_photon_sphere.  Below the ISCO,
    E^2 increases (unbound orbits); above the ISCO, E^2 also increases
    toward 1 (at infinity).

    We use golden section search after bracketing the minimum via a scan.

    Returns (r_isco, L_sq_isco, E_sq_isco) or None.
    """
    h = _step_size(M)

    # Determine where circular orbits start to exist.
    # The photon sphere is at ~3M (Schwarzschild) or somewhat less.
    # We scan from just above r+ to ~20M.
    r_start = r_plus * 1.2
    r_end = 20.0 * M
    n_scan = 2000

    # Find the range where circular orbits exist and locate the E^2 minimum
    e2_vals = []
    r_vals = []
    for i in range(n_scan):
        r = r_start + (r_end - r_start) * i / (n_scan - 1)
        e2 = _E_sq_circular(r, r_plus, r_minus, gam, h, is_rn)
        if e2 is not None:
            e2_vals.append(e2)
            r_vals.append(r)

    if len(r_vals) < 3:
        return None

    # Find the minimum E^2 in the scan
    min_idx = 0
    for i in range(1, len(e2_vals)):
        if e2_vals[i] < e2_vals[min_idx]:
            min_idx = i

    # Check that we found a proper interior minimum (not at an edge)
    # The ISCO minimum should have E^2 rising on both sides.
    # If min is at edge, the ISCO may be outside our range.
    if min_idx == 0 or min_idx == len(e2_vals) - 1:
        # Extend search
        r_end = 50.0 * M
        e2_vals = []
        r_vals = []
        for i in range(n_scan):
            r = r_start + (r_end - r_start) * i / (n_scan - 1)
            e2 = _E_sq_circular(r, r_plus, r_minus, gam, h, is_rn)
            if e2 is not None:
                e2_vals.append(e2)
                r_vals.append(r)
        if len(r_vals) < 3:
            return None
        min_idx = 0
        for i in range(1, len(e2_vals)):
            if e2_vals[i] < e2_vals[min_idx]:
                min_idx = i
        if min_idx == 0 or min_idx == len(e2_vals) - 1:
            return None

    # Bracket: [r_left, r_right] containing the minimum
    r_a = r_vals[max(0, min_idx - 2)]
    r_b = r_vals[min(len(r_vals) - 1, min_idx + 2)]

    # Golden section search to refine
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
    tol = 1.0e-10 * M

    for _ in range(200):
        if (r_b - r_a) < tol:
            break
        r_c = r_b - phi * (r_b - r_a)
        r_d = r_a + phi * (r_b - r_a)
        e2_c = _E_sq_circular(r_c, r_plus, r_minus, gam, h, is_rn)
        e2_d = _E_sq_circular(r_d, r_plus, r_minus, gam, h, is_rn)
        if e2_c is None or e2_d is None:
            break
        if e2_c < e2_d:
            r_b = r_d
        else:
            r_a = r_c

    r_isco = 0.5 * (r_a + r_b)
    L_sq_isco = _circular_L_sq(r_isco, r_plus, r_minus, gam, h, is_rn)
    if L_sq_isco is None:
        return None

    if is_rn:
        R2 = r_isco * r_isco
        f_isco = _rn_f(r_isco, r_plus, r_minus)
    else:
        R2 = _gm_R_sq(r_isco, r_minus, gam)
        f_isco = _gm_f(r_isco, r_plus, r_minus, gam)

    E_sq_isco = f_isco * (1.0 + L_sq_isco / R2) if R2 > 0 else 0.0

    return r_isco, L_sq_isco, E_sq_isco


# ---------------------------------------------------------------------------
# 1. circular_orbit
# ---------------------------------------------------------------------------

def circular_orbit(r, M, q, a=None):
    """
    Compute the specific energy, angular momentum, and stability of a
    circular orbit at radius r in the GM dilaton spacetime.

    Parameters
    ----------
    r : float
        Orbital radius in geometrized units (meters).
    M : float
        Mass in geometrized units (meters).
    q : float
        Charge ratio Q/Q_ext in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with r, L_squared, E_squared, E, L, v_orbital, is_stable.
    """
    if a is None:
        a = _default_a()

    r_plus, r_minus, gam = _horizons(M, q, a)
    h = _step_size(M)

    L_sq = _circular_L_sq(r, r_plus, r_minus, gam, h, is_rn=False)
    if L_sq is None:
        return {"error": "No circular orbit at this radius", "r": r}

    f = _gm_f(r, r_plus, r_minus, gam)
    R2 = _gm_R_sq(r, r_minus, gam)
    E_sq = f * (1.0 + L_sq / R2) if R2 > 0 else 0.0

    E = math.sqrt(E_sq) if E_sq > 0 else 0.0
    L = math.sqrt(L_sq)

    # Stability: V_eff'' > 0 means stable
    d2v = _d2veff_dr2(r, L_sq, r_plus, r_minus, gam, h, is_rn=False)
    is_stable = d2v > 0

    # Orbital velocity v = L / (R * E) (coordinate velocity in the frame)
    R = math.sqrt(R2) if R2 > 0 else 0.0
    v_orbital = L / (R * E) if (R > 0 and E > 0) else 0.0

    return {
        "r":            r,
        "r_over_M":     r / M if M > 0 else None,
        "L_squared":    L_sq,
        "E_squared":    E_sq,
        "E":            E,
        "L":            L,
        "v_orbital":    v_orbital,
        "is_stable":    is_stable,
        "V_eff_pp":     d2v,
        "r_plus":       r_plus,
        "r_minus":      r_minus,
        "gamma":        gam,
    }


# ---------------------------------------------------------------------------
# 2. find_isco
# ---------------------------------------------------------------------------

def find_isco(M, q, a=None):
    """
    Find the innermost stable circular orbit (ISCO) for a GM dilaton
    black hole.

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
    dict with r_isco, r_isco_over_M, E_isco, L_isco, eta,
    is_prograde, delta_r_vs_schwarz_percent.
    """
    if a is None:
        a = _default_a()

    r_plus, r_minus, gam = _horizons(M, q, a)
    result = _find_isco_radius(r_plus, r_minus, gam, M, is_rn=False)
    if result is None:
        return {"error": "ISCO not found", "q": q}

    r_isco, L_sq_isco, E_sq_isco = result
    E_isco = math.sqrt(E_sq_isco) if E_sq_isco > 0 else 0.0
    L_isco = math.sqrt(L_sq_isco) if L_sq_isco > 0 else 0.0
    eta = 1.0 - E_isco

    # Schwarzschild ISCO at 6M
    r_isco_schwarz = 6.0 * M
    delta_r = (r_isco / r_isco_schwarz - 1.0) * 100.0

    return {
        "r_isco":                   r_isco,
        "r_isco_over_M":            r_isco / M if M > 0 else None,
        "E_isco":                   E_isco,
        "L_isco":                   L_isco,
        "eta":                      eta,
        "eta_percent":              eta * 100.0,
        "is_prograde":              True,
        "delta_r_vs_schwarz_percent": delta_r,
        "r_plus":                   r_plus,
        "r_minus":                  r_minus,
        "gamma":                    gam,
        "q":                        q,
        "a":                        a,
    }


def _find_isco_rn(M, q):
    """Find ISCO for Reissner-Nordstrom (a=0)."""
    r_plus, r_minus = _rn_horizons(M, q)
    gam = 1.0  # RN has gamma=1
    result = _find_isco_radius(r_plus, r_minus, gam, M, is_rn=True)
    if result is None:
        return None
    r_isco, L_sq_isco, E_sq_isco = result
    E_isco = math.sqrt(E_sq_isco) if E_sq_isco > 0 else 0.0
    eta = 1.0 - E_isco
    return {
        "r_isco":        r_isco,
        "r_isco_over_M": r_isco / M if M > 0 else None,
        "E_isco":        E_isco,
        "eta":           eta,
        "eta_percent":   eta * 100.0,
    }


# ---------------------------------------------------------------------------
# 3. accretion_efficiency
# ---------------------------------------------------------------------------

def accretion_efficiency(M, q, a=None):
    """
    Compute the radiative accretion efficiency eta = 1 - E_ISCO.

    This is the fraction of rest-mass energy radiated when matter spirals
    from infinity to the ISCO in a thin accretion disk.

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
    dict with eta, eta_percent, eta_schwarz, eta_rn, delta_eta_vs_schwarz.
    """
    if a is None:
        a = _default_a()

    # GM efficiency
    isco_gm = find_isco(M, q, a)
    if "error" in isco_gm:
        return isco_gm

    eta_gm = isco_gm["eta"]

    # Schwarzschild efficiency: E_isco = sqrt(8/9), eta = 1 - sqrt(8/9)
    E_schwarz = math.sqrt(8.0 / 9.0)
    eta_schwarz = 1.0 - E_schwarz

    # RN efficiency at same q
    rn_result = _find_isco_rn(M, q)
    if rn_result is not None:
        eta_rn = rn_result["eta"]
    else:
        eta_rn = None

    delta_eta = eta_gm - eta_schwarz

    return {
        "eta":                  eta_gm,
        "eta_percent":          eta_gm * 100.0,
        "E_isco":               isco_gm["E_isco"],
        "r_isco_over_M":        isco_gm["r_isco_over_M"],
        "eta_schwarz":          eta_schwarz,
        "eta_schwarz_percent":  eta_schwarz * 100.0,
        "eta_rn":               eta_rn,
        "eta_rn_percent":       eta_rn * 100.0 if eta_rn is not None else None,
        "delta_eta_vs_schwarz": delta_eta,
        "q":                    q,
        "a":                    a,
    }


# ---------------------------------------------------------------------------
# 4. isco_scan
# ---------------------------------------------------------------------------

def isco_scan(q_values=None, a=None):
    """
    Scan ISCO and accretion efficiency over a range of charge ratios.

    Parameters
    ----------
    q_values : list of float or None
        Charge ratios to scan (default: 20 points from 0 to 0.95).
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    list of dicts with q, r_isco_over_M, E_isco, eta_percent,
    eta_rn_percent, delta_vs_schwarz.
    """
    if a is None:
        a = _default_a()
    if q_values is None:
        q_values = [i * 0.05 for i in range(20)]

    M = 1.0  # work in units of M

    E_schwarz = math.sqrt(8.0 / 9.0)
    eta_schwarz = 1.0 - E_schwarz

    results = []
    for q in q_values:
        isco_gm = find_isco(M, q, a)
        rn = _find_isco_rn(M, q)

        if "error" in isco_gm:
            results.append({
                "q": q, "error": isco_gm["error"],
            })
            continue

        eta_gm = isco_gm["eta"]
        eta_rn = rn["eta"] if rn is not None else None
        delta = (eta_gm / eta_schwarz - 1.0) * 100.0

        results.append({
            "q":                q,
            "r_isco_over_M":    isco_gm["r_isco_over_M"],
            "E_isco":           isco_gm["E_isco"],
            "eta_percent":      eta_gm * 100.0,
            "eta_rn_percent":   eta_rn * 100.0 if eta_rn is not None else None,
            "r_isco_rn_over_M": rn["r_isco_over_M"] if rn is not None else None,
            "delta_vs_schwarz": delta,
        })

    return results


# ---------------------------------------------------------------------------
# 5. luminosity_comparison
# ---------------------------------------------------------------------------

def luminosity_comparison(M_solar, mdot_solar_per_year, q, a=None):
    """
    Compute accretion luminosity L = eta * mdot * c^2 in physical units.

    Parameters
    ----------
    M_solar : float
        Black hole mass in solar masses.
    mdot_solar_per_year : float
        Accretion rate in solar masses per year.
    q : float
        Charge ratio Q/Q_ext in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with L_gm, L_schwarz, L_rn, delta_L_percent, L_eddington.
    """
    if a is None:
        a = _default_a()

    # Geometrized mass
    M_geom = G * M_solar * M_sun / (c * c)

    # Accretion rate in kg/s
    mdot_kg_s = mdot_solar_per_year * M_sun / year

    # Efficiencies
    eff = accretion_efficiency(M_geom, q, a)
    if "error" in eff:
        return eff

    eta_gm = eff["eta"]
    eta_schwarz = eff["eta_schwarz"]
    eta_rn = eff["eta_rn"]

    # Luminosities
    L_gm = eta_gm * mdot_kg_s * c * c
    L_schwarz = eta_schwarz * mdot_kg_s * c * c
    L_rn = eta_rn * mdot_kg_s * c * c if eta_rn is not None else None

    delta_L = (L_gm / L_schwarz - 1.0) * 100.0 if L_schwarz > 0 else 0.0

    # Eddington luminosity: L_Edd = 4 pi G M m_p c / sigma_T
    M_kg = M_solar * M_sun
    L_edd = 4.0 * math.pi * G * M_kg * m_p * c / sigma_T

    return {
        "eta_gm":               eta_gm,
        "eta_schwarz":          eta_schwarz,
        "eta_rn":               eta_rn,
        "L_gm_W":               L_gm,
        "L_gm_Lsun":            L_gm / L_sun,
        "L_schwarz_W":          L_schwarz,
        "L_schwarz_Lsun":       L_schwarz / L_sun,
        "L_rn_W":               L_rn,
        "L_rn_Lsun":            L_rn / L_sun if L_rn is not None else None,
        "delta_L_percent":      delta_L,
        "L_eddington_W":        L_edd,
        "L_eddington_Lsun":     L_edd / L_sun,
        "mdot_solar_per_year":  mdot_solar_per_year,
        "M_solar":              M_solar,
        "q":                    q,
    }


# ---------------------------------------------------------------------------
# 6. compare_gm_rn_kerr
# ---------------------------------------------------------------------------

def compare_gm_rn_kerr(q_values=None, a=None):
    """
    Compare ISCO and efficiency for GM, RN, and Schwarzschild.

    Parameters
    ----------
    q_values : list of float or None
        Charge ratios (default: 0 to 0.95 in steps of 0.05).
    a : float or None
        GM dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    list of dicts with q, r_isco_gm, r_isco_rn, eta_gm, eta_rn,
    gm_vs_rn ratios.
    """
    if a is None:
        a = _default_a()
    if q_values is None:
        q_values = [i * 0.05 for i in range(20)]

    M = 1.0  # work in units of M

    results = []
    for q in q_values:
        isco_gm = find_isco(M, q, a)
        rn = _find_isco_rn(M, q)

        if "error" in isco_gm:
            results.append({"q": q, "error": isco_gm["error"]})
            continue

        r_gm = isco_gm["r_isco_over_M"]
        eta_gm = isco_gm["eta"]
        r_rn = rn["r_isco_over_M"] if rn is not None else None
        eta_rn = rn["eta"] if rn is not None else None

        r_ratio = r_gm / r_rn if (r_rn is not None and r_rn > 0) else None
        eta_ratio = eta_gm / eta_rn if (eta_rn is not None and eta_rn > 0) else None

        results.append({
            "q":                q,
            "r_isco_gm_over_M": r_gm,
            "r_isco_rn_over_M": r_rn,
            "eta_gm_percent":   eta_gm * 100.0,
            "eta_rn_percent":   eta_rn * 100.0 if eta_rn is not None else None,
            "r_ratio_gm_rn":   r_ratio,
            "eta_ratio_gm_rn": eta_ratio,
        })

    return results


# ---------------------------------------------------------------------------
# 7. summarize_isco_analysis
# ---------------------------------------------------------------------------

def summarize_isco_analysis():
    """
    Run all ISCO and accretion analyses and return a comprehensive summary.

    Returns
    -------
    dict
    """
    a = _default_a()
    a_sq = a * a
    gam = _gamma(a)
    M = 1.0

    # Full scan
    scan = isco_scan(a=a)

    # GM vs RN comparison
    comparison = compare_gm_rn_kerr(a=a)

    # Key numbers at q = 0.5
    isco_half = find_isco(M, 0.5, a)
    rn_half = _find_isco_rn(M, 0.5)

    # Schwarzschild reference
    E_schwarz = math.sqrt(8.0 / 9.0)
    eta_schwarz = 1.0 - E_schwarz

    # Luminosity example: 10 solar mass BH accreting at 1e-8 Msun/yr
    lum = luminosity_comparison(10.0, 1.0e-8, 0.5, a)

    return {
        "framework":    "Alpha Ladder GM dilaton BH ISCO analysis",
        "coupling": {
            "a":        a,
            "a_sq":     a_sq,
            "gamma":    gam,
        },
        "schwarzschild_reference": {
            "r_isco_over_M":    6.0,
            "E_isco":           E_schwarz,
            "eta":              eta_schwarz,
            "eta_percent":      eta_schwarz * 100.0,
        },
        "key_numbers_q05": {
            "gm_r_isco_over_M":     isco_half.get("r_isco_over_M"),
            "gm_eta_percent":       isco_half.get("eta_percent"),
            "rn_r_isco_over_M":     rn_half["r_isco_over_M"] if rn_half else None,
            "rn_eta_percent":       rn_half["eta_percent"] if rn_half else None,
        },
        "scan":         scan,
        "comparison":   comparison,
        "luminosity_example": lum,
        "physical_caveats": [
            "Astrophysical black holes are expected to be nearly neutral "
            "(q ~ 0), so the ISCO is essentially at 6M (GR) for all "
            "observed systems.",
            "If the dilaton acquires Planck-scale mass from flux "
            "stabilization, it decouples and BH orbits revert to GR.",
            "For charged BHs, the ISCO moves inward and efficiency "
            "increases, but the GM dilaton modifies the rate of change "
            "compared to RN.",
            "Real accretion disks are thick, turbulent, and magnetized. "
            "The thin-disk ISCO efficiency is an idealization.",
            "The value of this analysis is theoretical: mapping the "
            "observable consequences of the Alpha Ladder dilaton coupling.",
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
    print("ISCO AND ACCRETION EFFICIENCY FOR GIBBONS-MAEDA DILATON BLACK HOLES")
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
    print(f"  L_sun         = {L_sun:.3e}  W")
    print(f"  a             = 1/sqrt(3) = {a:.6f}")
    print(f"  a^2           = 1/3 = {a_sq:.6f}")
    print(f"  gamma         = (1-a^2)/(1+a^2) = {gam:.4f}")

    # --- Schwarzschild reference ---
    E_schwarz = math.sqrt(8.0 / 9.0)
    eta_schwarz = 1.0 - E_schwarz
    print(f"\n--- Schwarzschild Reference ---")
    print(f"  ISCO at r = 6M")
    print(f"  E_isco    = sqrt(8/9) = {E_schwarz:.6f}")
    print(f"  eta       = 1 - E_isco = {eta_schwarz:.6f} = {eta_schwarz*100:.2f}%")

    # --- 1. Circular orbit examples ---
    print("\n--- 1. Circular Orbit Properties (q = 0.5, M = 1) ---")
    M = 1.0
    q_test = 0.5
    print(f"  {'r/M':>8s}  {'E':>10s}  {'L/M':>10s}  "
          f"{'v_orb':>10s}  {'stable':>8s}  {'V_eff\"':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*8}  {'-'*12}")
    r_plus_test, _, _ = _horizons(M, q_test, a)
    for r_over_M in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0]:
        r = r_over_M * M
        if r <= r_plus_test * 1.01:
            continue
        co = circular_orbit(r, M, q_test, a)
        if "error" in co:
            print(f"  {r_over_M:8.1f}  {'---':>10s}")
            continue
        stable_s = "YES" if co["is_stable"] else "no"
        print(f"  {r_over_M:8.1f}  {co['E']:10.6f}  {co['L']:10.6f}  "
              f"{co['v_orbital']:10.6f}  {stable_s:>8s}  {co['V_eff_pp']:12.4e}")

    # --- 2. ISCO scan over charge ---
    print("\n--- 2. ISCO and Efficiency vs Charge (GM dilaton) ---")
    print(f"  {'q':>6s}  {'r_isco/M':>10s}  {'E_isco':>10s}  "
          f"{'eta %':>8s}  {'eta_RN %':>8s}  {'r_RN/M':>10s}  "
          f"{'delta %':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  "
          f"{'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

    q_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    scan = isco_scan(q_list, a)
    for row in scan:
        if "error" in row:
            print(f"  {row['q']:6.2f}  {'error':>10s}")
            continue
        eta_rn_s = f"{row['eta_rn_percent']:.2f}" if row['eta_rn_percent'] is not None else "N/A"
        r_rn_s = f"{row['r_isco_rn_over_M']:.4f}" if row['r_isco_rn_over_M'] is not None else "N/A"
        print(f"  {row['q']:6.2f}  {row['r_isco_over_M']:10.4f}  "
              f"{row['E_isco']:10.6f}  {row['eta_percent']:8.2f}  "
              f"{eta_rn_s:>8s}  {r_rn_s:>10s}  "
              f"{row['delta_vs_schwarz']:10.2f}")

    # --- 3. GM vs RN comparison ---
    print("\n--- 3. GM vs RN Comparison ---")
    print(f"  GM coupling a = {a:.6f},  RN has a = 0 (gamma = 1)")
    print()
    comp = compare_gm_rn_kerr(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], a)
    print(f"  {'q':>6s}  {'r_GM/M':>10s}  {'r_RN/M':>10s}  "
          f"{'eta_GM %':>10s}  {'eta_RN %':>10s}  "
          f"{'r_GM/r_RN':>10s}  {'eta_GM/eta_RN':>14s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")
    for row in comp:
        if "error" in row:
            print(f"  {row['q']:6.2f}  {'error':>10s}")
            continue
        r_rn_s = f"{row['r_isco_rn_over_M']:.4f}" if row['r_isco_rn_over_M'] else "N/A"
        eta_rn_s = f"{row['eta_rn_percent']:.2f}" if row['eta_rn_percent'] else "N/A"
        r_rat_s = f"{row['r_ratio_gm_rn']:.6f}" if row['r_ratio_gm_rn'] else "N/A"
        eta_rat_s = f"{row['eta_ratio_gm_rn']:.6f}" if row['eta_ratio_gm_rn'] else "N/A"
        print(f"  {row['q']:6.2f}  {row['r_isco_gm_over_M']:10.4f}  "
              f"{r_rn_s:>10s}  {row['eta_gm_percent']:10.2f}  "
              f"{eta_rn_s:>10s}  {r_rat_s:>10s}  {eta_rat_s:>14s}")

    # --- 4. Luminosity comparison ---
    print("\n--- 4. Accretion Luminosity Comparison ---")
    cases = [
        ("Stellar BH (10 Msun, 1e-8 Msun/yr)", 10.0, 1.0e-8),
        ("Sgr A* (4e6 Msun, 1e-5 Msun/yr)", 4.0e6, 1.0e-5),
        ("Quasar (1e9 Msun, 1.0 Msun/yr)", 1.0e9, 1.0),
    ]
    for label, M_sol, mdot in cases:
        print(f"\n  {label}, q = 0.5:")
        lum = luminosity_comparison(M_sol, mdot, 0.5, a)
        if "error" in lum:
            print(f"    Error: {lum['error']}")
            continue
        print(f"    eta_GM     = {lum['eta_gm']*100:.2f}%")
        print(f"    eta_Schwarz= {lum['eta_schwarz']*100:.2f}%")
        if lum['eta_rn'] is not None:
            print(f"    eta_RN     = {lum['eta_rn']*100:.2f}%")
        print(f"    L_GM       = {lum['L_gm_W']:.4e} W  = {lum['L_gm_Lsun']:.4e} L_sun")
        print(f"    L_Schwarz  = {lum['L_schwarz_W']:.4e} W  = {lum['L_schwarz_Lsun']:.4e} L_sun")
        if lum['L_rn_W'] is not None:
            print(f"    L_RN       = {lum['L_rn_W']:.4e} W  = {lum['L_rn_Lsun']:.4e} L_sun")
        print(f"    delta_L    = {lum['delta_L_percent']:+.2f}% vs Schwarzschild")
        print(f"    L_Eddington= {lum['L_eddington_W']:.4e} W  = {lum['L_eddington_Lsun']:.4e} L_sun")

    # --- 5. Summary ---
    print("\n--- 5. Summary and Honest Assessment ---")
    print()

    # Retrieve key values
    row_05 = None
    for row in scan:
        if "error" not in row and abs(row["q"] - 0.5) < 0.01:
            row_05 = row
            break
    row_09 = None
    for row in scan:
        if "error" not in row and abs(row["q"] - 0.9) < 0.01:
            row_09 = row
            break

    print("  The Gibbons-Maeda dilaton black hole with the Alpha Ladder")
    print(f"  coupling a = 1/sqrt(3) (gamma = {gam:.4f}) modifies the ISCO")
    print("  and accretion efficiency relative to both Schwarzschild and RN.")
    print()
    print(f"  Schwarzschild baseline:")
    print(f"    r_ISCO = 6.0000 M,  eta = {eta_schwarz*100:.2f}%")
    print()
    if row_05:
        print(f"  At q = 0.5 (GM dilaton):")
        print(f"    r_ISCO = {row_05['r_isco_over_M']:.4f} M,  eta = {row_05['eta_percent']:.2f}%")
        if row_05['eta_rn_percent'] is not None:
            print(f"    RN comparison: eta_RN = {row_05['eta_rn_percent']:.2f}%")
        print(f"    Change vs Schwarzschild: {row_05['delta_vs_schwarz']:+.2f}%")
    print()
    if row_09:
        print(f"  At q = 0.9 (GM dilaton):")
        print(f"    r_ISCO = {row_09['r_isco_over_M']:.4f} M,  eta = {row_09['eta_percent']:.2f}%")
        if row_09['eta_rn_percent'] is not None:
            print(f"    RN comparison: eta_RN = {row_09['eta_rn_percent']:.2f}%")
    print()
    print("  Key physics:")
    print("  1. At q = 0 (neutral): ISCO = 6M, eta = 5.72% (identical to GR).")
    print("     No deviation whatsoever for uncharged black holes.")
    print()
    print("  2. With increasing q, the ISCO moves inward and efficiency rises.")
    print("     The dilaton field modifies the rate of ISCO shift compared to RN.")
    print()
    print("  3. The GM metric has gamma = 1/2 (vs gamma = 1 for RN), which")
    print("     changes R(r)^2 = r^2 (1 - r-/r)^{1/2}.  This modified areal")
    print("     radius alters the effective potential and hence the ISCO.")
    print()
    print("  HONEST CAVEATS:")
    print()
    print("  - Astrophysical black holes are expected to be electrically neutral")
    print("    to extremely high precision.  Selective charge accretion and pair")
    print("    production discharge any macroscopic charge on timescales much")
    print("    shorter than the BH age.  Therefore q ~ 0 for all observed BHs,")
    print("    and the dilaton ISCO deviation is unobservably small in practice.")
    print()
    print("  - If the dilaton acquires a Planck-scale mass (as predicted by the")
    print("    Alpha Ladder flux stabilization), it decouples at astrophysical")
    print("    scales.  In that case BH solutions revert exactly to GR and the")
    print("    entire ISCO analysis becomes academic.")
    print()
    print("  - Real accretion disks are thick, turbulent, magnetized, and")
    print("    subject to radiative transfer effects.  The thin-disk Novikov-")
    print("    Thorne model (which uses the ISCO) is an idealization.  GRMHD")
    print("    simulations show matter can plunge inside the ISCO and radiate,")
    print("    and magnetic stresses can extract additional energy.")
    print()
    print("  - The value of this analysis is theoretical: it maps out the")
    print("    observable consequences of the Alpha Ladder's specific dilaton")
    print("    coupling and provides concrete, falsifiable predictions for the")
    print("    accretion properties of charged dilaton black holes.")

    print("\n" + "=" * 76)
    print("Done.")

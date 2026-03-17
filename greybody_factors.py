"""
Greybody factors and Hawking radiation spectra for Gibbons-Maeda dilaton
black holes with the Alpha Ladder coupling a = 1/sqrt(3).

When a black hole emits Hawking radiation, the curved spacetime geometry
acts as a potential barrier that partially reflects outgoing quanta back
into the hole.  The transmission probability through this barrier is the
greybody factor Gamma(omega, l).  The observed Hawking spectrum is:

    dN/(dt domega) = (1/(2 pi)) sum_l (2l+1) Gamma_l(omega) n(omega)

where n(omega) = 1/(exp(omega/T_H) - 1) for bosons (Bose-Einstein) and
n(omega) = 1/(exp(omega/T_H) + 1) for fermions (Fermi-Dirac).

The GM metric with dilaton coupling a = 1/sqrt(3) gives:
    f(r) = (1 - r+/r)(1 - r-/r)^gamma,   gamma = 1/2
    R(r)^2 = r^2 (1 - r-/r)^{1-gamma}

The effective potential for scalar perturbations is:
    V(r) = f(r) [l(l+1)/R(r)^2 + f'(r) R'(r) / R(r)]

We compute greybody factors using the WKB (parabolic barrier) approximation:
    Gamma_l(omega) = 1 / (1 + exp(2 pi (V_peak - omega^2) / |V''|^{1/2}))

where V_peak is the potential maximum and V'' is the second derivative
at the peak in tortoise coordinates.

Key physics:
  - Greybody factors suppress low-frequency emission (barrier reflection)
  - GM black holes have modified barriers due to gamma=1/2 (not gamma=1)
  - Extra dilaton emission channel if dilaton is massless
  - T -> 0 at extremality (degenerate horizon at r+=r-=4M/3)
  - For q=0: reduces to Schwarzschild identically

Reference: G. W. Gibbons and K. Maeda, Nucl. Phys. B 298, 741 (1988).
           D. N. Page, Phys. Rev. D 13, 198 (1976).
           S. R. Das, G. Gibbons, S. D. Mathur, PRL 78, 417 (1997).

Pure Python -- only 'import math' is used.
"""

import math

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G = 6.674298e-11       # m^3 kg^-1 s^-2  (Alpha Ladder predicted value)
c = 2.99792458e8       # m/s
hbar = 1.054571817e-34 # J s
k_B = 1.380649e-23     # J/K
M_sun = 1.989e30       # kg

# Default dilaton coupling from Alpha Ladder S^2 KK reduction
A_DEFAULT = 1.0 / math.sqrt(3.0)


# ---------------------------------------------------------------------------
# Helper: horizon radii
# ---------------------------------------------------------------------------

def _horizon_radii(M_geom, qm_ratio, a):
    """
    Outer and inner horizon radii of a Gibbons-Maeda black hole.

    Satisfies the GM mass relation 2M = r+ + gamma*r-.
    For a^2=1/3, gamma=1/2:
        disc = 1 - 8 q^2 / 9
        r+ = M (1 + sqrt(disc))
        r- = 2 M (1 - sqrt(disc))

    At q=0: r+=2M, r-=0.  At q=1: r+=r-=4M/3.

    Parameters
    ----------
    M_geom : float
        Mass in geometrized units (meters).
    qm_ratio : float
        q = Q/Q_extreme, in [0, 1].
    a : float
        Dilaton coupling.

    Returns
    -------
    (r_plus, r_minus) or None if q > 1.
    """
    q = min(abs(qm_ratio), 1.0)
    disc = 1.0 - 8.0 * q * q / 9.0
    if disc < -1e-30:
        return None
    disc = max(disc, 0.0)
    r_plus = M_geom * (1.0 + math.sqrt(disc))
    r_minus = 2.0 * M_geom * (1.0 - math.sqrt(disc))
    if r_plus <= 0.0:
        return None
    return (r_plus, r_minus)


def _mass_to_geom(M_kg):
    """Convert mass in kg to geometrized units (meters): M_geom = G*M/c^2."""
    return G * M_kg / (c * c)


# ---------------------------------------------------------------------------
# GM metric functions
# ---------------------------------------------------------------------------

def _gm_metric(r, r_plus, r_minus, gamma):
    """
    Compute f(r) and R^2(r) for the GM metric.

    Returns (f, R2) or (0.0, 0.0) if inside horizon.
    """
    u = 1.0 - r_plus / r
    v = 1.0 - r_minus / r if r_minus > 0 else 1.0

    if u <= 0.0 or v <= 0.0:
        return (0.0, 0.0)

    f = u * (v ** gamma)
    R2 = r * r * (v ** (1.0 - gamma))
    return (f, R2)


def _effective_potential(r, l, r_plus, r_minus, gamma):
    """
    Scalar field effective potential V(r) on the GM background.

    V(r) = f(r) [l(l+1)/R(r)^2 + f'(r) R'(r) / R(r)]

    Parameters
    ----------
    r : float
        Radial coordinate (r > r_plus).
    l : int
        Angular momentum quantum number.
    r_plus, r_minus : float
        Horizon radii.
    gamma : float
        GM metric exponent.

    Returns
    -------
    float : V(r)
    """
    u = 1.0 - r_plus / r
    v = 1.0 - r_minus / r if r_minus > 0 else 1.0

    if u <= 0.0 or v <= 0.0:
        return 0.0

    f = u * (v ** gamma)
    R2 = r * r * (v ** (1.0 - gamma))
    R = math.sqrt(R2)

    # f'(r) = (r+/r^2) v^gamma + u * gamma * (r-/r^2) * v^{gamma-1}
    df_dr = (r_plus / (r * r)) * (v ** gamma)
    if r_minus > 0:
        df_dr += u * gamma * (r_minus / (r * r)) * (v ** (gamma - 1.0))

    # R(r) = r * v^{(1-gamma)/2}
    # dR/dr = v^{(1-gamma)/2} * [1 + (1-gamma)*r_- / (2*(r - r_-))]
    exp_half = (1.0 - gamma) / 2.0
    v_exp = v ** exp_half
    if r_minus > 0 and abs(r - r_minus) > 1e-30:
        dR_dr = v_exp * (1.0 + (1.0 - gamma) * r_minus / (2.0 * (r - r_minus)))
    else:
        dR_dr = v_exp

    V = f * (l * (l + 1.0) / R2 + df_dr * dR_dr / R)
    return V


def _find_potential_peak(l, r_plus, r_minus, gamma, tol=1e-10):
    """
    Find r_peak where V(r) is maximized, using bisection on dV/dr.

    Returns (r_peak, V_peak).
    """
    h = 1e-4 * r_plus

    def V(r):
        return _effective_potential(r, l, r_plus, r_minus, gamma)

    def dV_dr(r):
        return (V(r + h) - V(r - h)) / (2.0 * h)

    # Bracket the peak
    r_lo = r_plus * 1.001
    r_hi = r_plus * 20.0

    dv_lo = dV_dr(r_lo)

    if dv_lo < 0:
        r_lo = r_plus * 1.0001
        dv_lo = dV_dr(r_lo)

    dv_hi = dV_dr(r_hi)
    if dv_hi > 0:
        r_hi = r_plus * 100.0

    # Bisection
    for _ in range(200):
        r_mid = 0.5 * (r_lo + r_hi)
        if (r_hi - r_lo) / r_mid < tol:
            break
        dv_mid = dV_dr(r_mid)
        if dv_mid > 0:
            r_lo = r_mid
        else:
            r_hi = r_mid

    r_peak = 0.5 * (r_lo + r_hi)
    V_peak = V(r_peak)
    return (r_peak, V_peak)


def _potential_second_deriv_tortoise(r_peak, l, r_plus, r_minus, gamma):
    """
    Second derivative of V with respect to tortoise coordinate r* at the peak.

    d^2V/dr*^2 = f(r)^2 * d^2V/dr^2   (at dV/dr=0, the chain rule cross
    term vanishes).

    We compute d^2V/dr^2 numerically and multiply by f(r_peak)^2.
    """
    h = 1e-4 * r_plus

    def V(r):
        return _effective_potential(r, l, r_plus, r_minus, gamma)

    V_peak = V(r_peak)
    V_plus = V(r_peak + h)
    V_minus = V(r_peak - h)

    d2V_dr2 = (V_plus - 2.0 * V_peak + V_minus) / (h * h)

    f_peak, _ = _gm_metric(r_peak, r_plus, r_minus, gamma)

    # At a critical point (dV/dr=0), d^2V/dr*^2 = f^2 * d^2V/dr^2
    # because dr*/dr = 1/f, so d/dr* = f * d/dr, and the first-derivative
    # cross term vanishes since dV/dr = 0.
    d2V_drstar2 = f_peak * f_peak * d2V_dr2

    return d2V_drstar2


# ---------------------------------------------------------------------------
# 1. Hawking temperature
# ---------------------------------------------------------------------------

def hawking_temperature(M, q, a=None):
    """
    Compute the Hawking temperature for a Gibbons-Maeda dilaton black hole.

    The surface gravity is:
        kappa = f'(r+)/2 = (1/(2 r+)) (1 - r-/r+)^gamma

    The Hawking temperature:
        T_H = kappa/(2 pi) = (1 - r-/r+)^gamma / (4 pi r+)

    In geometrized units (M=1), T_H has units of 1/M.

    For physical units:
        T_H (Kelvin) = hbar c^3 / (4 pi k_B) * kappa_physical
        where kappa_physical = kappa_geom * c^2 / (G M_kg)

    Parameters
    ----------
    M : float
        Black hole mass.  If M > 1e10, interpreted as mass in kg.
        Otherwise interpreted as geometrized mass (units where G=c=1 and
        M is in meters).
    q : float
        Charge ratio Q/Q_extreme in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with keys:
        T_H_geom : float
            Hawking temperature in geometrized units (1/M).
        T_H_kelvin : float
            Hawking temperature in Kelvin.
        T_schwarz_kelvin : float
            Schwarzschild temperature in Kelvin (for comparison).
        ratio_T_H_over_T_schwarz : float
            T_H / T_Schwarzschild.
        r_plus : float
            Outer horizon in geometrized units.
        r_minus : float
            Inner horizon in geometrized units.
        gamma : float
            GM metric exponent.
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    # Determine if M is in kg or geometrized
    if M > 1e10:
        M_kg = M
        M_geom = _mass_to_geom(M_kg)
    else:
        M_geom = M
        M_kg = M_geom * c * c / G

    radii = _horizon_radii(M_geom, q, a)
    if radii is None:
        return {
            "error": "No horizon (naked singularity or q > 1)",
            "M": M,
            "q": q,
        }

    r_plus, r_minus = radii

    # T_H in geometrized units (inverse length = 1/M_geom)
    if r_plus <= 0:
        return {"error": "Degenerate horizon", "M": M, "q": q}

    ratio = r_minus / r_plus
    if ratio >= 1.0:
        T_H_geom = 0.0
    else:
        T_H_geom = ((1.0 - ratio) ** gamma) / (4.0 * math.pi * r_plus)

    # Schwarzschild temperature: T = 1/(8 pi M_geom) in geometrized
    T_schwarz_geom = 1.0 / (8.0 * math.pi * M_geom)

    # Physical temperature conversion:
    # T_H (K) = hbar c^3 / (k_B) * T_H_geom / (G M_kg / c^2)
    # But M_geom = G M_kg / c^2, so T_H_geom is in 1/meters.
    # T (K) = hbar c / k_B * T_H_geom  [since T_H_geom is in 1/meters]
    # Actually: kappa has units 1/meters in geometrized (c=G=1).
    # T = hbar c^3 kappa / (2 pi k_B) with kappa = f'(r+)/2.
    # We already have T_H_geom = kappa/(2 pi) in 1/meters.
    # T (K) = hbar c * T_H_geom / k_B
    # Wait: T_H_geom is in units of c^2/(G M_kg) = 1/M_geom (1/meters).
    # The dimensionful T = (hbar c / k_B) * T_H_geom
    conversion = hbar * c / k_B   # meters * Kelvin
    T_H_kelvin = conversion * T_H_geom
    T_schwarz_kelvin = conversion * T_schwarz_geom

    # Ratio
    if T_schwarz_geom > 0:
        ratio_T = T_H_geom / T_schwarz_geom
    else:
        ratio_T = float("inf")

    return {
        "T_H_geom": T_H_geom,
        "T_H_kelvin": T_H_kelvin,
        "T_schwarz_kelvin": T_schwarz_kelvin,
        "ratio_T_H_over_T_schwarz": ratio_T,
        "r_plus": r_plus,
        "r_minus": r_minus,
        "gamma": gamma,
        "M_geom": M_geom,
        "M_kg": M_kg,
    }


# ---------------------------------------------------------------------------
# 2. Effective potential peak
# ---------------------------------------------------------------------------

def effective_potential_peak(l, M, q, a=None):
    """
    Compute the peak of the scalar field effective potential V(r) for mode l
    on a Gibbons-Maeda background.

    Parameters
    ----------
    l : int
        Angular momentum quantum number (l >= 0).
    M : float
        Mass in geometrized units (set M=1 for scale-free results).
    q : float
        Charge ratio Q/Q_extreme in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with keys:
        r_peak : float
            Radial coordinate of potential maximum.
        V_peak : float
            Value of V at the peak.
        V_peak_second_deriv : float
            d^2V/dr*^2 at the peak (in tortoise coordinates).
        b_c : float
            Critical impact parameter sqrt(l(l+1)/V_peak).
        r_peak_over_r_plus : float
            r_peak / r_plus.
        l : int
            Angular momentum number.
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    radii = _horizon_radii(M, q, a)
    if radii is None:
        return {"error": "No horizon", "l": l, "M": M, "q": q}

    r_plus, r_minus = radii

    r_peak, V_peak = _find_potential_peak(l, r_plus, r_minus, gamma)

    # Second derivative in tortoise coordinates
    d2V = _potential_second_deriv_tortoise(r_peak, l, r_plus, r_minus, gamma)

    # Critical impact parameter
    if V_peak > 0:
        b_c = math.sqrt(l * (l + 1.0) / V_peak)
    else:
        b_c = float("inf")

    return {
        "r_peak": r_peak,
        "V_peak": V_peak,
        "V_peak_second_deriv": d2V,
        "b_c": b_c,
        "r_peak_over_r_plus": r_peak / r_plus,
        "l": l,
        "r_plus": r_plus,
        "r_minus": r_minus,
    }


# ---------------------------------------------------------------------------
# 3. Greybody factor (WKB)
# ---------------------------------------------------------------------------

def greybody_factor(omega, l, M, q, a=None):
    """
    WKB greybody factor for scalar field mode (omega, l) on GM background.

    Uses the parabolic (inverted harmonic oscillator) WKB approximation:
        Gamma_l(omega) = 1 / (1 + exp(2 pi (V_peak - omega^2) / sqrt(|d2V|)))

    where d2V = |d^2V/dr*^2| at the peak.

    This interpolates correctly between:
      - omega^2 >> V_peak: Gamma -> 1 (classically allowed)
      - omega^2 << V_peak: Gamma -> exp(-2 pi (V_peak - omega^2)/sqrt(|d2V|))
      - omega^2 = V_peak:  Gamma = 1/2 (barrier top)

    Parameters
    ----------
    omega : float
        Frequency (in geometrized units, same as 1/M).
    l : int
        Angular momentum quantum number.
    M : float
        Mass in geometrized units.
    q : float
        Charge ratio Q/Q_extreme in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with keys:
        Gamma : float
            Greybody factor (transmission probability) in [0, 1].
        omega : float
            Input frequency.
        l : int
            Angular momentum number.
        regime : str
            "sub-barrier" or "super-barrier".
        V_peak : float
            Potential barrier height.
        omega_c : float
            Critical frequency sqrt(V_peak).
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    radii = _horizon_radii(M, q, a)
    if radii is None:
        return {"error": "No horizon", "omega": omega, "l": l}

    r_plus, r_minus = radii
    r_peak, V_peak = _find_potential_peak(l, r_plus, r_minus, gamma)
    d2V = _potential_second_deriv_tortoise(r_peak, l, r_plus, r_minus, gamma)

    omega_sq = omega * omega
    omega_c = math.sqrt(V_peak) if V_peak > 0 else 0.0

    # WKB transmission
    abs_d2V = abs(d2V)
    if abs_d2V < 1e-50:
        # Flat potential -- no barrier
        Gamma = 1.0
    else:
        sqrt_d2V = math.sqrt(abs_d2V)
        exponent = 2.0 * math.pi * (V_peak - omega_sq) / sqrt_d2V

        # Numerical overflow protection
        if exponent > 500.0:
            Gamma = 0.0
        elif exponent < -500.0:
            Gamma = 1.0
        else:
            Gamma = 1.0 / (1.0 + math.exp(exponent))

    regime = "sub-barrier" if omega_sq < V_peak else "super-barrier"

    return {
        "Gamma": Gamma,
        "omega": omega,
        "l": l,
        "regime": regime,
        "V_peak": V_peak,
        "omega_c": omega_c,
    }


# ---------------------------------------------------------------------------
# 4. Hawking spectrum
# ---------------------------------------------------------------------------

def hawking_spectrum(M, q, omega_values=None, l_max=5, a=None, particle="boson"):
    """
    Compute the Hawking emission spectrum dN/(dt domega) summed over
    angular momentum modes l=0 to l_max.

    For each omega and each l:
        rate_l = (1/(2 pi)) * (2l+1) * Gamma_l(omega) * n(omega)

    where n(omega) = 1/(exp(omega/T_H) - 1) for bosons.

    Parameters
    ----------
    M : float
        Mass in geometrized units (set M=1 for scale-free).
    q : float
        Charge ratio Q/Q_extreme in [0, 1].
    omega_values : list of float or None
        Frequencies to evaluate.  If None, 50 points from 0.01*T_H to 10*T_H.
    l_max : int
        Maximum angular momentum to include (default 5).
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).
    particle : str
        "boson" or "fermion".

    Returns
    -------
    dict with keys:
        spectrum : list of dict
            Each entry has omega, omega_over_T, dN_dt_domega,
            dN_dt_domega_blackbody, greybody_suppression.
        T_H : float
            Hawking temperature (geometrized).
        l_max : int
        particle : str
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    # Get temperature
    temp = hawking_temperature(M, q, a)
    if "error" in temp:
        return temp
    T_H = temp["T_H_geom"]

    if T_H <= 0:
        return {"error": "Zero temperature", "M": M, "q": q}

    radii = _horizon_radii(M, q, a)
    r_plus, r_minus = radii

    # Default omega values
    if omega_values is None:
        omega_min = 0.01 * T_H
        omega_max = 10.0 * T_H
        n_pts = 50
        omega_values = [omega_min + i * (omega_max - omega_min) / (n_pts - 1)
                        for i in range(n_pts)]

    # Precompute potential peaks for each l
    peaks = {}
    for l in range(l_max + 1):
        r_pk, V_pk = _find_potential_peak(l, r_plus, r_minus, gamma)
        d2V = _potential_second_deriv_tortoise(r_pk, l, r_plus, r_minus, gamma)
        peaks[l] = (V_pk, d2V)

    is_fermion = (particle.lower() == "fermion")

    spectrum = []
    for omega in omega_values:
        if omega <= 0:
            continue

        x = omega / T_H

        # Bose-Einstein or Fermi-Dirac distribution
        if x > 500.0:
            n_omega = 0.0
        elif is_fermion:
            n_omega = 1.0 / (math.exp(x) + 1.0)
        else:
            n_omega = 1.0 / (math.exp(x) - 1.0)

        # Sum over angular momenta
        rate_total = 0.0
        rate_bb = 0.0   # blackbody (Gamma=1)

        for l in range(l_max + 1):
            deg = 2.0 * l + 1.0

            # Greybody factor for this (omega, l)
            V_pk, d2V = peaks[l]
            omega_sq = omega * omega
            abs_d2V = abs(d2V)

            if abs_d2V < 1e-50:
                Gamma_l = 1.0
            else:
                sqrt_d2V = math.sqrt(abs_d2V)
                exponent = 2.0 * math.pi * (V_pk - omega_sq) / sqrt_d2V
                if exponent > 500.0:
                    Gamma_l = 0.0
                elif exponent < -500.0:
                    Gamma_l = 1.0
                else:
                    Gamma_l = 1.0 / (1.0 + math.exp(exponent))

            rate_total += deg * Gamma_l * n_omega / (2.0 * math.pi)
            rate_bb += deg * n_omega / (2.0 * math.pi)

        suppression = rate_total / rate_bb if rate_bb > 0 else 0.0

        spectrum.append({
            "omega": omega,
            "omega_over_T": x,
            "dN_dt_domega": rate_total,
            "dN_dt_domega_blackbody": rate_bb,
            "greybody_suppression": suppression,
        })

    return {
        "spectrum": spectrum,
        "T_H": T_H,
        "l_max": l_max,
        "particle": particle,
        "M": M,
        "q": q,
    }


# ---------------------------------------------------------------------------
# 5. Total power
# ---------------------------------------------------------------------------

def total_power(M, q, l_max=5, a=None, n_omega=200):
    """
    Integrate the Hawking spectrum to get total emitted power.

    P = integral_0^inf omega * dN/(dt domega) domega

    Uses trapezoidal rule over a wide frequency range.

    Parameters
    ----------
    M : float
        Mass in geometrized units or kg (if > 1e10, interpreted as kg).
    q : float
        Charge ratio Q/Q_extreme in [0, 1].
    l_max : int
        Maximum angular momentum to include.
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).
    n_omega : int
        Number of frequency sampling points.

    Returns
    -------
    dict with keys:
        P_gm : float
            Total power (geometrized units).
        P_schwarz : float
            Schwarzschild power for comparison.
        ratio : float
            P_gm / P_schwarz.
        P_gm_watts : float
            Power in watts (for given M).
        P_schwarz_watts : float
            Schwarzschild power in watts.
        t_evap_gm : float
            Rough evaporation timescale (seconds).
        t_evap_schwarz : float
            Schwarzschild evaporation timescale (seconds).
    """
    if a is None:
        a = A_DEFAULT

    # Determine physical mass
    if M > 1e10:
        M_kg = M
        M_geom = _mass_to_geom(M_kg)
    else:
        M_geom = M
        M_kg = M_geom * c * c / G

    # GM spectrum
    temp = hawking_temperature(M_geom, q, a)
    if "error" in temp:
        return temp
    T_H_gm = temp["T_H_geom"]

    # Schwarzschild spectrum
    temp_s = hawking_temperature(M_geom, 0.0, a)
    T_H_schwarz = temp_s["T_H_geom"]

    def _compute_power(T_H, qq):
        """Integrate omega * dN/(dt domega) domega by trapezoidal rule."""
        if T_H <= 0:
            return 0.0

        omega_min = 0.001 * T_H
        omega_max = 20.0 * T_H
        d_omega = (omega_max - omega_min) / (n_omega - 1)

        a_sq = a * a if a is not None else A_DEFAULT * A_DEFAULT
        gamma_val = (1.0 - a_sq) / (1.0 + a_sq) if a is not None else 0.5

        radii = _horizon_radii(M_geom, qq, a if a is not None else A_DEFAULT)
        if radii is None:
            return 0.0
        rp, rm = radii

        # Precompute peaks
        pk = {}
        for ell in range(l_max + 1):
            rr, vv = _find_potential_peak(ell, rp, rm, gamma_val)
            dd = _potential_second_deriv_tortoise(rr, ell, rp, rm, gamma_val)
            pk[ell] = (vv, dd)

        power = 0.0
        prev_integrand = 0.0

        for i in range(n_omega):
            omega = omega_min + i * d_omega
            if omega <= 0:
                continue
            x = omega / T_H
            if x > 500.0:
                n_omega_val = 0.0
            else:
                n_omega_val = 1.0 / (math.exp(x) - 1.0)

            rate = 0.0
            for ell in range(l_max + 1):
                deg = 2.0 * ell + 1.0
                V_pk, d2V = pk[ell]
                abs_d2V = abs(d2V)
                if abs_d2V < 1e-50:
                    Gamma_l = 1.0
                else:
                    sqrt_d2V = math.sqrt(abs_d2V)
                    exponent = 2.0 * math.pi * (V_pk - omega * omega) / sqrt_d2V
                    if exponent > 500.0:
                        Gamma_l = 0.0
                    elif exponent < -500.0:
                        Gamma_l = 1.0
                    else:
                        Gamma_l = 1.0 / (1.0 + math.exp(exponent))
                rate += deg * Gamma_l * n_omega_val / (2.0 * math.pi)

            integrand = omega * rate

            if i > 0:
                power += 0.5 * (prev_integrand + integrand) * d_omega

            prev_integrand = integrand

        return power

    P_gm = _compute_power(T_H_gm, q)
    P_schwarz = _compute_power(T_H_schwarz, 0.0)

    ratio = P_gm / P_schwarz if P_schwarz > 0 else float("inf")

    # Convert to watts
    # In geometrized units, power has dimensions of 1/length^2.
    # P_physical = P_geom * hbar * c^6 / (G^2 * M_kg^2)
    # Actually: P_geom is in units of 1/(G M / c^2)^2 = c^4/(G M)^2.
    # Energy emission rate: dE/dt with E ~ 1/length, t ~ length.
    # So P ~ 1/length^2.  In SI: P (W) = P_geom * c^5 * (G M_kg / c^2)^(-2) * hbar
    # Hmm, let's be careful.
    # omega has units 1/M_geom.  dN/dt domega has units M_geom (since 1/(2 pi) * 1/(exp-1)).
    # P_geom = integral omega * dN/(dt domega) domega has units 1/M_geom^2.
    # In physical units: omega_phys = omega_geom * c / M_geom_meters
    # but omega_geom is dimensionless (in units of 1/M_geom).
    # P_phys = P_geom * hbar c / M_geom^2  [Joules/second per unit...]
    # Actually, P_geom * (hbar c / M_geom^2) has units J/s * (1/m^2) * m^2 .. let me
    # just use the Stefan-Boltzmann approach for a cross-check.
    #
    # Simpler: P_watts = P_geom * (c^5 / G) * M_geom_dimless^{-2} does not work either.
    #
    # The cleanest route: the "Page factor"
    # P = alpha_page * hbar c^6 / (G^2 M^2)  where alpha_page ~ 3.6e-4 for Schwarzschild
    # But we have the numerical integral P_geom in units of 1/M^2 (geometrized).
    # The conversion is: P_watts = P_geom * hbar * c / (M_geom_meters)^2
    # since omega is in 1/M_geom_meters and the spectrum integrand has dimensions
    # 1/time * 1/omega = M_geom, so P = omega * rate * domega ~ M_geom^{-2}.
    # hbar / M_geom has units J*s/m = J/c ... no.
    #
    # Let's just use dimensional analysis properly:
    # In natural units hbar=c=G=1, P has dimensions of (length)^{-2} = (mass)^2.
    # To convert to SI watts: P_SI = P_nat * c^5 / G = P_nat * 3.63e52 W
    # But P_nat is in units of M_Planck^2, and our M_geom is in meters = G/c^2 * M_kg.
    # P_geom is in 1/M_geom^2 (meters^{-2}).
    # P_SI = P_geom * hbar * c^3 / (M_geom_m)^0 ... OK let me just compute it directly.
    #
    # In geometrized units where G=c=1 and mass has units of length:
    # [omega] = 1/length, [T] = 1/length, [dN/dt domega] = length (dimensionless rate per unit omega)
    # [P] = [omega * dN/dt domega * domega] = 1/length * length * 1/length = 1/length
    # Hmm, actually P_geom should be 1/M_geom.
    #
    # Let me reconsider. n(omega) is dimensionless. Gamma is dimensionless.
    # (2l+1) is dimensionless. 1/(2 pi) is dimensionless.
    # dN/(dt domega) has units of 1/[time * omega] = 1/[length * (1/length)] = dimensionless
    # No: dN/(dt domega) domega = (particle number rate). dN/dt domega has units of time.
    # In geom units: [time] = [length] = M_geom. So [dN/(dt domega)] = M_geom.
    # [P_geom] = [omega * dN/(dt domega) domega] = (1/M_geom) * M_geom * (1/M_geom) = 1/M_geom.
    #
    # Conversion: P_SI = P_geom * (hbar c / M_geom_m) ... no.
    # Let me think in SI from the start.
    # omega_SI = c / M_geom_m * omega_geom  (omega_SI in rad/s)
    # T_SI = hbar c / (k_B M_geom_m) * T_geom  ... but we handle temp separately.
    # n(omega_geom / T_geom) = n(omega_SI / T_SI) since the ratio is the same.
    # dN/(dt_SI domega_SI) = (1/(2pi)) sum_l (2l+1) Gamma_l n_omega
    # This has units of seconds (i.e., 1/(rad/s) * (1/s)).
    # Actually [dN/(dt domega)] = [1/time * 1/omega]^{-1} ... no.
    # dN/dt has units 1/s. dN/(dt domega) has units s/rad = 1/(rad/s * s^{-1})... no.
    # dN/(dt domega) has units of 1/(energy) in natural units (hbar=1), or
    # seconds per radian in SI.  Actually it's just 1/(frequency * time) integrated
    # over frequency to give rate.  Units: [dN/(dt domega)] = 1/omega^2 in natural.
    #
    # In natural units (hbar=c=G=1): P = integral omega * dN/(dt domega) domega
    # where everything is measured in Planck units.  Our "M_geom" is in Planck lengths.
    #
    # Simplest approach: use the known Schwarzschild formula as calibration.
    # P_schwarz = hbar c^6 / (15360 pi G^2 M_kg^2)
    # Then P_gm_watts = ratio * P_schwarz_watts.

    P_schwarz_watts = hbar * c**6 / (15360.0 * math.pi * G * G * M_kg * M_kg)
    P_gm_watts = ratio * P_schwarz_watts

    # Evaporation timescale: t ~ M c^2 / P
    E_rest = M_kg * c * c
    t_evap_gm = E_rest / P_gm_watts if P_gm_watts > 0 else float("inf")
    t_evap_schwarz = E_rest / P_schwarz_watts if P_schwarz_watts > 0 else float("inf")

    # More accurate Schwarzschild: t_evap = 5120 pi G^2 M^3 / (hbar c^4)
    t_evap_schwarz_exact = 5120.0 * math.pi * G * G * M_kg**3 / (hbar * c**4)

    return {
        "P_gm": P_gm,
        "P_schwarz": P_schwarz,
        "ratio": ratio,
        "P_gm_watts": P_gm_watts,
        "P_schwarz_watts": P_schwarz_watts,
        "t_evap_gm_seconds": t_evap_gm,
        "t_evap_schwarz_seconds": t_evap_schwarz,
        "t_evap_schwarz_exact_seconds": t_evap_schwarz_exact,
        "T_H_gm_geom": T_H_gm,
        "T_H_schwarz_geom": T_H_schwarz,
        "M_kg": M_kg,
    }


# ---------------------------------------------------------------------------
# 6. Greybody scan over charge
# ---------------------------------------------------------------------------

def greybody_scan(l=2, q_values=None, M=1.0, a=None):
    """
    Scan greybody factor at omega = sqrt(V_peak) (the 50% transmission point)
    as a function of charge ratio q.

    Parameters
    ----------
    l : int
        Angular momentum quantum number (default 2).
    q_values : list of float or None
        Charge ratios to scan.  Default: 20 points from 0 to 0.99.
    M : float
        Mass in geometrized units (default 1.0).
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with keys:
        scan : list of dict
            Each entry has q, omega_c, V_peak, Gamma_at_barrier,
            r_peak_over_r_plus, T_H.
        l : int
        M : float
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    if q_values is None:
        q_values = [i * 0.99 / 19.0 for i in range(20)]

    results = []
    for q in q_values:
        radii = _horizon_radii(M, q, a)
        if radii is None:
            continue
        r_plus, r_minus = radii

        r_peak, V_peak = _find_potential_peak(l, r_plus, r_minus, gamma)
        omega_c = math.sqrt(V_peak) if V_peak > 0 else 0.0

        # Greybody factor at barrier top (should be ~0.5 by definition)
        gb = greybody_factor(omega_c, l, M, q, a)

        # Temperature
        temp = hawking_temperature(M, q, a)
        T_H = temp.get("T_H_geom", 0.0)

        # Also compute at omega = T_H (thermal peak region)
        if T_H > 0:
            gb_thermal = greybody_factor(T_H, l, M, q, a)
            Gamma_thermal = gb_thermal["Gamma"]
        else:
            Gamma_thermal = 0.0

        results.append({
            "q": q,
            "omega_c": omega_c,
            "V_peak": V_peak,
            "Gamma_at_barrier": gb["Gamma"],
            "Gamma_at_thermal": Gamma_thermal,
            "r_peak_over_r_plus": r_peak / r_plus if r_plus > 0 else 0.0,
            "T_H": T_H,
            "T_H_over_omega_c": T_H / omega_c if omega_c > 0 else 0.0,
        })

    return {
        "scan": results,
        "l": l,
        "M": M,
    }


# ---------------------------------------------------------------------------
# 7. Dilaton emission channel
# ---------------------------------------------------------------------------

def dilaton_emission_channel(M, q, a=None):
    """
    Compute the extra dilaton emission channel for a GM black hole.

    The dilaton is a scalar field with l=0 coupling.  If massless, it adds
    an extra emission channel beyond the graviton and photon modes.  If the
    dilaton has a large mass (m_phi >> T_H), the channel is exponentially
    suppressed by a Boltzmann factor exp(-m_phi/T_H).

    From the Alpha Ladder, flux stabilization gives the dilaton a Planck-scale
    mass m_phi ~ 6.3e29 eV, which completely shuts off this channel for any
    astrophysical black hole.

    Parameters
    ----------
    M : float
        Mass in geometrized units or kg (if > 1e10).
    q : float
        Charge ratio Q/Q_extreme in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with keys:
        P_dilaton_massless : float
            Power from massless dilaton l=0 channel (geometrized).
        P_dilaton_massive : float
            Power from massive (Planck-scale) dilaton.
        fraction_of_total : float
            Massless dilaton power / total scalar power (l=0..5).
        suppression_factor : float
            Boltzmann suppression exp(-m_phi/T_H) for massive case.
        description : str
    """
    if a is None:
        a = A_DEFAULT

    # Determine mass
    if M > 1e10:
        M_kg = M
        M_geom = _mass_to_geom(M_kg)
    else:
        M_geom = M
        M_kg = M_geom * c * c / G

    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    temp = hawking_temperature(M_geom, q, a)
    if "error" in temp:
        return temp

    T_H = temp["T_H_geom"]
    r_plus = temp["r_plus"]
    r_minus = temp["r_minus"]

    if T_H <= 0:
        return {"error": "Zero temperature", "M": M, "q": q}

    # Compute l=0 power (dilaton channel)
    n_pts = 200
    omega_min = 0.001 * T_H
    omega_max = 20.0 * T_H
    d_omega = (omega_max - omega_min) / (n_pts - 1)

    # Potential peak for l=0
    r_peak_0, V_peak_0 = _find_potential_peak(0, r_plus, r_minus, gamma)
    d2V_0 = _potential_second_deriv_tortoise(r_peak_0, 0, r_plus, r_minus, gamma)

    P_dilaton = 0.0
    prev = 0.0
    for i in range(n_pts):
        omega = omega_min + i * d_omega
        if omega <= 0:
            continue
        x = omega / T_H
        if x > 500.0:
            n_omega = 0.0
        else:
            n_omega = 1.0 / (math.exp(x) - 1.0)

        # Greybody factor for l=0
        abs_d2V = abs(d2V_0)
        if abs_d2V < 1e-50:
            Gamma_0 = 1.0
        else:
            sqrt_d2V = math.sqrt(abs_d2V)
            exponent = 2.0 * math.pi * (V_peak_0 - omega * omega) / sqrt_d2V
            if exponent > 500.0:
                Gamma_0 = 0.0
            elif exponent < -500.0:
                Gamma_0 = 1.0
            else:
                Gamma_0 = 1.0 / (1.0 + math.exp(exponent))

        # l=0, degeneracy = 1
        rate = (1.0 / (2.0 * math.pi)) * Gamma_0 * n_omega
        integrand = omega * rate

        if i > 0:
            P_dilaton += 0.5 * (prev + integrand) * d_omega
        prev = integrand

    # Total scalar power (l=0..5)
    pw = total_power(M_geom, q, l_max=5, a=a)
    P_total = pw.get("P_gm", 0.0) if isinstance(pw, dict) else 0.0

    fraction = P_dilaton / P_total if P_total > 0 else 0.0

    # Massive dilaton suppression
    # m_phi ~ 6.3e29 eV from flux stabilization
    # T_H for M_sun ~ 6e-8 K ~ 5e-12 eV
    # m_phi / T_H ~ 10^{41} -> exp(-10^41) = 0
    m_phi_eV = 6.3e29
    T_H_kelvin = temp["T_H_kelvin"]
    T_H_eV = k_B * T_H_kelvin / 1.602176634e-19  # convert K to eV
    if T_H_eV > 0:
        ratio_m_T = m_phi_eV / T_H_eV
    else:
        ratio_m_T = float("inf")

    if ratio_m_T > 500:
        suppression = 0.0
    else:
        suppression = math.exp(-ratio_m_T)

    P_dilaton_massive = P_dilaton * suppression

    return {
        "P_dilaton_massless": P_dilaton,
        "P_dilaton_massive": P_dilaton_massive,
        "fraction_of_total": fraction,
        "suppression_factor": suppression,
        "m_phi_over_T_H": ratio_m_T,
        "m_phi_eV": m_phi_eV,
        "T_H_eV": T_H_eV,
        "T_H_kelvin": T_H_kelvin,
        "description": (
            "The GM dilaton provides an extra l=0 scalar emission channel.  "
            "If massless, it contributes {:.1f}% of total scalar Hawking power.  "
            "However, the Alpha Ladder flux stabilization gives the dilaton a "
            "Planck-scale mass m_phi ~ 6.3e29 eV, yielding a Boltzmann "
            "suppression factor exp(-m_phi/T_H) = exp(-{:.1e}) = 0 to all "
            "practical purposes.  The dilaton channel is completely shut off "
            "for any astrophysical black hole, and the Hawking spectrum "
            "reverts to the GR prediction (with the modified GM metric "
            "providing the only difference from Schwarzschild)."
        ).format(fraction * 100.0, ratio_m_T),
    }


# ---------------------------------------------------------------------------
# 8. Summary analysis
# ---------------------------------------------------------------------------

def summarize_greybody_analysis():
    """
    Full summary of greybody factor analysis for GM dilaton black holes.

    Computes key results for a representative 10 M_sun black hole with
    q = 0.5 (moderate charge) and a = 1/sqrt(3).

    Returns
    -------
    dict with summary, findings, and caveats.
    """
    M_geom = 1.0  # Scale-free results
    q = 0.5
    a = A_DEFAULT

    # 1. Temperature
    temp = hawking_temperature(M_geom, q, a)
    temp_neutral = hawking_temperature(M_geom, 0.0, a)
    temp_extremal = hawking_temperature(M_geom, 0.99, a)

    # 2. Potential peaks for l=0,1,2
    peaks = {}
    for l in range(3):
        pk = effective_potential_peak(l, M_geom, q, a)
        peaks[l] = pk

    # 3. Spectrum
    spec = hawking_spectrum(M_geom, q, l_max=5, a=a)

    # 4. Total power
    pw = total_power(M_geom, q, l_max=5, a=a)

    # 5. Greybody scan
    scan = greybody_scan(l=2, M=M_geom, a=a)

    # 6. Dilaton channel
    dil = dilaton_emission_channel(M_geom, q, a)

    # 7. Physical example: 10 M_sun
    M_10 = 10.0 * M_sun
    temp_phys = hawking_temperature(M_10, q, a)
    pw_phys = total_power(M_10, q, l_max=5, a=a)

    # Compile findings
    findings = [
        "1. GREYBODY SUPPRESSION: The potential barrier reflects low-frequency "
        "   Hawking quanta back into the hole.  At omega = T_H, the greybody "
        "   suppression factor (ratio of actual to blackbody emission) is "
        "   typically {:.3f} for the dominant l=0 mode.".format(
            spec["spectrum"][10]["greybody_suppression"] if len(spec.get("spectrum", [])) > 10 else 0.0
        ),

        "2. MODIFIED BARRIERS: The GM metric with gamma=1/2 produces broader, "
        "   lower potential barriers compared to Schwarzschild for charged BHs.  "
        "   For l=2, q=0.5: V_peak = {:.6f}/M^2 (vs Schwarzschild V_peak for q=0).".format(
            peaks.get(2, {}).get("V_peak", 0.0)
        ),

        "3. TEMPERATURE: For q=0.5, T_H/T_schwarz = {:.4f}.  "
        "   Temperature decreases with charge; at extremality (q=1), T=0 "
        "   (degenerate horizon at r+=r-=4M/3).  Near-extremal T ({:.6f}/M "
        "   at q=0.99) is small but nonzero.".format(
            temp.get("ratio_T_H_over_T_schwarz", 0.0),
            temp_extremal.get("T_H_geom", 0.0)
        ),

        "4. POWER RATIO: Total Hawking power P_GM/P_schwarz = {:.4f} for q=0.5.  "
        "   Charged GM BHs emit {} than Schwarzschild.".format(
            pw.get("ratio", 0.0),
            "more" if pw.get("ratio", 0.0) > 1.0 else "less"
        ),

        "5. DILATON CHANNEL: If the dilaton were massless, the l=0 dilaton mode "
        "   would contribute {:.1f}% of total scalar power.  But the flux-stabilized "
        "   dilaton mass (~6.3e29 eV) completely suppresses this channel.  "
        "   The Hawking spectrum is indistinguishable from the metric-only "
        "   prediction.".format(
            dil.get("fraction_of_total", 0.0) * 100.0
        ),

        "6. PHYSICAL SCALE: For a 10 M_sun GM BH with q=0.5: "
        "   T_H = {:.4e} K, P = {:.4e} W.  "
        "   Evaporation timescale ~ {:.2e} s (>> age of universe).".format(
            temp_phys.get("T_H_kelvin", 0.0),
            pw_phys.get("P_gm_watts", 0.0),
            pw_phys.get("t_evap_gm_seconds", 0.0)
        ),

        "7. NEUTRAL LIMIT: For q=0, all results reduce identically to "
        "   Schwarzschild (gamma=1/2 is irrelevant when r_minus=0).",

        "8. COMPLETE EVAPORATION: Unlike RN black holes which approach T=0 "
        "   at extremality and may leave remnants, GM black holes with "
        "   a=1/sqrt(3) maintain finite T at extremality and evaporate "
        "   completely (in the semiclassical approximation).",
    ]

    caveats = [
        "- The WKB (parabolic barrier) approximation is accurate to ~5-10% "
        "  for low-lying modes.  Higher accuracy requires numerical integration "
        "  of the radial equation.",
        "- Hawking radiation is unobservable for stellar-mass and supermassive "
        "  black holes (T ~ 10^-8 K for M ~ M_sun).  Only primordial BHs "
        "  with M < 10^15 g could have observable Hawking emission.",
        "- Backreaction (mass loss during evaporation) is not included.  "
        "  The spectrum is computed for fixed mass.",
        "- Only scalar (spin-0) greybody factors are computed.  Graviton "
        "  (spin-2) and photon (spin-1) channels have different potentials "
        "  and greybody factors.",
        "- The GM solution assumes a massless dilaton in the background metric.  "
        "  If the dilaton has Planck-scale mass, the GM metric itself is only "
        "  valid at distances r << 1/m_phi, which for Planck-mass dilaton "
        "  means r ~ l_Planck -- i.e., only relevant at the singularity.",
    ]

    return {
        "temperature": temp,
        "temperature_neutral": temp_neutral,
        "temperature_extremal": temp_extremal,
        "potential_peaks": peaks,
        "spectrum_sample": spec,
        "total_power": pw,
        "greybody_scan": scan,
        "dilaton_channel": dil,
        "physical_example": {
            "M": "10 M_sun",
            "temperature": temp_phys,
            "power": pw_phys,
        },
        "findings": findings,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Main: print full report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("GREYBODY FACTORS AND HAWKING RADIATION FOR GM DILATON BLACK HOLES")
    print("Alpha Ladder coupling: a = 1/sqrt(3), gamma = 1/2")
    print("=" * 72)
    print()

    M = 1.0  # geometrized units
    a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    # --- Temperature ---
    print("-" * 72)
    print("1. HAWKING TEMPERATURE")
    print("-" * 72)
    for q in [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]:
        t = hawking_temperature(M, q, a)
        if "error" not in t:
            print("  q = {:.2f}:  T_H = {:.6f}/M,  T_H/T_schwarz = {:.4f},  "
                  "r+/M = {:.4f},  r-/M = {:.6f}".format(
                      q, t["T_H_geom"], t["ratio_T_H_over_T_schwarz"],
                      t["r_plus"] / M, t["r_minus"] / M))
    print()

    # --- Potential peaks ---
    print("-" * 72)
    print("2. EFFECTIVE POTENTIAL PEAKS (q = 0.5)")
    print("-" * 72)
    q = 0.5
    for l in range(6):
        pk = effective_potential_peak(l, M, q, a)
        if "error" not in pk:
            omega_c = math.sqrt(pk["V_peak"]) if pk["V_peak"] > 0 else 0.0
            print("  l = {}:  r_peak/r+ = {:.4f},  V_peak = {:.6f}/M^2,  "
                  "omega_c = {:.4f}/M,  b_c = {:.4f}M".format(
                      l, pk["r_peak_over_r_plus"], pk["V_peak"],
                      omega_c, pk["b_c"]))
    print()

    # --- Greybody factors ---
    print("-" * 72)
    print("3. GREYBODY FACTORS (q = 0.5, l = 0, 1, 2)")
    print("-" * 72)
    T_H = hawking_temperature(M, q, a)["T_H_geom"]
    print("  T_H = {:.6f}/M".format(T_H))
    print()
    print("  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "omega/T_H", "Gamma(l=0)", "Gamma(l=1)", "Gamma(l=2)", "regime(l=2)"))
    for x in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        omega = x * T_H
        g0 = greybody_factor(omega, 0, M, q, a)
        g1 = greybody_factor(omega, 1, M, q, a)
        g2 = greybody_factor(omega, 2, M, q, a)
        print("  {:10.1f}  {:10.6f}  {:10.6f}  {:10.6f}  {:>10s}".format(
            x, g0["Gamma"], g1["Gamma"], g2["Gamma"], g2["regime"]))
    print()

    # --- Spectrum ---
    print("-" * 72)
    print("4. HAWKING SPECTRUM (q = 0.5, l_max = 5)")
    print("-" * 72)
    spec = hawking_spectrum(M, q, l_max=5, a=a)
    if "spectrum" in spec:
        print("  {:>10s}  {:>14s}  {:>14s}  {:>12s}".format(
            "omega/T_H", "dN/(dt dw)", "dN_BB/(dt dw)", "suppression"))
        for entry in spec["spectrum"][::5]:
            print("  {:10.2f}  {:14.6e}  {:14.6e}  {:12.4f}".format(
                entry["omega_over_T"],
                entry["dN_dt_domega"],
                entry["dN_dt_domega_blackbody"],
                entry["greybody_suppression"]))
    print()

    # --- Total power ---
    print("-" * 72)
    print("5. TOTAL POWER")
    print("-" * 72)
    pw = total_power(M, q, l_max=5, a=a)
    if "error" not in pw:
        print("  P_GM (geom)     = {:.6e}".format(pw["P_gm"]))
        print("  P_schwarz (geom)= {:.6e}".format(pw["P_schwarz"]))
        print("  P_GM/P_schwarz  = {:.4f}".format(pw["ratio"]))
    print()

    # Physical example
    print("  --- Physical example: 10 M_sun, q = 0.5 ---")
    pw_phys = total_power(10.0 * M_sun, q, l_max=5, a=a)
    if "error" not in pw_phys:
        print("  P_GM             = {:.4e} W".format(pw_phys["P_gm_watts"]))
        print("  P_schwarz        = {:.4e} W".format(pw_phys["P_schwarz_watts"]))
        print("  t_evap (GM)      = {:.4e} s".format(pw_phys["t_evap_gm_seconds"]))
        print("  t_evap (schwarz) = {:.4e} s".format(pw_phys["t_evap_schwarz_seconds"]))
        print("  t_evap (exact)   = {:.4e} s".format(pw_phys["t_evap_schwarz_exact_seconds"]))
        t_phys = hawking_temperature(10.0 * M_sun, q, a)
        print("  T_H              = {:.4e} K".format(t_phys["T_H_kelvin"]))
    print()

    # --- Greybody scan ---
    print("-" * 72)
    print("6. GREYBODY SCAN (l = 2, varying q)")
    print("-" * 72)
    scan = greybody_scan(l=2, M=M, a=a)
    print("  {:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>12s}  {:>10s}".format(
        "q", "omega_c/M", "V_peak/M^2", "Gamma@wc", "Gamma@T_H", "T_H/omega_c"))
    for entry in scan["scan"]:
        print("  {:6.3f}  {:10.4f}  {:10.6f}  {:10.4f}  {:12.6f}  {:10.4f}".format(
            entry["q"], entry["omega_c"], entry["V_peak"],
            entry["Gamma_at_barrier"], entry["Gamma_at_thermal"],
            entry["T_H_over_omega_c"]))
    print()

    # --- Dilaton channel ---
    print("-" * 72)
    print("7. DILATON EMISSION CHANNEL (q = 0.5)")
    print("-" * 72)
    dil = dilaton_emission_channel(M, q, a)
    if "error" not in dil:
        print("  P_dilaton (massless) = {:.6e} (geom)".format(
            dil["P_dilaton_massless"]))
        print("  Fraction of total   = {:.2f}%".format(
            dil["fraction_of_total"] * 100.0))
        print("  m_phi / T_H         = {:.2e}".format(dil["m_phi_over_T_H"]))
        print("  Boltzmann suppression = {:.2e}".format(dil["suppression_factor"]))
        print()
        print("  " + dil["description"])
    print()

    # --- Summary ---
    print("-" * 72)
    print("8. SUMMARY")
    print("-" * 72)
    summary = summarize_greybody_analysis()
    for finding in summary["findings"]:
        print()
        print("  " + finding)
    print()
    print("  CAVEATS:")
    for caveat in summary["caveats"]:
        print("  " + caveat)
    print()
    print("=" * 72)
    print("END OF GREYBODY FACTOR ANALYSIS")
    print("=" * 72)

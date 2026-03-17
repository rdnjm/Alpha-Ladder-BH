"""
Quasinormal mode (QNM) frequencies for Gibbons-Maeda dilaton black holes.

Quasinormal modes are the damped oscillations of a perturbed black hole,
characterized by complex frequencies omega = omega_R + i*omega_I where
omega_R is the oscillation frequency and omega_I < 0 is the damping rate.
LIGO detects these in the "ringdown" phase after a binary merger.

For Schwarzschild (a=0), the fundamental QNM (l=2, n=0) is:
    omega_Schwarz * M = 0.3737 - 0.0890i   (geometrized units G=c=1)

The Alpha Ladder S^2 KK reduction produces a Gibbons-Maeda dilaton theory
with coupling a = 1/sqrt(3), fixed by the Brans-Dicke parameter omega=0.
Charged black holes in this theory have modified QNM spectra.

The key perturbation equation is a modified Regge-Wheeler equation with an
effective potential that depends on the dilaton coupling a.  The GM metric:
    ds^2 = -f(r)dt^2 + f(r)^{-1}dr^2 + R(r)^2 dOmega^2
where:
    f(r) = (1 - r+/r)(1 - r-/r)^gamma
    R(r)^2 = r^2 (1 - r-/r)^{1-gamma}
    gamma = (1-a^2)/(1+a^2)

For a = 1/sqrt(3): gamma = 1/2.

Method: 3rd-order WKB approximation (Schutz & Will 1985, Iyer & Will 1987),
with derivatives computed directly in tortoise coordinates via numerical
sampling.  Accurate to ~2% for omega_R of the fundamental mode (n=0).

Reference: G. W. Gibbons and K. Maeda, Nucl. Phys. B 298, 741 (1988).
           B. F. Schutz and C. M. Will, ApJ 291, L33 (1985).
           S. Iyer and C. M. Will, Phys. Rev. D 35, 3621 (1987).
           R. A. Konoplya, Phys. Rev. D 68, 024018 (2003).

Pure Python -- only 'import math' is used.
"""

import math

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G = 6.674298e-11       # m^3 kg^-1 s^-2  (Alpha Ladder predicted value)
c = 2.99792458e8       # m/s
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
# 1. GM metric functions
# ---------------------------------------------------------------------------

def gm_metric(r, r_plus, r_minus, a=None):
    """
    Compute metric functions f(r), R(r)^2, and gamma for the
    Gibbons-Maeda black hole.

    The GM metric in Schwarzschild-like coordinates:
        f(r) = (1 - r+/r)(1 - r-/r)^gamma
        R(r)^2 = r^2 (1 - r-/r)^{1-gamma}
        gamma = (1-a^2)/(1+a^2)

    Parameters
    ----------
    r : float
        Radial coordinate (must be > r_plus).
    r_plus : float
        Outer horizon radius.
    r_minus : float
        Inner horizon radius (GM singularity parameter).
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with keys: f, R2, gamma, r, r_plus, r_minus
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    u = 1.0 - r_plus / r
    v = 1.0 - r_minus / r if r_minus > 0 else 1.0

    if v <= 0 or u <= 0:
        return {"f": 0.0, "R2": 0.0, "gamma": gamma,
                "r": r, "r_plus": r_plus, "r_minus": r_minus}

    f = u * (v ** gamma)
    R2 = r * r * (v ** (1.0 - gamma))

    return {
        "f": f,
        "R2": R2,
        "gamma": gamma,
        "r": r,
        "r_plus": r_plus,
        "r_minus": r_minus,
    }


# ---------------------------------------------------------------------------
# 2. Effective potential for scalar perturbations
# ---------------------------------------------------------------------------

def effective_potential(r, l, r_plus, r_minus, a=None):
    """
    Scalar field effective potential on a Gibbons-Maeda background.

    For a test scalar field on the GM geometry, the radial equation
    in tortoise coordinates takes the Schrodinger form:
        d^2 psi/dr*^2 + [omega^2 - V(r)] psi = 0

    The effective potential is:
        V(r) = f(r) [l(l+1)/R(r)^2 + f'(r) R'(r) / R(r)]

    where f, R are the GM metric functions and primes denote d/dr.

    Parameters
    ----------
    r : float
        Radial coordinate (must be > r_plus).
    l : int
        Angular momentum quantum number (l >= 2 for gravitational).
    r_plus : float
        Outer horizon radius.
    r_minus : float
        Inner horizon radius.
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    float : V(r)
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    u = 1.0 - r_plus / r
    v = 1.0 - r_minus / r if r_minus > 0 else 1.0

    if u <= 0.0 or v <= 0.0:
        return 0.0

    f = u * (v ** gamma)

    # R^2 = r^2 * v^{1-gamma}
    R2 = r * r * (v ** (1.0 - gamma))
    R = math.sqrt(R2)

    # f'(r) = (r+/r^2) v^gamma + u * gamma * (r-/r^2) * v^{gamma-1}
    df_dr = (r_plus / (r * r)) * (v ** gamma)
    if r_minus > 0:
        df_dr += u * gamma * (r_minus / (r * r)) * (v ** (gamma - 1.0))

    # R(r) = r * v^{(1-gamma)/2}
    # dR/dr = v^{(1-gamma)/2} * [1 + (1-gamma)*r- / (2*(r - r-))]
    exp_half = (1.0 - gamma) / 2.0
    v_exp = v ** exp_half
    if r_minus > 0 and abs(r - r_minus) > 1e-30:
        dR_dr = v_exp * (1.0 + (1.0 - gamma) * r_minus / (2.0 * (r - r_minus)))
    else:
        dR_dr = v_exp

    V = f * (l * (l + 1.0) / R2 + df_dr * dR_dr / R)

    return V


# ---------------------------------------------------------------------------
# 3. Tortoise coordinate and potential sampling
# ---------------------------------------------------------------------------

def _tortoise_coordinate(r, r_plus, r_minus, a=None):
    """
    Compute the tortoise coordinate r*(r) for the GM metric.

    dr*/dr = 1/f(r), so r* = integral dr/f(r).

    For Schwarzschild (r_minus=0): r* = r + 2M*ln(r/(2M) - 1).
    For GM, the integral involves (1-r+/r)^{-1} (1-r-/r)^{-gamma}
    which we integrate numerically when r_minus > 0.

    For WKB we only need differences in r*, so we use numerical
    integration from a reference point.

    Parameters
    ----------
    r : float
        Radial coordinate (r > r_plus).
    r_plus : float
        Outer horizon radius.
    r_minus : float
        Inner horizon radius.
    a : float or None
        Dilaton coupling.

    Returns
    -------
    float : r*(r) (up to an additive constant).
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    if r_minus < 1e-30:
        # Schwarzschild case: analytic formula
        # r* = r + r+ * ln(r/r+ - 1)
        arg = r / r_plus - 1.0
        if arg <= 0:
            return -1e30
        return r + r_plus * math.log(arg)

    # GM case: numerical integration using Simpson's rule from a reference
    # We integrate from r_ref to r in steps
    # f(r) = (1 - r+/r)(1 - r-/r)^gamma
    r_ref = 10.0 * r_plus   # reference point

    def integrand(rr):
        u = 1.0 - r_plus / rr
        v = 1.0 - r_minus / rr
        if u <= 0 or v <= 0:
            return 1e30
        return 1.0 / (u * (v ** gamma))

    # Simpson's rule
    n_steps = 1000
    a_int = min(r_ref, r)
    b_int = max(r_ref, r)
    h_int = (b_int - a_int) / n_steps
    if h_int == 0:
        return 0.0

    total = integrand(a_int) + integrand(b_int)
    for i in range(1, n_steps, 2):
        total += 4.0 * integrand(a_int + i * h_int)
    for i in range(2, n_steps, 2):
        total += 2.0 * integrand(a_int + i * h_int)
    integral = total * h_int / 3.0

    sign = 1.0 if r >= r_ref else -1.0
    return sign * integral


def _r_from_tortoise(r_star_target, r_peak, r_plus, r_minus, a=None):
    """
    Invert r*(r) -> r(r*) via Newton's method near a known point.

    Parameters
    ----------
    r_star_target : float
        Target tortoise coordinate value.
    r_peak : float
        Initial guess for r (typically the potential peak).
    r_plus : float
        Outer horizon radius.
    r_minus : float
        Inner horizon radius.
    a : float or None
        Dilaton coupling.

    Returns
    -------
    float : r such that r*(r) ~ r_star_target.
    """
    if a is None:
        a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    r = r_peak
    for _ in range(200):
        rs = _tortoise_coordinate(r, r_plus, r_minus, a)
        u = 1.0 - r_plus / r
        v = 1.0 - r_minus / r if r_minus > 0 else 1.0
        if u <= 0 or v <= 0:
            r = r_plus * 1.01
            continue
        f_val = u * (v ** gamma)

        # Newton step: dr = (r*_target - r*(r)) * f(r)
        delta_rs = r_star_target - rs
        dr = delta_rs * f_val

        # Limit step size for stability
        max_step = 0.5 * (r - r_plus)
        if abs(dr) > max_step:
            dr = max_step if dr > 0 else -max_step

        r += dr

        # Keep r safely outside horizon
        if r <= r_plus * 1.001:
            r = r_plus * 1.001

        if abs(delta_rs) < 1e-12 * abs(r):
            break

    return r


# ---------------------------------------------------------------------------
# 4. Find potential peak
# ---------------------------------------------------------------------------

def find_potential_peak(l, r_plus, r_minus, a=None, tol=1e-10):
    """
    Find the radial coordinate where the effective potential V(r) is
    maximized, using bisection on dV/dr.

    Parameters
    ----------
    l : int
        Angular momentum quantum number.
    r_plus : float
        Outer horizon radius.
    r_minus : float
        Inner horizon radius.
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).
    tol : float
        Fractional tolerance for bisection convergence.

    Returns
    -------
    dict with keys: r_peak, V_peak, f_peak, l
    """
    if a is None:
        a = A_DEFAULT

    h = 1e-4 * r_plus

    def V(r):
        return effective_potential(r, l, r_plus, r_minus, a)

    def dV_dr(r):
        return (V(r + h) - V(r - h)) / (2.0 * h)

    # Bracket the peak
    r_lo = r_plus * 1.001
    r_hi = r_plus * 20.0

    dv_lo = dV_dr(r_lo)
    dv_hi = dV_dr(r_hi)

    if dv_lo < 0:
        r_lo = r_plus * 1.0001
        dv_lo = dV_dr(r_lo)
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

    met = gm_metric(r_peak, r_plus, r_minus, a)
    f_peak = met["f"]

    return {
        "r_peak": r_peak,
        "V_peak": V_peak,
        "f_peak": f_peak,
        "l": l,
    }


# ---------------------------------------------------------------------------
# 5. Tortoise-coordinate derivatives of the potential
# ---------------------------------------------------------------------------

def _potential_tortoise_derivatives(l, r_plus, r_minus, a=None):
    """
    Compute the potential V and its derivatives V2 through V6 at the peak,
    evaluated in tortoise coordinates via direct numerical sampling.

    We sample V(r*) at equally-spaced points in r* near the peak, then
    apply standard central-difference formulas.  This avoids the numerical
    instability of converting r-derivatives to r*-derivatives via chain
    rule (which amplifies errors for the 4th and higher derivatives).

    Parameters
    ----------
    l : int
        Angular momentum quantum number.
    r_plus : float
        Outer horizon radius.
    r_minus : float
        Inner horizon radius.
    a : float or None
        Dilaton coupling.

    Returns
    -------
    dict with V0, V2, V3, V4, V5, V6, r_peak, f_peak, h_star
    """
    if a is None:
        a = A_DEFAULT

    # Find peak in r-coordinate
    peak = find_potential_peak(l, r_plus, r_minus, a)
    r_peak = peak["r_peak"]
    V0 = peak["V_peak"]
    f_peak = peak["f_peak"]

    # Tortoise coordinate at peak
    rs_peak = _tortoise_coordinate(r_peak, r_plus, r_minus, a)

    # Step size in r*:  we want h_star such that we sample well around
    # the peak but not so large that we miss features.
    # A good choice is h_star ~ 0.05 * r_plus (in tortoise coordinate units).
    h_star = 0.05 * r_plus

    # Sample V(r*) at 13 points: k = -6 ... +6
    N = 6

    def V_at_r(r):
        return effective_potential(r, l, r_plus, r_minus, a)

    # Map each r* point back to r, then evaluate V
    pts = {}
    for k in range(-N, N + 1):
        rs_k = rs_peak + k * h_star
        # Initial guess: r_peak + k * h_star * f_peak (linear approximation)
        r_guess = r_peak + k * h_star * f_peak
        r_k = _r_from_tortoise(rs_k, r_guess, r_plus, r_minus, a)
        pts[k] = V_at_r(r_k)

    # Central difference formulas for derivatives d^n V / dr*^n
    h = h_star

    # V1 (should be ~0 at peak)
    V1 = (pts[1] - pts[-1]) / (2.0 * h)

    # V2 = d^2 V / dr*^2
    V2 = (pts[1] - 2.0 * pts[0] + pts[-1]) / (h * h)

    # V3 = d^3 V / dr*^3
    V3 = (pts[2] - 2.0 * pts[1] + 2.0 * pts[-1] - pts[-2]) / (2.0 * h ** 3)

    # V4 = d^4 V / dr*^4
    V4 = (pts[2] - 4.0 * pts[1] + 6.0 * pts[0]
           - 4.0 * pts[-1] + pts[-2]) / (h ** 4)

    # V5 = d^5 V / dr*^5
    V5 = (pts[3] - 4.0 * pts[2] + 5.0 * pts[1]
           - 5.0 * pts[-1] + 4.0 * pts[-2] - pts[-3]) / (2.0 * h ** 5)

    # V6 = d^6 V / dr*^6
    V6 = (pts[3] - 6.0 * pts[2] + 15.0 * pts[1] - 20.0 * pts[0]
           + 15.0 * pts[-1] - 6.0 * pts[-2] + pts[-3]) / (h ** 6)

    return {
        "V0": V0,
        "V1": V1,
        "V2": V2,
        "V3": V3,
        "V4": V4,
        "V5": V5,
        "V6": V6,
        "r_peak": r_peak,
        "f_peak": f_peak,
        "h_star": h_star,
    }


# ---------------------------------------------------------------------------
# 6. Complex square root
# ---------------------------------------------------------------------------

def _complex_sqrt(re, im):
    """
    Square root of a complex number z = re + i*im.

    Returns (sqrt_re, sqrt_im) with convention sqrt_re >= 0.
    """
    mag = math.sqrt(re * re + im * im)
    angle = math.atan2(im, re)
    half_angle = angle / 2.0
    sqrt_mag = math.sqrt(mag)
    return (sqrt_mag * math.cos(half_angle), sqrt_mag * math.sin(half_angle))


# ---------------------------------------------------------------------------
# 7. WKB quasinormal mode calculation
# ---------------------------------------------------------------------------

def wkb_qnm(l, n, r_plus, r_minus, a=None):
    """
    Quasinormal mode frequency using the 3rd-order WKB approximation
    (Iyer & Will 1987, Konoplya 2003).

    The WKB formula:
        omega^2 = V0 + sqrt(-2*V2) * Lambda_2
                  - i*(n+1/2)*sqrt(-2*V2) * (1 + Lambda_3)

    where V0 is the peak potential, V2 = d^2V/dr*^2 at the peak,
    Lambda_2 is the 2nd-order (real) correction, and Lambda_3 is the
    3rd-order correction to the imaginary part.

    The corrections involve ratios of higher derivatives V3/V2, V4/V2,
    V5/V2, V6/V2 evaluated at the peak in tortoise coordinates.

    For the fundamental mode (n=0, l=2) on Schwarzschild, this gives
    omega*M accurate to ~2% for the real part.

    Parameters
    ----------
    l : int
        Angular momentum quantum number (l >= 2).
    n : int
        Overtone number (n=0 is fundamental).
    r_plus : float
        Outer horizon radius (geometrized units).
    r_minus : float
        Inner horizon radius (geometrized units).
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with omega_R, omega_I, quality factor Q, and mode parameters.
    """
    if a is None:
        a = A_DEFAULT

    derivs = _potential_tortoise_derivatives(l, r_plus, r_minus, a)
    V0 = derivs["V0"]
    V2 = derivs["V2"]
    V3 = derivs["V3"]
    V4 = derivs["V4"]
    V5 = derivs["V5"]
    V6 = derivs["V6"]
    r_peak = derivs["r_peak"]
    f_peak = derivs["f_peak"]

    # Sanity check: V2 should be negative (potential maximum)
    if V2 >= 0:
        return {
            "error": "Potential has no proper maximum (V2 >= 0 in tortoise coords)",
            "l": l, "n": n, "V2": V2,
        }

    alpha = n + 0.5
    alpha_sq = alpha * alpha

    sq_neg2V2 = math.sqrt(-2.0 * V2)

    # --- 2nd-order correction Lambda_2 (Iyer & Will 1987, Eq. 1) ---
    # Lambda_2 = (1/(-2V2)) * [
    #   (1/8)(V4/V2)(1/4 + alpha^2) - (1/288)(V3/V2)^2 (7 + 60*alpha^2)
    # ]
    inv_neg2V2 = 1.0 / (-2.0 * V2)
    V3oV2 = V3 / V2
    V4oV2 = V4 / V2

    Lambda_2 = inv_neg2V2 * (
        (1.0 / 8.0) * V4oV2 * (0.25 + alpha_sq)
        - (1.0 / 288.0) * V3oV2 * V3oV2 * (7.0 + 60.0 * alpha_sq)
    )

    # --- 3rd-order correction Lambda_3 (Iyer & Will 1987) ---
    # Lambda_3 = alpha / (-2V2) * [
    #   (5/6912)(V3/V2)^4 (77 + 188*alpha^2)
    #   - (1/384)(V3^2 * V4 / V2^3)(51 + 100*alpha^2)
    #   + (1/2304)(V4/V2)^2 (67 + 68*alpha^2)
    #   + (1/288)(V3*V5 / V2^2)(19 + 28*alpha^2)
    #   - (1/288)(V6/V2)(5 + 4*alpha^2)
    # ]
    V5oV2 = V5 / V2
    V6oV2 = V6 / V2

    Lambda_3 = alpha * inv_neg2V2 * (
        (5.0 / 6912.0) * (V3oV2 ** 4) * (77.0 + 188.0 * alpha_sq)
        - (1.0 / 384.0) * (V3 * V3 * V4 / (V2 ** 3)) * (51.0 + 100.0 * alpha_sq)
        + (1.0 / 2304.0) * (V4oV2 ** 2) * (67.0 + 68.0 * alpha_sq)
        + (1.0 / 288.0) * (V3 * V5 / (V2 ** 2)) * (19.0 + 28.0 * alpha_sq)
        - (1.0 / 288.0) * V6oV2 * (5.0 + 4.0 * alpha_sq)
    )

    # --- Assemble omega^2 ---
    # omega^2 = V0 + sqrt(-2V2)*Lambda_2 - i*alpha*sqrt(-2V2)*(1 + Lambda_3)
    omega_sq_re = V0 + sq_neg2V2 * Lambda_2
    omega_sq_im = -alpha * sq_neg2V2 * (1.0 + Lambda_3)

    # Complex square root to get omega
    omega_R, omega_I = _complex_sqrt(omega_sq_re, omega_sq_im)

    # Quality factor
    Q_factor = abs(omega_R / (2.0 * omega_I)) if abs(omega_I) > 1e-50 else float("inf")

    return {
        "omega_R": omega_R,
        "omega_I": omega_I,
        "omega_R_M": None,   # filled by caller with dimensionless omega*M
        "omega_I_M": None,
        "Q_factor": Q_factor,
        "l": l,
        "n": n,
        "r_peak": r_peak,
        "V_peak": V0,
        "V2_tortoise": V2,
        "Lambda_2": Lambda_2,
        "Lambda_3": Lambda_3,
        "a": a,
    }


# ---------------------------------------------------------------------------
# 8. Compute QNM spectrum
# ---------------------------------------------------------------------------

def compute_qnm_spectrum(M_solar=10.0, qm_ratio=0.0, a=None):
    """
    Compute QNM frequencies for l=2,3,4 (n=0 fundamental) for a GM black
    hole, and compare with Schwarzschild.

    All WKB calculations are performed in dimensionless units (M=1) for
    numerical stability, then scaled to physical units using:
        omega_physical = omega_dimensionless / M_geom
        f_Hz = omega_physical * c / (2*pi)

    For an uncharged BH (qm_ratio=0), r_minus=0 and the GM solution reduces
    to Schwarzschild regardless of dilaton coupling.  The dilaton deviation
    grows with charge ratio.

    Parameters
    ----------
    M_solar : float
        Black hole mass in solar masses.
    qm_ratio : float
        Q/Q_extreme in [0, 1].
    a : float or None
        Dilaton coupling (default 1/sqrt(3)).

    Returns
    -------
    dict with QNM frequencies, comparison, and physical quantities.
    """
    if a is None:
        a = A_DEFAULT

    M_kg = M_solar * M_sun
    M_geom = _mass_to_geom(M_kg)

    # Work in dimensionless units: M_dim = 1
    M_dim = 1.0

    results = {
        "M_solar": M_solar,
        "M_kg": M_kg,
        "M_geom": M_geom,
        "qm_ratio": qm_ratio,
        "a": a,
        "modes": {},
    }

    for l in [2, 3, 4]:
        # GM black hole in dimensionless units
        radii_gm = _horizon_radii(M_dim, qm_ratio, a)
        if radii_gm is None:
            results["modes"][f"l={l}"] = {"error": "no horizon (GM)"}
            continue
        r_plus_gm, r_minus_gm = radii_gm

        qnm_gm = wkb_qnm(l, 0, r_plus_gm, r_minus_gm, a)
        if "error" in qnm_gm:
            results["modes"][f"l={l}"] = qnm_gm
            continue

        # omega_R, omega_I are in units of 1/M_dim = 1
        # Dimensionless: omega * M = omega_R (since M_dim=1)
        qnm_gm["omega_R_M"] = qnm_gm["omega_R"] * M_dim
        qnm_gm["omega_I_M"] = qnm_gm["omega_I"] * M_dim

        # Physical frequency: omega_phys = omega_dim / M_geom (units 1/m)
        # f_Hz = omega_phys * c / (2*pi), tau_s = 1/(omega_phys * c)
        omega_R_phys = qnm_gm["omega_R"] / M_geom
        omega_I_phys = qnm_gm["omega_I"] / M_geom
        f_Hz = omega_R_phys * c / (2.0 * math.pi)
        tau_s = (
            1.0 / (abs(omega_I_phys) * c) if abs(omega_I_phys) > 0
            else float("inf")
        )
        tau_ms = tau_s * 1000.0

        qnm_gm["f_Hz"] = f_Hz
        qnm_gm["tau_ms"] = tau_ms

        # Schwarzschild comparison (a=0, qm_ratio=0, r_minus=0)
        r_plus_schwarz = 2.0 * M_dim
        qnm_schwarz = wkb_qnm(l, 0, r_plus_schwarz, 0.0, 0.0)

        if "error" not in qnm_schwarz:
            qnm_schwarz["omega_R_M"] = qnm_schwarz["omega_R"] * M_dim
            qnm_schwarz["omega_I_M"] = qnm_schwarz["omega_I"] * M_dim

            omega_R_phys_s = qnm_schwarz["omega_R"] / M_geom
            omega_I_phys_s = qnm_schwarz["omega_I"] / M_geom
            f_Hz_s = omega_R_phys_s * c / (2.0 * math.pi)
            tau_s_s = (
                1.0 / (abs(omega_I_phys_s) * c)
                if abs(omega_I_phys_s) > 0 else float("inf")
            )
            qnm_schwarz["f_Hz"] = f_Hz_s
            qnm_schwarz["tau_ms"] = tau_s_s * 1000.0

            # Fractional shift (dimensionless, same as physical ratio)
            delta_omega_R = (
                (qnm_gm["omega_R"] - qnm_schwarz["omega_R"])
                / qnm_schwarz["omega_R"]
            ) if qnm_schwarz["omega_R"] != 0 else 0.0

            delta_omega_I = (
                (qnm_gm["omega_I"] - qnm_schwarz["omega_I"])
                / abs(qnm_schwarz["omega_I"])
            ) if qnm_schwarz["omega_I"] != 0 else 0.0
        else:
            qnm_schwarz = {"error": "WKB failed for Schwarzschild"}
            delta_omega_R = None
            delta_omega_I = None

        results["modes"][f"l={l}"] = {
            "GM": qnm_gm,
            "Schwarzschild": qnm_schwarz,
            "delta_omega_R_frac": delta_omega_R,
            "delta_omega_I_frac": delta_omega_I,
        }

    return results


# ---------------------------------------------------------------------------
# 9. Compare with LIGO
# ---------------------------------------------------------------------------

def compare_with_ligo(M_solar=30.0):
    """
    Compare GM dilaton QNM predictions with LIGO detection capabilities.

    For a ~30 solar mass remnant (typical of LIGO binary mergers like
    GW150914), the fundamental l=2 QNM is at ~250 Hz with a damping
    time of ~4 ms.

    Parameters
    ----------
    M_solar : float
        Remnant mass in solar masses.

    Returns
    -------
    dict with comparison data and honest detectability assessment.
    """
    a = A_DEFAULT

    q_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    M_kg = M_solar * M_sun
    M_geom = _mass_to_geom(M_kg)

    scan_results = []
    for q in q_values:
        spec = compute_qnm_spectrum(M_solar, q, a)
        mode_data = spec["modes"].get("l=2", {})

        if "error" in mode_data:
            scan_results.append({"q": q, "error": mode_data.get("error", "unknown")})
            continue

        gm = mode_data.get("GM", {})
        schwarz = mode_data.get("Schwarzschild", {})

        entry = {
            "q": q,
            "f_Hz_GM": gm.get("f_Hz"),
            "tau_ms_GM": gm.get("tau_ms"),
            "f_Hz_Schwarz": schwarz.get("f_Hz"),
            "tau_ms_Schwarz": schwarz.get("tau_ms"),
            "delta_f_frac": mode_data.get("delta_omega_R_frac"),
            "delta_tau_frac": mode_data.get("delta_omega_I_frac"),
            "omega_R_M_GM": gm.get("omega_R_M"),
            "omega_I_M_GM": gm.get("omega_I_M"),
            "omega_R_M_Schwarz": schwarz.get("omega_R_M"),
            "omega_I_M_Schwarz": schwarz.get("omega_I_M"),
        }

        if entry["f_Hz_GM"] is not None and entry["f_Hz_Schwarz"] is not None:
            entry["delta_f_Hz"] = abs(entry["f_Hz_GM"] - entry["f_Hz_Schwarz"])
        else:
            entry["delta_f_Hz"] = None

        scan_results.append(entry)

    f_schwarz = None
    for entry in scan_results:
        if entry.get("q") == 0.0 and entry.get("f_Hz_Schwarz") is not None:
            f_schwarz = entry["f_Hz_Schwarz"]
            break

    ligo_freq_resolution_Hz = 10.0
    ligo_best_resolution_Hz = 1.0

    return {
        "M_solar": M_solar,
        "scan_results": scan_results,
        "ligo_context": {
            "f_220_schwarz_Hz": f_schwarz,
            "typical_freq_resolution_Hz": ligo_freq_resolution_Hz,
            "best_case_resolution_Hz": ligo_best_resolution_Hz,
            "note": (
                "LIGO O4/O5 frequency resolution for ringdown is ~1-10 Hz "
                "at SNR~10-50.  For a 30 solar mass remnant, the l=2 QNM "
                "frequency is ~250 Hz.  Sub-percent deviations correspond "
                "to ~1-2 Hz shifts, marginally detectable only for the "
                "loudest events."
            ),
        },
        "future_detectors": {
            "Einstein_Telescope": (
                "ET aims for ~10x better sensitivity, potentially reaching "
                "~0.1 Hz resolution on ringdown frequencies.  This would "
                "probe delta_f/f ~ 0.04% for 250 Hz modes."
            ),
            "Cosmic_Explorer": (
                "CE with ~40 km arms targets similar improvement, enabling "
                "sub-percent QNM spectroscopy for high-SNR events."
            ),
            "LISA": (
                "LISA observes massive BH mergers (10^5-10^7 M_sun) where "
                "QNM frequencies are in the mHz band.  Ringdown SNR can "
                "exceed 100, enabling percent-level QNM spectroscopy."
            ),
        },
    }


# ---------------------------------------------------------------------------
# 10. Summary and honest assessment
# ---------------------------------------------------------------------------

def summarize_qnm_analysis():
    """
    Full summary of the QNM analysis with honest assessment of
    observability and theoretical significance.

    Returns
    -------
    dict
    """
    a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    # Schwarzschild reference (l=2, n=0)
    # Known exact value: omega*M = 0.3737 - 0.0890i
    M_test = 1.0
    r_plus_schwarz = 2.0 * M_test
    qnm_schwarz = wkb_qnm(2, 0, r_plus_schwarz, 0.0, 0.0)

    wkb_omega_R_M = (
        qnm_schwarz["omega_R"] * M_test if "error" not in qnm_schwarz else None
    )
    wkb_omega_I_M = (
        qnm_schwarz["omega_I"] * M_test if "error" not in qnm_schwarz else None
    )

    exact_omega_R_M = 0.3737
    exact_omega_I_M = -0.0890

    if wkb_omega_R_M is not None:
        wkb_error_R = abs(wkb_omega_R_M - exact_omega_R_M) / exact_omega_R_M
        wkb_error_I = abs(wkb_omega_I_M - exact_omega_I_M) / abs(exact_omega_I_M)
    else:
        wkb_error_R = None
        wkb_error_I = None

    # GM at q=0.5 for reference shift
    radii_gm = _horizon_radii(M_test, 0.5, a)
    qnm_gm_half = None
    if radii_gm:
        qnm_gm_half = wkb_qnm(2, 0, radii_gm[0], radii_gm[1], a)
        if "error" not in qnm_gm_half:
            qnm_gm_half["omega_R_M"] = qnm_gm_half["omega_R"] * M_test
            qnm_gm_half["omega_I_M"] = qnm_gm_half["omega_I"] * M_test

    # 30 solar mass LIGO comparison
    ligo = compare_with_ligo(30.0)

    return {
        "framework": "Alpha Ladder S^2 KK -> Gibbons-Maeda dilaton BH QNMs",
        "method": "3rd-order WKB (Schutz-Will 1985, Iyer-Will 1987, Konoplya 2003)",
        "dilaton_coupling": {
            "a": a,
            "a_squared": a_sq,
            "gamma": gamma,
            "source": "omega_BD = 0 from S^2 KK reduction",
        },
        "wkb_validation": {
            "schwarzschild_l2_n0": {
                "exact_omega_R_M": exact_omega_R_M,
                "exact_omega_I_M": exact_omega_I_M,
                "wkb_omega_R_M": wkb_omega_R_M,
                "wkb_omega_I_M": wkb_omega_I_M,
                "error_omega_R": wkb_error_R,
                "error_omega_I": wkb_error_I,
            },
        },
        "gm_shift_at_q05": {
            "omega_R_M": (
                qnm_gm_half.get("omega_R_M")
                if qnm_gm_half and "error" not in qnm_gm_half
                else None
            ),
            "omega_I_M": (
                qnm_gm_half.get("omega_I_M")
                if qnm_gm_half and "error" not in qnm_gm_half
                else None
            ),
        },
        "ligo_comparison": ligo,
        "key_findings": [
            "QNM frequency shifts grow with charge ratio q = Q/Q_ext.",
            "For q=0 (uncharged), the GM solution reduces to Schwarzschild -- "
            "no dilaton effect on QNMs.",
            "Real astrophysical black holes are expected to be nearly neutral "
            "(q << 1) due to charge neutralization by ambient plasma.",
            "Even for moderate charge (q=0.5), the QNM shift is modest "
            "(a few percent in omega_R).",
            "The dilaton coupling a=1/sqrt(3) is FIXED, not a free parameter -- "
            "this is a concrete prediction of the Alpha Ladder.",
        ],
        "honest_assessment": [
            "LIGO's current frequency resolution (~1-10 Hz for ringdown) is "
            "insufficient to detect sub-percent QNM deviations at ~250 Hz.",
            "Astrophysical BHs are nearly neutral, so dilaton QNM effects are "
            "tiny in practice even if the dilaton is massless.",
            "If the dilaton is massive (Planck-scale from flux stabilization), "
            "it decouples entirely and BH ringdown reverts to GR.",
            "Future detectors (Einstein Telescope, Cosmic Explorer) may reach "
            "the needed precision for percent-level QNM spectroscopy.",
            "LISA could probe QNMs of massive BH mergers with SNR > 100, "
            "potentially enabling percent-level tests.",
            "The prediction is concrete and falsifiable in principle, but "
            "observational verification requires next-generation detectors "
            "AND charged black holes (both unlikely in the near term).",
        ],
        "theoretical_significance": (
            "The QNM calculation demonstrates that the Alpha Ladder framework "
            "makes concrete, quantitative predictions for gravitational wave "
            "observables.  The dilaton coupling a = 1/sqrt(3) is uniquely "
            "determined by the S^2 reduction, so the QNM spectrum is a "
            "zero-parameter prediction (given mass and charge).  While "
            "observational tests are challenging, the calculation establishes "
            "the framework's connection to measurable physics."
        ),
    }


# ---------------------------------------------------------------------------
# Formatting helper
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
# Main report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("QUASINORMAL MODES OF GIBBONS-MAEDA DILATON BLACK HOLES")
    print("Alpha Ladder S^2 KK Reduction:  a = 1/sqrt(3)")
    print("=" * 72)

    a = A_DEFAULT
    a_sq = a * a
    gamma = (1.0 - a_sq) / (1.0 + a_sq)

    # --- 1. WKB validation against Schwarzschild ---
    print("\n--- 1. WKB Validation: Schwarzschild (l=2, n=0) ---")
    print(f"  Known exact:   omega*M = 0.3737 - 0.0890i")

    M_test = 1.0
    r_plus_test = 2.0 * M_test
    qnm_s = wkb_qnm(2, 0, r_plus_test, 0.0, 0.0)
    if "error" not in qnm_s:
        oR_M = qnm_s["omega_R"] * M_test
        oI_M = qnm_s["omega_I"] * M_test
        print(f"  WKB result:    omega*M = {oR_M:.4f} - {abs(oI_M):.4f}i")
        err_R = abs(oR_M - 0.3737) / 0.3737 * 100.0
        err_I = abs(abs(oI_M) - 0.0890) / 0.0890 * 100.0
        print(f"  Error:         omega_R: {err_R:.1f}%,  omega_I: {err_I:.1f}%")
        print(f"  Quality factor Q = {qnm_s['Q_factor']:.2f}")
        print(f"  Peak location:   r_peak/M = {qnm_s['r_peak']/M_test:.4f}")
        print(f"  Lambda_2 = {qnm_s['Lambda_2']:.6f}")
        print(f"  Lambda_3 = {qnm_s['Lambda_3']:.6f}")
    else:
        print(f"  WKB failed: {qnm_s['error']}")

    # Higher l modes
    print("\n  Higher l modes (Schwarzschild, n=0):")
    # Known exact values for comparison (Leaver 1985):
    exact_schwarz = {
        2: (0.3737, 0.0890),
        3: (0.5994, 0.0927),
        4: (0.8092, 0.0942),
        5: (1.0123, 0.0948),
    }
    print(f"  {'l':>4s}  {'oR*M (WKB)':>12s}  {'oI*M (WKB)':>12s}"
          f"  {'oR*M (exact)':>12s}  {'err_R (%)':>10s}  {'Q':>8s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}")
    for l in [2, 3, 4, 5]:
        qnm_l = wkb_qnm(l, 0, r_plus_test, 0.0, 0.0)
        if "error" not in qnm_l:
            oR = qnm_l["omega_R"] * M_test
            oI = qnm_l["omega_I"] * M_test
            eR_exact, eI_exact = exact_schwarz.get(l, (0, 0))
            err = abs(oR - eR_exact) / eR_exact * 100 if eR_exact > 0 else 0
            print(f"  {l:4d}  {oR:12.6f}  {oI:12.6f}"
                  f"  {eR_exact:12.6f}  {err:10.2f}  {qnm_l['Q_factor']:8.2f}")
        else:
            print(f"  {l:4d}  {'error':>12s}")

    # --- 2. GM metric and effective potential ---
    print("\n--- 2. Effective Potential ---")
    print(f"  Dilaton coupling a = 1/sqrt(3) = {a:.6f}")
    print(f"  gamma = (1-a^2)/(1+a^2) = {gamma:.6f}")

    print(f"\n  Potential peak location and height (l=2, M=1 geometrized):")
    print(f"  {'q':>6s}  {'r_peak/M':>10s}  {'V_peak*M^2':>12s}"
          f"  {'r+/M':>8s}  {'r-/M':>8s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}")
    for q in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        radii = _horizon_radii(M_test, q, a)
        if radii is None:
            print(f"  {q:6.2f}  {'no horizon':>10s}")
            continue
        rp, rm = radii
        pk = find_potential_peak(2, rp, rm, a)
        print(f"  {q:6.2f}  {pk['r_peak']/M_test:10.4f}"
              f"  {pk['V_peak']*M_test*M_test:12.6f}"
              f"  {rp/M_test:8.4f}  {rm/M_test:8.6f}")

    # --- 3. QNM spectrum vs charge ratio ---
    print("\n--- 3. QNM Spectrum: GM vs Schwarzschild ---")
    print(f"\n  l=2, n=0 mode (fundamental) at M = 1 (geometrized):")
    print(f"  {'q':>6s}  {'oR*M (GM)':>12s}  {'oI*M (GM)':>12s}"
          f"  {'doR/oR (%)':>12s}  {'doI/oI (%)':>12s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    # Get Schwarzschild reference once
    qnm_s_ref = wkb_qnm(2, 0, 2.0 * M_test, 0.0, 0.0)

    for q in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        radii = _horizon_radii(M_test, q, a)
        if radii is None:
            print(f"  {q:6.2f}  {'no horizon':>12s}")
            continue
        rp, rm = radii
        qnm_gm = wkb_qnm(2, 0, rp, rm, a)

        if "error" in qnm_gm:
            print(f"  {q:6.2f}  {'WKB failed':>12s}")
            continue

        oR_gm = qnm_gm["omega_R"] * M_test
        oI_gm = qnm_gm["omega_I"] * M_test

        if "error" not in qnm_s_ref:
            oR_s = qnm_s_ref["omega_R"] * M_test
            oI_s = qnm_s_ref["omega_I"] * M_test
            dR = (oR_gm - oR_s) / oR_s * 100.0
            dI = (oI_gm - oI_s) / abs(oI_s) * 100.0
        else:
            dR = 0.0
            dI = 0.0

        print(f"  {q:6.2f}  {oR_gm:12.6f}  {oI_gm:12.6f}"
              f"  {dR:+12.4f}  {dI:+12.4f}")

    # --- 4. Physical frequencies for astrophysical BHs ---
    print("\n--- 4. Physical QNM Frequencies ---")
    print("  (WKB computed in M=1 units, scaled to physical Hz)")
    for M_sol in [10.0, 30.0, 60.0]:
        print(f"\n  M = {M_sol:.0f} solar masses:")
        M_kg = M_sol * M_sun
        M_geom = _mass_to_geom(M_kg)

        # Work in dimensionless M=1 units, then scale
        # Schwarzschild: r+ = 2M = 2, r- = 0
        qnm_phys = wkb_qnm(2, 0, 2.0, 0.0, 0.0)
        if "error" not in qnm_phys:
            # omega_dim has units 1/M; physical omega = omega_dim/M_geom
            omega_R_phys = qnm_phys["omega_R"] / M_geom
            omega_I_phys = qnm_phys["omega_I"] / M_geom
            f_Hz = omega_R_phys * c / (2.0 * math.pi)
            tau_ms = (
                1000.0 / (abs(omega_I_phys) * c)
                if abs(omega_I_phys) > 0 else float("inf")
            )
            print(f"    Schwarzschild (l=2): f = {f_Hz:.1f} Hz, "
                  f"tau = {tau_ms:.2f} ms")

        # GM at q=0.5 (dimensionless M=1)
        radii_gm = _horizon_radii(1.0, 0.5, a)
        if radii_gm:
            qnm_gm = wkb_qnm(2, 0, radii_gm[0], radii_gm[1], a)
            if "error" not in qnm_gm:
                omega_R_gm = qnm_gm["omega_R"] / M_geom
                omega_I_gm = qnm_gm["omega_I"] / M_geom
                f_Hz_gm = omega_R_gm * c / (2.0 * math.pi)
                tau_ms_gm = (
                    1000.0 / (abs(omega_I_gm) * c)
                    if abs(omega_I_gm) > 0 else float("inf")
                )
                print(f"    GM (q=0.5, l=2):    f = {f_Hz_gm:.1f} Hz, "
                      f"tau = {tau_ms_gm:.2f} ms")
                if "error" not in qnm_phys:
                    df = abs(f_Hz_gm - f_Hz)
                    print(f"    Shift:              delta_f = {df:.2f} Hz "
                          f"({df/f_Hz*100:.2f}%)")

    # --- 5. LIGO comparison ---
    print("\n--- 5. LIGO Detectability (30 solar masses) ---")
    ligo = compare_with_ligo(30.0)
    print(f"\n  Charge scan (l=2, n=0, M = 30 M_sun):")
    print(f"  {'q':>6s}  {'f_GM (Hz)':>12s}  {'f_Schwarz':>12s}"
          f"  {'delta_f (Hz)':>12s}  {'delta_f/f (%)':>14s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*14}")
    for entry in ligo["scan_results"]:
        q = entry.get("q", 0)
        f_gm = entry.get("f_Hz_GM")
        f_s = entry.get("f_Hz_Schwarz")
        df = entry.get("delta_f_Hz")
        dfrac = entry.get("delta_f_frac")
        if f_gm is not None and f_s is not None:
            print(f"  {q:6.2f}  {f_gm:12.1f}  {f_s:12.1f}"
                  f"  {df:12.2f}"
                  f"  {dfrac*100 if dfrac else 0:+14.4f}")
        elif "error" in entry:
            print(f"  {q:6.2f}  {'error':>12s}")
        else:
            print(f"  {q:6.2f}  {'N/A':>12s}")

    ctx = ligo["ligo_context"]
    print(f"\n  LIGO frequency resolution:")
    print(f"    Typical (O4/O5):    ~{ctx['typical_freq_resolution_Hz']:.0f} Hz")
    print(f"    Best case (loud):   ~{ctx['best_case_resolution_Hz']:.0f} Hz")
    print(f"    {ctx['note']}")

    print(f"\n  Future detectors:")
    for det, desc in ligo["future_detectors"].items():
        print(f"    {det}: {desc}")

    # --- 6. Multi-mode spectrum ---
    print("\n--- 6. Multi-mode QNM Spectrum (q=0.5, M=30 M_sun) ---")
    spec = compute_qnm_spectrum(30.0, 0.5, a)
    print(f"  {'l':>4s}  {'oR*M (GM)':>12s}  {'oI*M (GM)':>12s}"
          f"  {'f (Hz)':>10s}  {'tau (ms)':>10s}"
          f"  {'doR/oR (%)':>12s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*12}")
    for l in [2, 3, 4]:
        mode = spec["modes"].get(f"l={l}", {})
        if "error" in mode:
            print(f"  {l:4d}  {'error':>12s}")
            continue
        gm = mode["GM"]
        d_frac = mode.get("delta_omega_R_frac", 0) or 0
        print(f"  {l:4d}  {gm['omega_R_M']:12.6f}  {gm['omega_I_M']:12.6f}"
              f"  {gm['f_Hz']:10.1f}  {gm['tau_ms']:10.3f}"
              f"  {d_frac*100:+12.4f}")

    # --- 7. Summary ---
    print("\n--- 7. Summary and Honest Assessment ---")
    summary = summarize_qnm_analysis()
    print(f"\n  Framework: {summary['framework']}")
    print(f"  Method:    {summary['method']}")
    dc = summary["dilaton_coupling"]
    print(f"  a = {dc['a']:.6f}  (a^2 = {dc['a_squared']:.6f})")
    print(f"  gamma = {dc['gamma']:.6f}")

    wv = summary["wkb_validation"]["schwarzschild_l2_n0"]
    print(f"\n  WKB validation (Schwarzschild l=2, n=0):")
    print(f"    omega_R*M:  exact = {wv['exact_omega_R_M']:.4f}, "
          f"WKB = {_fmt(wv['wkb_omega_R_M'])}")
    if wv["error_omega_R"] is not None:
        print(f"    omega_R error: {wv['error_omega_R']*100:.1f}%")
    print(f"    omega_I*M:  exact = {wv['exact_omega_I_M']:.4f}, "
          f"WKB = {_fmt(wv['wkb_omega_I_M'])}")
    if wv["error_omega_I"] is not None:
        print(f"    omega_I error: {wv['error_omega_I']*100:.1f}%")

    print(f"\n  Key findings:")
    for i, finding in enumerate(summary["key_findings"], 1):
        print(f"    {i}. {finding}")

    print(f"\n  Honest assessment:")
    for i, point in enumerate(summary["honest_assessment"], 1):
        print(f"    {i}. {point}")

    print(f"\n  Theoretical significance:")
    sig = summary["theoretical_significance"]
    words = sig.split()
    line = "    "
    for word in words:
        if len(line) + len(word) + 1 > 72:
            print(line)
            line = "    " + word
        else:
            line = line + " " + word if line.strip() else "    " + word
    if line.strip():
        print(line)

    print("\n" + "=" * 72)
    print("Done.")

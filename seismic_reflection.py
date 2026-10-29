"""
Applied Geophysics (PVU, Sep-Dec 2026) - Assignment 06
Solutions to Chapter 4 problems (Seismic Exploration: The Reflection Method)
Burger, Sheehan & Jones, Introduction to Applied Geophysics.

Run:  python seismic_reflection.py
Output: results.json and figures/*.png

Units inside the code: distance in m, time in s (converted from ms on input),
velocity in m/s.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})
RESULTS = {}
ms = 1e-3


# ---------------------------------------------------------------------------
# Core reflection tools
# ---------------------------------------------------------------------------
def x2t2_fit(x, t):
    """Least-squares fit of t^2 = t0^2 + x^2 / V^2 (Burger eq. 4.6 / 4.10).

    x in m, t in s. Returns V (m/s), t0 (s), h = V*t0/2 (m), R^2.
    """
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    X, Y = x ** 2, t ** 2
    b, a = np.polyfit(X, Y, 1)
    pred = a + b * X
    r2 = 1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)
    V = 1 / np.sqrt(b)
    t0 = np.sqrt(a)
    return {"V": float(V), "t0": float(t0), "h": float(V * t0 / 2),
            "slope": float(b), "intercept": float(a), "R2": float(r2)}


def dix(vrms, t0):
    """Interval velocities and thicknesses from rms velocities (eq. 4.21, 4.22)."""
    vrms = np.asarray(vrms, float)
    t0 = np.asarray(t0, float)
    vint, h = [], []
    for i in range(len(vrms)):
        if i == 0:
            v2 = vrms[0] ** 2
            dt = t0[0]
        else:
            v2 = (vrms[i] ** 2 * t0[i] - vrms[i - 1] ** 2 * t0[i - 1]) / (t0[i] - t0[i - 1])
            dt = t0[i] - t0[i - 1]
        v = float(np.sqrt(v2)) if v2 > 0 else float("nan")
        vint.append(v)
        h.append(v * dt / 2)
    return vint, h


def plot_x2t2(ax, x, t, fit, label, color):
    X = np.asarray(x) ** 2
    ax.plot(X, (np.asarray(t) / ms) ** 2, "o", ms=4, color=color, label=label)
    xx = np.linspace(0, X.max() * 1.05, 50)
    ax.plot(xx, (fit["intercept"] + fit["slope"] * xx) / ms ** 2, "-", color=color, lw=1,
            label=f"V = {fit['V']:.0f} m/s, t0 = {fit['t0'] / ms:.1f} ms")
    ax.set_xlabel("x² (m²)")
    ax.set_ylabel("t² (ms²)")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


RNG = np.random.default_rng(2026)


def monte_carlo(x, t, sigma, n=4000):
    """Spread of V, t0, h when every pick has a random error N(0, sigma) (s)."""
    V, t0, h = [], [], []
    for _ in range(n):
        f = x2t2_fit(x, t + RNG.normal(0, sigma, len(t)))
        if np.isfinite(f["V"]) and f["slope"] > 0 and f["intercept"] > 0:
            V.append(f["V"]); t0.append(f["t0"]); h.append(f["h"])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in
            {"V": V, "t0": t0, "h": h}.items()}


def single_reflection(problem, x, t_ms, title, source, sigma_ms=1.0):
    t = np.asarray(t_ms) * ms
    fit = x2t2_fit(x, t)
    fit["mc_sigma_ms"] = sigma_ms
    fit["mc"] = monte_carlo(np.asarray(x, float), t, sigma_ms * ms)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(x, t_ms, "o", color="C0")
    xx = np.linspace(0, max(x) * 1.05, 100)
    axs[0].plot(xx, np.sqrt(fit["t0"] ** 2 + xx ** 2 / fit["V"] ** 2) / ms, "-", color="C0", lw=1)
    axs[0].set_xlabel("Source–geophone distance x (m)")
    axs[0].set_ylabel("Reflection time (ms)")
    axs[0].set_title("Time–distance curve")
    plot_x2t2(axs[1], x, t, fit, "picks", "C0")
    axs[1].set_title("x²–t² graph")
    axs[1].legend()
    fig.suptitle(title)
    save(fig, f"p{problem}.png")
    RESULTS[problem] = {"x": list(map(float, x)), "t_ms": list(map(float, t_ms)),
                        "source": source, **fit}
    return fit


def multi_reflection(problem, x, picks_ms, title, source):
    fits = []
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for k, tk in enumerate(picks_ms):
        t = np.asarray(tk) * ms
        f = x2t2_fit(x, t)
        fits.append(f)
        axs[0].plot(x, tk, "o", ms=4, color=f"C{k}", label=f"Reflection #{k + 1}")
        plot_x2t2(axs[1], x, t, f, f"#{k + 1}", f"C{k}")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("t (ms)")
    axs[0].legend()
    axs[1].legend(fontsize=7)
    fig.suptitle(title)
    save(fig, f"p{problem}.png")
    vrms = [f["V"] for f in fits]
    t0 = [f["t0"] for f in fits]
    vint, h = dix(vrms, t0)
    depth = list(np.nancumsum(h))
    RESULTS[problem] = {"source": source, "fits": fits, "Vrms": vrms, "t0": t0,
                        "Vint": vint, "h": h, "depth": depth}
    return RESULTS[problem]


# ---------------------------------------------------------------------------
# Problem 4.1 and 4.2 - field seismograms (Fig. 4.51), times picked on the
# first peak of the arrowed reflection after calibrating the time axis
# (0, 100, 200 ms grid lines) on a 2400-px render of the page.
# ---------------------------------------------------------------------------
x41 = np.arange(40, 74, 3)
t41 = [79.9, 80.8, 81.7, 82.7, 83.8, 84.8, 86.4, 87.5, 88.7, 90.5, 92.1, 93.1]
single_reflection("4.1", x41, t41, "Problem 4.1 – Fig. 4.51(a)", "picked from Fig. 4.51(a)")

x42 = np.arange(50, 84, 3)
t42 = [99.6, 101.2, 101.8, 102.5, 103.4, 104.2, 104.9, 105.4, 106.6, 107.6, 108.2, 109.3]
single_reflection("4.2", x42, t42, "Problem 4.2 – Fig. 4.51(b)", "picked from Fig. 4.51(b)")

# ---------------------------------------------------------------------------
# Problem 4.3 - center-shot record (Fig. 4.52). The three arrowed reflections
# were read at the nearest trace (34 m, enlarged panel b) and the farthest trace
# (418 m, panel a). Two offsets give an exact x^2-t^2 line for each event.
# ---------------------------------------------------------------------------
x43 = np.array([34.0, 418.0])
picks43 = [[453.0, 494.0], [507.0, 548.0], [651.0, 699.0]]
multi_reflection("4.3", x43, picks43, "Problem 4.3 – Fig. 4.52 (34 m and 418 m traces)",
                 "picked from Fig. 4.52(a,b)")
# Pick uncertainty on this poor-quality record: +/-3 ms on each pick
RESULTS["4.3"]["mc"] = [monte_carlo(x43, np.array(p) * ms, 3 * ms) for p in picks43]

# ---------------------------------------------------------------------------
# Problem 4.4 and 4.5 - tabulated data
# ---------------------------------------------------------------------------
x44 = np.arange(3, 37, 3)
p44 = [[21.4, 25, 30.1, 36.1, 42.5, 49.2, 56.2, 63.3, 70.4, 77.6, 84.9, 92.2],
       [62.3, 62.4, 62.6, 62.9, 63.2, 63.6, 64.1, 64.7, 65.4, 66.1, 66.9, 67.7],
       [79.4, 79.5, 79.6, 79.9, 80.1, 80.5, 80.9, 81.3, 81.8, 82.4, 83, 83.7]]
multi_reflection("4.4", x44, p44, "Problem 4.4", "table in problem statement")

x45 = np.arange(3, 114, 10)
p45 = [[40.5, 47.7, 61, 77.2, 94.9, 113.3, 132.2, 151.4, 170.8, 190.3, 209.9, 229.5],
       [102.5, 103.1, 104.4, 106.3, 108.9, 112.1, 115.8, 120, 124.6, 129.7, 135.1, 140.8],
       [116.8, 117.2, 118, 119.2, 120.9, 123, 125.5, 128.4, 131.6, 135.1, 138.9, 143]]
multi_reflection("4.5", x45, p45, "Problem 4.5", "table in problem statement")


# ---------------------------------------------------------------------------
# Dipping reflector tools (Burger section 4.3)
# t^2 = (4j^2 + x^2 - 4 j x sin(beta)) / V^2   (eq. 4.26), x positive down-dip
# here we keep the book's sign: x_min = 2 j sin(beta) lies UP-dip.
# ---------------------------------------------------------------------------
def dip_tmin_t0(x, t):
    """t_min / t0 method (eqs. 4.29-4.37) using a hyperbola fitted to all picks."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    # Fit t^2 = A + B x + C x^2 (exact form of eq. 4.26)
    C, B, A = np.polyfit(x, t ** 2, 2)
    x_min = -B / (2 * C)
    t_min = np.sqrt(A - B ** 2 / (4 * C))
    t0 = np.sqrt(A)
    beta = np.degrees(np.arccos(t_min / t0))
    V = 1 / np.sqrt(C)
    j = abs(x_min) / (2 * np.sin(np.radians(beta)))
    h = j / np.cos(np.radians(beta))
    return {"t0": float(t0), "t_min": float(t_min), "x_min": float(x_min),
            "beta": float(beta), "V": float(V), "j": float(j), "h": float(h),
            "fit": [float(A), float(B), float(C)]}


def branches(x, t):
    """Split a split-spread record into +x and -x branches, both sorted by |x|."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    p, n = x > 0, x < 0
    ip, iN = np.argsort(x[p]), np.argsort(-x[n])
    return x[p][ip], t[p][ip], -x[n][iN], t[n][iN]


def paired_times(x, t, offsets=None):
    """t(+d) and t(-d) at common offsets d. Exact pairs when the stations are
    symmetric; linear interpolation along each branch when they are not
    (digitized figures)."""
    xp, tp, xn, tn = branches(x, t)
    if offsets is None:
        offsets = [d for d in xp if np.any(np.isclose(xn, d, atol=0.05))]
        if len(offsets) < 4:   # irregular stations: use the +x stations inside both ranges
            lo, hi = max(xp.min(), xn.min()), min(xp.max(), xn.max())
            offsets = [d for d in xp if lo <= d <= hi]
    offsets = np.asarray(offsets, float)
    return offsets, np.interp(offsets, xp, tp), np.interp(offsets, xn, tn)


def dip_x2t2_average(x, t):
    """x^2-t^2 'average data' method (eq. 4.39): average t^2 at +x and -x."""
    d, tp, tn = paired_times(x, t)
    ta = np.sqrt((tp ** 2 + tn ** 2) / 2)
    xa = list(map(float, d))
    fit = x2t2_fit(d, ta)
    V, j = fit["V"], fit["h"]           # j = perpendicular distance shot -> reflector
    # dip from the farthest +x/-x pair: subtracting eq. 4.26 written for -x and +x
    # gives t(-x)^2 - t(+x)^2 = 8 j x sin(beta) / V^2
    sinb = (tn[-1] ** 2 - tp[-1] ** 2) * V ** 2 / (8 * j * d[-1])
    beta = np.degrees(np.arcsin(sinb))
    return {"V": V, "j": j, "beta": float(beta), "h": float(j / np.cos(np.radians(beta))),
            "t0": fit["t0"], "R2": fit["R2"], "x_avg": xa, "t_avg": [float(v) for v in ta]}


def dip_dmo(x, t, V, offsets=None):
    """Dip move-out (eqs. 4.48, 4.57): T_DMO = t(+x) - t(-x), beta = asin(-V T_DMO / 2x).
    Returns rows (x, T_DMO in ms, beta in degrees)."""
    d, tp, tn = paired_times(x, t, offsets)
    tdmo = tp - tn
    beta = np.degrees(np.arcsin(np.clip(-V * tdmo / (2 * d), -1, 1)))
    return [(float(a), float(b / ms), float(c)) for a, b, c in zip(d, tdmo, beta)]


def dipping_problem(problem, x, t_ms, title, source, pair_tol=0.6):
    t = np.asarray(t_ms) * ms
    r1 = dip_tmin_t0(x, t)
    r2 = dip_x2t2_average(x, t)
    dmo = dip_dmo(x, t, r2["V"])
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
    axs[0].plot(x, t_ms, "s", ms=4, color="C0", label="data")
    xx = np.linspace(min(x), max(x), 200)
    A, B, C = r1["fit"]
    axs[0].plot(xx, np.sqrt(A + B * xx + C * xx ** 2) / ms, "-", lw=1, color="C0", label="fitted hyperbola (eq. 4.26)")
    axs[0].axvline(r1["x_min"], ls="--", color="C3", lw=0.8)
    axs[0].annotate(f"x_min = {r1['x_min']:.1f} m\nt_min = {r1['t_min'] / ms:.2f} ms",
                    (r1["x_min"], r1["t_min"] / ms), textcoords="offset points", xytext=(8, 20), fontsize=8)
    axs[0].set_xlabel("Distance from shot (m)")
    axs[0].set_ylabel("Time (ms)")
    axs[0].legend(fontsize=7)
    axs[0].set_title("t_min / t0 method")
    X = np.asarray(x) ** 2
    axs[1].plot(X[np.asarray(x) > 0], (t[np.asarray(x) > 0] / ms) ** 2, "^", ms=4, color="C1", label="+x side")
    axs[1].plot(X[np.asarray(x) < 0], (t[np.asarray(x) < 0] / ms) ** 2, "v", ms=4, color="C2", label="−x side")
    xa = np.array(r2["x_avg"]) ** 2
    axs[1].plot(xa, (np.array(r2["t_avg"]) / ms) ** 2, "o", ms=4, color="k", label="average")
    xx2 = np.linspace(0, X.max(), 50)
    axs[1].plot(xx2, (r2["t0"] ** 2 + xx2 / r2["V"] ** 2) / ms ** 2, "k-", lw=1,
                label=f"V = {r2['V']:.0f} m/s")
    axs[1].set_xlabel("x² (m²)")
    axs[1].set_ylabel("t² (ms²)")
    axs[1].legend(fontsize=7)
    axs[1].set_title("x²–t² average-data method")
    fig.suptitle(title)
    save(fig, f"p{problem}.png")
    RESULTS[problem] = {"source": source, "tmin_t0": r1, "x2t2_avg": r2,
                        "dmo": dmo, "x": list(map(float, x)), "t_ms": list(map(float, t_ms))}
    return RESULTS[problem]


# Problem 4.6 - Fig. 4.53(a), digitized automatically (digitize_fig453.py)
dig = json.load(open("fig453_digitized.json"))
xa_, ta_ = np.array(dig["a"]).T
dipping_problem("4.6", xa_, ta_, "Problem 4.6 – Fig. 4.53(a)", "digitized from Fig. 4.53(a)")

# Problem 4.8 - tabulated split-spread data. The upper half of the table is the
# side with the larger times (written here as negative x).
x48 = np.array([-36, -33, -30, -27, -24, -21, -18, -15, -12, -9, -6, -3,
                3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36], float)
t48 = [22.1, 21.44, 20.8, 20.21, 19.66, 19.15, 18.68, 18.27, 17.92, 17.62, 17.38, 17.21,
       17.06, 17.09, 17.18, 17.34, 17.56, 17.84, 18.19, 18.59, 19.04, 19.54, 20.08, 20.67]
dipping_problem("4.8", x48, t48, "Problem 4.8", "table in problem statement")

# Problem 4.10 - Fig. 4.53(b): dip move-out. Digitized x values are snapped to
# the nominal geophone stations before pairing +x with -x.
xb_, tb_ = np.array(dig["b"]).T
r410 = dipping_problem("4.10", xb_, tb_, "Problem 4.10 – Fig. 4.53(b)", "digitized from Fig. 4.53(b)")


# DMO at round offsets for the report tables (V from the x^2-t^2 average-data fit)
for key, (xx_, tt_), offs in [("4.6", (xa_, ta_), [6, 12, 18, 24, 30, 36]),
                              ("4.10", (xb_, tb_), [16, 24, 32, 40, 48, 56, 64, 72, 80, 88])]:
    Vk = RESULTS[key]["x2t2_avg"]["V"]
    rows = dip_dmo(xx_, np.asarray(tt_) * ms, Vk, offs)
    bet = np.array([r[2] for r in rows])
    RESULTS[key]["dmo_table"] = rows
    RESULTS[key]["dmo_beta_mean"] = float(bet.mean())
    RESULTS[key]["dmo_beta_std"] = float(bet.std(ddof=1))
    jj = RESULTS[key]["x2t2_avg"]["j"]
    RESULTS[key]["dmo_h"] = float(jj / np.cos(np.radians(bet.mean())))
RESULTS["4.8"]["dmo_table"] = RESULTS["4.8"]["dmo"]

# ---------------------------------------------------------------------------
# Problem 4.9 - Fig. 4.54 (picks on the arrowed peak; traces 26 and 28 m are
# masked by ground roll and are not used)
# ---------------------------------------------------------------------------
x49 = np.arange(30, 49, 2)
t49 = [84.0, 84.3, 84.9, 85.2, 85.6, 85.7, 86.2, 86.6, 87.3, 87.5]
single_reflection("4.9", x49, t49, "Problem 4.9 – Fig. 4.54", "picked from Fig. 4.54")

# Problem 4.12 - Fig. 4.26(b): offset 60 m, interval 3 m, timing lines 10 ms
x412 = np.arange(60, 97, 3)
t412 = [174.7, 175.2, 176.6, 178.2, 178.8, 179.7, 180.7, 182.2, 183.0, 184.2, 185.7, 187.5, 189.0]
single_reflection("4.12", x412, t412, "Problem 4.12 – Fig. 4.26(b)", "picked from Fig. 4.26(b)")


# ---------------------------------------------------------------------------
# Problem 4.7 and 4.11 - optimum window, wide-angle effects
# ---------------------------------------------------------------------------
def ray_reflection_time(x, h, v):
    """Exact two-way reflection time from the base of a stack of flat layers
    (thicknesses h, velocities v) at offset x, by shooting on ray parameter p."""
    h = np.asarray(h, float)
    v = np.asarray(v, float)
    pmax = 1 / v.max() * (1 - 1e-12)

    def offset(p):
        s = p * v
        return 2 * np.sum(h * s / np.sqrt(1 - s ** 2))

    lo, hi = 0.0, pmax
    for _ in range(200):
        mid = (lo + hi) / 2
        if offset(mid) < x:
            lo = mid
        else:
            hi = mid
    p = (lo + hi) / 2
    s = p * v
    return 2 * np.sum(h / (v * np.sqrt(1 - s ** 2)))


def head_wave_time(x, h, v, k):
    """Refraction (head wave) along the top of layer k (0-based, k>=1)."""
    vk = v[k]
    t = x / vk
    for i in range(k):
        t += 2 * h[i] * np.sqrt(1 / v[i] ** 2 - 1 / vk ** 2)
    xc = sum(2 * h[i] * np.tan(np.arcsin(v[i] / vk)) for i in range(k))
    return t, xc


def optimum_window(problem, h, v, names, v_low=(330.0, 250.0), pulse=0.020):
    """Near side: bedrock reflection clears the low-velocity waves (air wave /
    ground roll) by one pulse length. Far side: the earliest of
    (a) the shallower reflection merges with the bedrock reflection (< pulse apart),
    (b) the head wave from the bedrock reaches the critical distance
        (beyond it the reflection is wide-angle and interferes with refractions),
    (c) offset equal to the bedrock depth (Dix/NMO and phase-change limit)."""
    h = np.asarray(h, float)
    v = np.asarray(v, float)
    xs = np.arange(1.0, 400.0, 1.0)
    t_bed = np.array([ray_reflection_time(x, h, v[:len(h)]) for x in xs])
    t_top = np.array([ray_reflection_time(x, h[:-1], v[:len(h) - 1]) for x in xs])
    out = {}
    v_slowest = min(v_low)
    near = xs[np.argmax(xs / v_slowest > t_bed + pulse)]
    out["near_side_m"] = float(near)
    sep = t_bed - t_top
    merge = xs[np.argmax(sep < pulse)] if np.any(sep < pulse) else None
    _, xc_bed = head_wave_time(1.0, h, v, len(h))
    depth = float(h.sum())
    cands = {"reflections merge": merge, "critical distance (bedrock)": xc_bed, "offset = depth": depth}
    out["far_candidates_m"] = {k: (None if c is None else float(c)) for k, c in cands.items()}
    out["far_side_m"] = float(min(c for c in cands.values() if c is not None))
    out["t0_bed_ms"] = float(t_bed[0] / ms)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, t_bed / ms, lw=2, color="C0", label=f"Bedrock reflection ({names[-1]} top)")
    ax.plot(xs, t_top / ms, lw=1.5, color="C1", label=f"Reflection from top of {names[-2]}")
    for vl, ls in zip(v_low, ["--", ":"]):
        ax.plot(xs, xs / vl / ms, "k" + ls, lw=1, label=f"Low-velocity wave {vl:.0f} m/s")
    for k in range(1, len(v)):
        th, xc = head_wave_time(xs, h, v, k)
        ax.plot(xs[xs >= xc], th[xs >= xc] / ms, lw=1, color=f"C{k + 2}",
                label=f"Head wave, top of layer {k + 1} ({v[k]:.0f} m/s)")
    ax.axvspan(out["near_side_m"], out["far_side_m"], color="C2", alpha=0.15, label="Optimum window")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, max(300, t_bed[0] / ms * 1.8))
    ax.set_xlabel("Source–geophone distance (m)")
    ax.set_ylabel("Time (ms)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"Problem {problem} – optimum window")
    save(fig, f"p{problem}.png")
    RESULTS[problem] = out
    return out


optimum_window("4.11", [80, 40], [1350, 2800, 5500], ["sands", "till", "bedrock"])


def problem_47():
    """Illustrate the two reasons for x_max <= depth for a 35 m deep target."""
    h, V = 35.0, 1500.0
    x = np.linspace(1, 120, 240)
    t0 = 2 * h / V
    exact = np.sqrt(t0 ** 2 + x ** 2 / V ** 2)
    nmo_exact = exact - t0
    nmo_approx = x ** 2 / (2 * t0 * V ** 2)          # first term of eq. 4.45
    err = (nmo_approx - nmo_exact) / nmo_exact * 100
    angle = np.degrees(np.arctan(x / (2 * h)))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    axs[0].plot(x, err, color="C3")
    axs[0].axvline(35, ls="--", color="k", lw=0.8)
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("Error of 1-term NMO / Dix approx. (%)")
    axs[0].set_title("(1) NMO/Dix approximation error")
    axs[1].plot(x, angle, color="C0")
    axs[1].axvline(35, ls="--", color="k", lw=0.8)
    axs[1].axhline(26.6, ls=":", color="C0", lw=0.8)
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("Angle of incidence at reflector (°)")
    axs[1].set_title("(2) Wide-angle incidence (x = h → 26.6°)")
    fig.suptitle("Problem 4.7 – target at 35 m, V = 1500 m/s (illustration)")
    save(fig, "p4.7.png")
    i35 = np.argmin(abs(x - 35))
    i70 = np.argmin(abs(x - 70))
    RESULTS["4.7"] = {"err_at_35_pct": float(err[i35]), "err_at_70_pct": float(err[i70]),
                      "angle_at_35_deg": float(angle[i35]), "angle_at_70_deg": float(angle[i70])}


problem_47()


# ---------------------------------------------------------------------------
# Problem 4.13 - step model: reflections and diffractions
# Coordinates: x along the line with the shot at x = 0, geophones every 10 m.
# Step edge at x = -20 m; reflector depth 25 m for x < -20, 20 m for x > -20.
# ---------------------------------------------------------------------------
def problem_413():
    V = 2200.0
    xs, xe = 0.0, -20.0
    geos = np.array([-80, -70, -60, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60], float)
    rows = []
    for g in geos:
        # image-point reflection from each flat segment; valid if the reflection
        # point (midpoint for flat beds) lies on that segment
        mid = (xs + g) / 2
        t_hi = np.hypot(g - xs, 2 * 20) / V if mid >= xe else None
        t_lo = np.hypot(g - xs, 2 * 25) / V if mid <= xe else None
        # diffraction from the upper corner of the step (xe, 20 m)
        t_dif = (np.hypot(xe - xs, 20) + np.hypot(g - xe, 20)) / V
        rows.append({"x": float(g),
                     "refl_20m_ms": None if t_hi is None else float(t_hi / ms),
                     "refl_25m_ms": None if t_lo is None else float(t_lo / ms),
                     "diffraction_ms": float(t_dif / ms)})
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 6.2), gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axs[0]
    xr = np.array([r["x"] for r in rows])
    hi = np.array([np.nan if r["refl_20m_ms"] is None else r["refl_20m_ms"] for r in rows])
    lo = np.array([np.nan if r["refl_25m_ms"] is None else r["refl_25m_ms"] for r in rows])
    df = np.array([r["diffraction_ms"] for r in rows])
    xx = np.linspace(-80, 60, 400)
    ax.plot(xx[xx >= 2 * xe], np.hypot(xx[xx >= 2 * xe], 40) / V / ms, "C0-", lw=1)
    ax.plot(xx[xx <= 2 * xe], np.hypot(xx[xx <= 2 * xe], 50) / V / ms, "C1-", lw=1)
    ax.plot(xx, (np.hypot(20, 20) + np.hypot(xx - xe, 20)) / V / ms, "C3--", lw=1)
    ax.plot(xr, hi, "o", color="C0", label="Reflection, 20 m bed")
    ax.plot(xr, lo, "s", color="C1", label="Reflection, 25 m bed")
    ax.plot(xr, df, "x", color="C3", label="Diffraction from step edge")
    ax.invert_yaxis()
    ax.set_ylabel("Time (ms)")
    ax.legend(fontsize=7)
    ax.set_title("Problem 4.13 – travel-time curves (V = 2200 m/s)")
    ax2 = axs[1]
    ax2.plot([-90, xe, xe, 70], [-25, -25, -20, -20], "k-", lw=2)
    ax2.plot(geos, np.zeros_like(geos), "kv")
    ax2.plot([0], [0], "r*", ms=12)
    ax2.set_ylim(-30, 3)
    ax2.set_xlabel("Distance from shot (m)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Model")
    for ax_ in axs:
        ax_.set_xlim(-90, 70)
    save(fig, "p4.13.png")
    RESULTS["4.13"] = rows


problem_413()


# ---------------------------------------------------------------------------
# Problem 4.14 - static corrections (eq. 4.62, 4.63)
# ---------------------------------------------------------------------------
def problem_414():
    Vw, V1, ed = 440.0, 1400.0, 100.0
    geos = [("Geophone 1", 102.0, 4.0), ("Geophone 2", 105.0, 6.0), ("Geophone 3", 103.0, 5.0)]
    rows = []
    for name, eg, hw in geos:
        tw = hw / Vw - hw / V1
        te = (eg - ed) / V1
        rows.append({"geophone": name, "elevation_m": eg, "hw_m": hw,
                     "tw_ms": tw / ms, "te_ms": te / ms, "total_ms": (tw + te) / ms})
    RESULTS["4.14"] = rows


problem_414()


# ---------------------------------------------------------------------------
# Problem 4.15 - automated velocity analysis (Velocity Analysis A, Table 4.11)
# Trial (t0, Vst) pairs; NMO from eq. 4.65 (two-term) and eq. 4.66 (exact);
# the best pair minimises the scatter of the NMO-corrected times.
# ---------------------------------------------------------------------------
def problem_415():
    x = np.arange(3, 37, 3, dtype=float)
    data = {"Reflection 1": np.array([33.71, 34.8, 36.55, 38.87, 41.67, 44.85, 48.33, 52.07, 56, 60.09, 64.31, 68.64]),
            "Reflection 2": np.array([93.35, 93.42, 93.53, 93.68, 93.87, 94.1, 94.38, 94.69, 95.05, 95.45, 95.89, 96.37])}
    out = {}
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7))
    for k, (name, t_ms) in enumerate(data.items()):
        t = t_ms * ms
        t0s = np.arange(25.0, 100.0, 0.01) * ms
        t0s = t0s[(t0s > t.min() - 15 * ms) & (t0s <= t.min())]
        vs = np.arange(300.0, 3000.0, 1.0)
        misfit = np.full((len(t0s), len(vs)), np.nan)
        for i, t0 in enumerate(t0s):
            # exact hyperbola (eq. 4.66): tx = t0 * sqrt(1 + x^2 / (V^2 t0^2))
            pred = t0 * np.sqrt(1 + x[None, :] ** 2 / (vs[:, None] ** 2 * t0 ** 2))
            misfit[i] = np.sqrt(np.mean((pred - t[None, :]) ** 2, axis=1))
        i, j = np.unravel_index(np.nanargmin(misfit), misfit.shape)
        t0b, vb = t0s[i], vs[j]
        nmo_exact = t - t0b
        nmo_2term = x ** 2 / (2 * t0b * vb ** 2) - x ** 4 / (8 * t0b ** 3 * vb ** 4)
        corrected = t - nmo_2term
        fit = x2t2_fit(x, t)
        out[name] = {"t0_ms": float(t0b / ms), "Vst": float(vb), "rms_misfit_ms": float(misfit[i, j] / ms),
                     "x2t2_check": {"V": fit["V"], "t0_ms": fit["t0"] / ms},
                     "corrected_eq465_ms": [float(c / ms) for c in corrected],
                     "corrected_exact_ms": [float(c / ms) for c in (t - nmo_exact)]}
        ax = axs[0, k]
        im = ax.pcolormesh(vs, t0s / ms, np.log10(misfit / ms), shading="auto", cmap="viridis_r")
        ax.plot(vb, t0b / ms, "r*", ms=12)
        ax.set_xlabel("Trial stacking velocity Vst (m/s)")
        ax.set_ylabel("Trial t0 (ms)")
        ax.set_title(f"{name}: misfit map (log10 ms)")
        fig.colorbar(im, ax=ax)
        ax = axs[1, k]
        ax.plot(x, t_ms, "o", label="recorded")
        ax.plot(x, corrected / ms, "s", label=f"NMO-corrected (eq. 4.65), Vst={vb:.0f}, t0={t0b / ms:.2f}")
        ax.invert_yaxis()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Time (ms)")
        ax.legend(fontsize=7)
    fig.suptitle("Problem 4.15 – velocity analysis (grid search on t0 and Vst)")
    save(fig, "p4.15.png")
    vint, h = dix([out["Reflection 1"]["Vst"], out["Reflection 2"]["Vst"]],
                  [out["Reflection 1"]["t0_ms"] * ms, out["Reflection 2"]["t0_ms"] * ms])
    out["Dix"] = {"Vint": vint, "h": h}
    RESULTS["4.15"] = out


problem_415()


# ---------------------------------------------------------------------------
# Problem 4.16 - migration of a zero-offset depth section (circle/wavefront method)
# Event depths measured on the figure (traces every 5 m, depth scale 10 m).
# ---------------------------------------------------------------------------
def problem_416():
    xt = np.array([0, 5, 10, 15, 20, 25], float)
    upper = np.array([15.63, 18.14, 20.79, 22.11, 19.75, 16.99])
    lower = np.array([30.25, 27.92, 25.13, 23.73, 25.77, 28.53])
    # Two straight events cross near trace 4 (a "bow-tie"); split them by
    # choosing, at every trace, the pick that continues each straight line.
    A = np.array([upper[0], upper[1], upper[2], lower[3], lower[4], lower[5]])  # deepens to the right
    B = np.array([lower[0], lower[1], lower[2], upper[3], upper[4], upper[5]])  # shallows to the right
    out = {}
    fig, ax = plt.subplots(figsize=(7, 5.4))
    th = np.linspace(0, np.pi, 200)
    for xi, d in zip(np.r_[xt, xt], np.r_[upper, lower]):
        ax.plot(xi + d * np.cos(th), d * np.sin(th), color="0.8", lw=0.6)
    for name, ev, c in [("Event A (dips right on section)", A, "C0"), ("Event B (dips left on section)", B, "C1")]:
        m, b = np.polyfit(xt, ev, 1)          # apparent slope dd/dx on the section
        sinb = m                             # zero-offset depth section: sin(true dip) = dd/dx
        xm = xt - ev * sinb
        zm = ev * np.sqrt(1 - sinb ** 2)
        out[name] = {"section_slope": float(m), "apparent_dip_deg": float(np.degrees(np.arctan(m))),
                     "true_dip_deg": float(np.degrees(np.arcsin(sinb))),
                     "unmigrated": [(float(a), float(b_)) for a, b_ in zip(xt, ev)],
                     "migrated": [(float(a), float(b_)) for a, b_ in zip(xm, zm)]}
        ax.plot(xt, ev, "o--", color=c, ms=4, lw=0.8, label=f"{name}: unmigrated")
        ax.plot(xm, zm, "s-", color=c, ms=4, lw=2, label=f"migrated (true dip {np.degrees(np.arcsin(sinb)):.1f}°)")
    ax.plot(xt, np.zeros_like(xt), "kv", ms=8)
    ax.set_xlim(-25, 50)
    ax.set_ylim(35, -2)
    ax.set_aspect("equal")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Problem 4.16 – migration by wavefront (circle) construction")
    save(fig, "p4.16.png")
    RESULTS["4.16"] = out


problem_416()

json.dump(RESULTS, open("results.json", "w"), indent=1, default=float)
print("Done. Results in results.json, figures in", FIG)

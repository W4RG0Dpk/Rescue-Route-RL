"""
Candidate stochastic weather models: fitting and simulation.

Rainfall (hourly, mm/h)
-----------------------
R-V1   The CURRENT simulator process, run through the actual
       ``RainModelV1`` code: clipped Gaussian AR(1) on the normalised
       intensity R = min(P / P_ref, 1). Fitted by moments (mean, std,
       lag-1 autocorrelation) on the normalised training series.
R-MG   Two-state wet/dry Markov chain + Gamma wet-hour amounts
       (the alternative proposed in the report, section 10.2):
           P(wet_t | dry_{t-1}) = p01,  P(wet_t | wet_{t-1}) = p11
           P_t - P_wet | wet ~ Gamma(k, theta)
R-LG   Latent-Gaussian censored AR(1) ("meta-Gaussian" rainfall model,
       cf. Bardossy & Plate 1992; Allcroft & Glasbey 2003):
           Z_t = rho Z_{t-1} + sqrt(1 - rho^2) eps_t,     eps ~ N(0, 1)
           dry          if Z_t <= c,     c = Phi^{-1}(1 - p_wet)
           P_t = P_wet + F_Gamma^{-1}( (Phi(Z_t) - Phi(c)) / (1 - Phi(c)) )   otherwise
       One latent process carries occurrence AND intensity persistence;
       rho is fitted so the model reproduces P(wet_t, wet_{t+1}) exactly.

Wind (hourly, 10 m, m/s)
------------------------
W-V1   The CURRENT ``WindModel`` code: AR(1) speed (floor 0.5 m/s),
       uniform random-walk direction, Bernoulli gusts with uniform
       magnitude and x0.5 decay. Parameters fitted by moments.
W-VAR  Bivariate VAR(1) on the horizontal components w = (u, v)
       (report section 10.3):
           w_{t+1} = mu + A (w_t - mu) + e_t,   e_t ~ N(0, Sigma)
       speed = |w|, direction = atan2(u, v). Gust excess G = gust - speed:
           G_t = max(0, a + b S_t + eta_t),   eta_t = phi eta_{t-1} + s xi_t
       Fitted by ordinary least squares on within-segment pairs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import linalg, optimize, stats

from ..weather.rain import RainModelV1
from ..weather.wind import WindModel
from .stats import WET_THRESHOLD_MM, acf

# ---------------------------------------------------------------------------
# Rainfall
# ---------------------------------------------------------------------------


@dataclass
class RainV1Params:
    mean: float
    variability: float
    correlation: float
    p_ref_mm_h: float


@dataclass
class MarkovGammaParams:
    p01: float
    p11: float
    gamma_shape: float
    gamma_scale: float
    wet_threshold: float = WET_THRESHOLD_MM


@dataclass
class LatentGaussianParams:
    p_wet: float
    threshold_c: float
    rho: float
    gamma_shape: float
    gamma_scale: float
    wet_threshold: float = WET_THRESHOLD_MM


def _pairs(segments, fn):
    a, b = [], []
    for s in segments:
        s = fn(np.asarray(s, dtype=np.float64))
        a.append(s[:-1])
        b.append(s[1:])
    return np.concatenate(a), np.concatenate(b)


def fit_rain_v1(segments: Sequence[np.ndarray], p_ref: float) -> RainV1Params:
    norm = [np.minimum(np.asarray(s) / p_ref, 1.0) for s in segments]
    x = np.concatenate(norm)
    rho = float(np.clip(acf(norm, 1)[1], 0.0, 0.999))
    return RainV1Params(float(x.mean()), float(x.std()), rho, float(p_ref))


def simulate_rain_v1(p: RainV1Params, lengths: Sequence[int], rng: np.random.Generator) -> List[np.ndarray]:
    out = []
    for n in lengths:
        model = RainModelV1(np.clip(p.mean, 0, 1), p.variability, p.correlation, rng=rng)
        x = np.empty(n)
        x[0] = model.reset()
        for t in range(1, n):
            x[t] = model.step()
        out.append(x * p.p_ref_mm_h)
    return out


def _fit_gamma_excess(wet_amounts: np.ndarray, thr: float):
    y = np.maximum(wet_amounts - thr, 0.0) + 1e-3
    shape, _, scale = stats.gamma.fit(y, floc=0.0)
    return float(shape), float(scale)


def fit_markov_gamma(segments, thr=WET_THRESHOLD_MM) -> MarkovGammaParams:
    a, b = _pairs(segments, lambda s: (s >= thr).astype(int))
    p01 = float(np.mean(b[a == 0])) if np.any(a == 0) else 0.0
    p11 = float(np.mean(b[a == 1])) if np.any(a == 1) else 0.0
    x = np.concatenate(segments)
    k, th = _fit_gamma_excess(x[x >= thr], thr)
    return MarkovGammaParams(p01, p11, k, th, thr)


def simulate_markov_gamma(p: MarkovGammaParams, lengths, rng) -> List[np.ndarray]:
    pi_wet = p.p01 / max(1e-12, (1.0 - p.p11 + p.p01))
    out = []
    for n in lengths:
        wet = np.empty(n, dtype=bool)
        wet[0] = rng.random() < pi_wet
        u = rng.random(n)
        for t in range(1, n):
            wet[t] = u[t] < (p.p11 if wet[t - 1] else p.p01)
        amt = p.wet_threshold + rng.gamma(p.gamma_shape, p.gamma_scale, n)
        out.append(np.where(wet, amt, 0.0))
    return out


def fit_latent_gaussian(segments, thr=WET_THRESHOLD_MM) -> LatentGaussianParams:
    x = np.concatenate(segments)
    p_wet = float(np.mean(x >= thr))
    c = float(stats.norm.ppf(1.0 - p_wet))
    a, b = _pairs(segments, lambda s: (s >= thr).astype(int))
    p_ww = float(np.mean((a == 1) & (b == 1)))

    def joint(rho):
        # P(Z1 > c, Z2 > c) = Phi2(-c, -c; rho)
        return float(stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([-c, -c])) - p_ww

    lo, hi = 0.0, 0.9999
    rho = float(optimize.brentq(joint, lo, hi)) if joint(lo) < 0 < joint(hi) else float(np.clip(acf(segments, 1)[1], 0, 0.999))
    k, th = _fit_gamma_excess(x[x >= thr], thr)
    return LatentGaussianParams(p_wet, c, rho, k, th, thr)


def latent_to_rain(z: np.ndarray, p: LatentGaussianParams) -> np.ndarray:
    phi_c = stats.norm.cdf(p.threshold_c)
    u = (stats.norm.cdf(z) - phi_c) / max(1e-12, 1.0 - phi_c)
    u = np.clip(u, 1e-9, 1 - 1e-9)
    amt = p.wet_threshold + stats.gamma.ppf(u, p.gamma_shape, scale=p.gamma_scale)
    return np.where(z > p.threshold_c, amt, 0.0)


def simulate_latent_gaussian(p: LatentGaussianParams, lengths, rng) -> List[np.ndarray]:
    out = []
    s = np.sqrt(1.0 - p.rho ** 2)
    for n in lengths:
        z = np.empty(n)
        z[0] = rng.standard_normal()
        e = rng.standard_normal(n)
        for t in range(1, n):
            z[t] = p.rho * z[t - 1] + s * e[t]
        out.append(latent_to_rain(z, p))
    return out


@dataclass
class LatentGaussian2Params:
    """Two-timescale latent process: Z = sqrt(w) Z_fast + sqrt(1-w) Z_slow."""

    p_wet: float
    threshold_c: float
    weight_fast: float
    rho_fast: float
    rho_slow: float
    gamma_shape: float
    gamma_scale: float
    wet_threshold: float = WET_THRESHOLD_MM

    def latent_acf(self, k):
        k = np.asarray(k, dtype=np.float64)
        return self.weight_fast * self.rho_fast ** k + (1 - self.weight_fast) * self.rho_slow ** k


def _latent_corr_from_joint(c: float, p_joint: float) -> float:
    """Invert P(Z1 > c, Z2 > c) = Phi2(-c, -c; r) for the latent correlation r."""

    def f(r):
        return float(stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-c, -c])) - p_joint

    lo, hi = -0.5, 0.9999
    if f(lo) >= 0:
        return lo
    if f(hi) <= 0:
        return hi
    return float(optimize.brentq(f, lo, hi, xtol=1e-5))


def fit_latent_gaussian_2scale(segments, thr=WET_THRESHOLD_MM, max_lag=48) -> LatentGaussian2Params:
    """
    Fit the latent correlation function r(k) = w rho_f^k + (1-w) rho_s^k to the
    lag-k wet/wet joint probabilities (k = 1..max_lag) of the training data.
    The slow component represents multi-day active/break monsoon regimes.
    """
    x = np.concatenate(segments)
    p_wet = float(np.mean(x >= thr))
    c = float(stats.norm.ppf(1.0 - p_wet))
    wet = [(np.asarray(s) >= thr).astype(int) for s in segments]
    lags = np.arange(1, max_lag + 1)
    r_hat = []
    for k in lags:
        both = sum(int(np.sum(w[:-k] & w[k:])) for w in wet if len(w) > k)
        total = sum(len(w) - k for w in wet if len(w) > k)
        r_hat.append(_latent_corr_from_joint(c, both / total))
    r_hat = np.array(r_hat)

    def resid(theta):
        w, rf, rs = theta
        return w * rf ** lags + (1 - w) * rs ** lags - r_hat

    sol = optimize.least_squares(resid, x0=[0.5, 0.8, 0.99], bounds=([0, 0, 0], [1, 0.9999, 0.9999]))
    w, rf, rs = sol.x
    if rf > rs:  # keep naming consistent: the "fast" component decays faster
        w, rf, rs = 1 - w, rs, rf
    k, th = _fit_gamma_excess(x[x >= thr], thr)
    return LatentGaussian2Params(p_wet, c, float(w), float(rf), float(rs), k, th, thr)


def simulate_latent_gaussian_2scale(p: LatentGaussian2Params, lengths, rng) -> List[np.ndarray]:
    out = []
    sf, ss = np.sqrt(1 - p.rho_fast ** 2), np.sqrt(1 - p.rho_slow ** 2)
    a, b = np.sqrt(p.weight_fast), np.sqrt(1 - p.weight_fast)
    for n in lengths:
        zf, zs = np.empty(n), np.empty(n)
        zf[0], zs[0] = rng.standard_normal(2)
        e = rng.standard_normal((n, 2))
        for t in range(1, n):
            zf[t] = p.rho_fast * zf[t - 1] + sf * e[t, 0]
            zs[t] = p.rho_slow * zs[t - 1] + ss * e[t, 1]
        out.append(latent_to_rain(a * zf + b * zs, p))
    return out


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------


@dataclass
class WindV1Params:
    base_speed: float
    speed_std: float
    correlation: float
    direction_drift_deg: float
    gust_probability: float
    min_gust: float
    max_gust: float
    gust_decay: float = 0.5
    min_speed: float = 0.5


@dataclass
class WindVARParams:
    mu: List[float]            # [u, v] mean, m/s (east, north)
    A: List[List[float]]
    Sigma: List[List[float]]
    gust_a: float
    gust_b: float
    gust_phi: float
    gust_sigma: float

    def stationary_cov(self) -> np.ndarray:
        A, S = np.asarray(self.A), np.asarray(self.Sigma)
        return linalg.solve_discrete_lyapunov(A, S)


def fit_wind_v1(speed_seg, dir_seg, gust_seg) -> WindV1Params:
    s = np.concatenate(speed_seg)
    rho = float(np.clip(acf(speed_seg, 1)[1], 0.0, 0.999))
    d = []
    for th in dir_seg:
        dd = np.diff(np.asarray(th, dtype=np.float64))
        d.append((dd + 180.0) % 360.0 - 180.0)
    drift = float(np.sqrt(3.0) * np.concatenate(d).std())  # Uniform(-d, d) has std d / sqrt(3)
    g = np.concatenate(gust_seg)
    gmin, gmax = float(np.percentile(g, 10)), float(np.percentile(g, 99))
    m = 0.5 * (gmin + gmax)
    target = float(g.mean())
    # Stationary mean of the V1 gust process: E[G] = 2 p m / (1 + p)  =>  p = E / (2m - E)
    p = float(np.clip(target / max(1e-9, 2 * m - target), 0.0, 1.0))
    return WindV1Params(float(s.mean()), float(s.std()), rho, drift, p, gmin, gmax)


def simulate_wind_v1(p: WindV1Params, lengths, rng) -> Dict[str, List[np.ndarray]]:
    out = {k: [] for k in ("speed", "direction", "u", "v", "gust")}
    for n in lengths:
        wm = WindModel(
            base_speed=p.base_speed, speed_std=p.speed_std, correlation_rho=p.correlation,
            gust_probability=p.gust_probability, max_gust=max(p.max_gust, p.min_gust), min_gust=p.min_gust,
            direction_drift_deg=p.direction_drift_deg, gust_decay=p.gust_decay, min_speed=p.min_speed,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        st = wm.reset()
        sp, th, gu = np.empty(n), np.empty(n), np.empty(n)
        for t in range(n):
            if t > 0:
                st = wm.step()
            sp[t], th[t], gu[t] = st.speed, st.direction_deg, st.gust_strength
        rad = np.radians(th)
        out["speed"].append(sp)
        out["direction"].append(th)
        out["u"].append(sp * np.sin(rad))
        out["v"].append(sp * np.cos(rad))
        out["gust"].append(gu)
    return out


def fit_wind_var(u_seg, v_seg, speed_seg, gust_seg) -> WindVARParams:
    X, Y = [], []
    for u, v in zip(u_seg, v_seg):
        w = np.column_stack([u, v])
        X.append(w[:-1])
        Y.append(w[1:])
    X, Y = np.concatenate(X), np.concatenate(Y)
    mu = np.concatenate([X, Y]).mean(axis=0)
    Xc, Yc = X - mu, Y - mu
    A = np.linalg.lstsq(Xc, Yc, rcond=None)[0].T
    resid = Yc - Xc @ A.T
    Sigma = np.cov(resid.T)

    # Gust excess: linear in speed, AR(1) residual.
    s = np.concatenate(speed_seg)
    g = np.concatenate(gust_seg)
    b, a = np.polyfit(s, g, 1)
    eta_seg = [np.asarray(gg) - (a + b * np.asarray(ss)) for gg, ss in zip(gust_seg, speed_seg)]
    phi = float(np.clip(acf(eta_seg, 1)[1], 0.0, 0.999))
    eta = np.concatenate(eta_seg)
    s_eta = float(eta.std() * np.sqrt(1 - phi ** 2))
    return WindVARParams(mu.tolist(), A.tolist(), Sigma.tolist(), float(a), float(b), phi, s_eta)


@dataclass
class WindVARDiurnalParams(WindVARParams):
    """VAR(1) on deviations from a diurnal mean m(h) = mu + sum_j c_j cos(2 pi j h/24) + s_j sin(2 pi j h/24)."""

    harmonics: Optional[List[List[float]]] = None   # rows: cos1, sin1, cos2, sin2; columns: u, v


def _harmonic_design(hours: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * np.asarray(hours, dtype=np.float64) / 24.0
    return np.column_stack([np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])


def diurnal_mean(p: WindVARParams, hours: np.ndarray) -> np.ndarray:
    hours = np.asarray(hours)
    if getattr(p, "harmonics", None) is None:
        return np.tile(np.asarray(p.mu), (len(hours), 1))
    return np.asarray(p.mu) + _harmonic_design(hours) @ np.asarray(p.harmonics)


def fit_wind_var_diurnal(u_seg, v_seg, speed_seg, gust_seg, hour_seg) -> WindVARDiurnalParams:
    W = np.column_stack([np.concatenate(u_seg), np.concatenate(v_seg)])
    H = np.concatenate(hour_seg)
    X = np.column_stack([np.ones(len(H)), _harmonic_design(H)])
    beta = np.linalg.lstsq(X, W, rcond=None)[0]
    mu, harm = beta[0], beta[1:]
    Xa, Ya = [], []
    for u, v, h in zip(u_seg, v_seg, hour_seg):
        d = np.column_stack([u, v]) - (mu + _harmonic_design(h) @ harm)
        Xa.append(d[:-1])
        Ya.append(d[1:])
    Xa, Ya = np.concatenate(Xa), np.concatenate(Ya)
    A = np.linalg.lstsq(Xa, Ya, rcond=None)[0].T
    Sigma = np.cov((Ya - Xa @ A.T).T)
    base = fit_wind_var(u_seg, v_seg, speed_seg, gust_seg)
    return WindVARDiurnalParams(mu.tolist(), A.tolist(), Sigma.tolist(), base.gust_a, base.gust_b,
                                base.gust_phi, base.gust_sigma, harm.tolist())


def simulate_wind_var(p: WindVARParams, lengths, rng, start_hours=None) -> Dict[str, List[np.ndarray]]:
    A = np.asarray(p.A)
    L = np.linalg.cholesky(np.asarray(p.Sigma))
    L0 = np.linalg.cholesky(p.stationary_cov())
    s_eta0 = p.gust_sigma / np.sqrt(max(1e-9, 1 - p.gust_phi ** 2))
    out = {k: [] for k in ("speed", "direction", "u", "v", "gust")}
    for i, n in enumerate(lengths):
        h0 = 0 if start_hours is None else start_hours[i]
        m = diurnal_mean(p, (h0 + np.arange(n)) % 24)
        d = np.empty((n, 2))
        d[0] = L0 @ rng.standard_normal(2)
        e = rng.standard_normal((n, 2)) @ L.T
        for t in range(1, n):
            d[t] = A @ d[t - 1] + e[t]
        w = m + d
        eta = np.empty(n)
        eta[0] = s_eta0 * rng.standard_normal()
        xi = rng.standard_normal(n)
        for t in range(1, n):
            eta[t] = p.gust_phi * eta[t - 1] + p.gust_sigma * xi[t]
        speed = np.hypot(w[:, 0], w[:, 1])
        out["speed"].append(speed)
        out["direction"].append(np.degrees(np.arctan2(w[:, 0], w[:, 1])) % 360.0)
        out["u"].append(w[:, 0])
        out["v"].append(w[:, 1])
        out["gust"].append(np.maximum(0.0, p.gust_a + p.gust_b * speed + eta))
    return out


# V1 defaults as generated by mission.py (mean of the scenario ranges).
V1_DEFAULT_RAIN = RainV1Params(mean=0.525, variability=0.05 + 0.20 * 0.525, correlation=0.85, p_ref_mm_h=0.0)
V1_DEFAULT_WIND = WindV1Params(
    base_speed=3.0 + 9.0 * 0.6, speed_std=0.8 + 2.5 * 0.6, correlation=0.85, direction_drift_deg=15.0,
    gust_probability=0.05 + 0.25 * 0.6, min_gust=2.0, max_gust=3.0 + 8.0 * 0.6,
)


def as_dict(obj) -> dict:
    return asdict(obj)

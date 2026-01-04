import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2, t

from statsmodels.stats.diagnostic import acorr_ljungbox


# =========================
# CONFIG
# =========================
TRADING_DAYS = 252
RECALIBRATION_FREQ = 10      # ~2 semanas hábiles
LAMBDA = 1                   # forgetting factor
ALPHA_LEVEL = 0.01           # VaR 99% (cola izquierda 1%)
LB_LAGS = 20
ACTIVE = "NVDA"
# Rangos típicos (equities) para starting values
ALPHA_RANGE = (0.03, 0.10)
BETA_RANGE  = (0.85, 0.95)
GAMMA_RANGE = (0.05, 0.20)

# Starting conservador para retry
RETRY_DEFAULT = {
    "mu": 0.0,
    "omega": 0.01,
    "alpha": 0.05,
    "beta": 0.92,
    "gamma": 0.10,
    "nu": 8.0
}


# =========================
# DATA
# =========================
def download_prices(ticker=ACTIVE, start="2000-01-01", end=None) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError("No se descargaron datos. Revisa ticker/fechas/conexión.")

    # FIX: yfinance puede devolver MultiIndex (tuplas) en columnas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]   # primer nivel ("Adj Close", etc.)
    else:
        df.columns = [str(c).lower() for c in df.columns]

    if "adj close" in df.columns:
        price = df["adj close"]
    elif "close" in df.columns:
        price = df["close"]
    else:
        raise KeyError("No se encontró 'Adj Close' ni 'Close' en los datos descargados.")

    return price.dropna().rename("price")


def log_returns_pct(price: pd.Series) -> pd.Series:
    return (100.0 * np.log(price).diff().dropna()).rename("r_pct")


# =========================
# WEIGHTS + DISTRIBUTION
# =========================
def exp_weights(T: int, lam: float) -> np.ndarray:
    ages = np.arange(T - 1, -1, -1)  # antiguo...reciente
    w = lam ** ages
    return w / w.sum()


def student_t_logpdf(z, nu):
    return (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log((nu - 2) * np.pi)
        - ((nu + 1) / 2) * np.log(1 + (z**2) / (nu - 2))
    )


# =========================
# WEIGHTED GJR-GARCH(1,1) FIT
# =========================
def fit_gjr_garch_weighted(
    r_window_pct: np.ndarray,
    lam: float,
    start_params: dict | None = None,
    hard_start: dict | None = None
):
    r = r_window_pct.astype(float)
    T = len(r)
    w = exp_weights(T, lam)

    # Starting values
    if hard_start is not None:
        mu0 = float(hard_start["mu"])
        omega0 = max(float(hard_start["omega"]), 1e-6)
        alpha0 = float(np.clip(hard_start["alpha"], *ALPHA_RANGE))
        beta0  = float(np.clip(hard_start["beta"],  *BETA_RANGE))
        gamma0 = float(np.clip(hard_start["gamma"], *GAMMA_RANGE))
        nu0 = float(np.clip(hard_start["nu"], 3.0, 80.0))
    elif start_params is None:
        mu0 = 0.0
        omega0 = 0.01
        alpha0 = float(np.mean(ALPHA_RANGE))
        beta0  = float(np.mean(BETA_RANGE))
        gamma0 = float(np.mean(GAMMA_RANGE))
        nu0 = 8.0
    else:
        mu0 = float(start_params["mu"])
        omega0 = max(float(start_params["omega"]), 1e-6)
        alpha0 = float(np.clip(start_params["alpha"], *ALPHA_RANGE))
        beta0  = float(np.clip(start_params["beta"],  *BETA_RANGE))
        gamma0 = float(np.clip(start_params["gamma"], *GAMMA_RANGE))
        nu0 = float(np.clip(start_params["nu"], 3.0, 80.0))

    x0 = np.array([mu0, omega0, alpha0, gamma0, beta0, nu0], dtype=float)

    bnds = [
        (-2.0, 2.0),       # mu (%)
        (1e-8, 10.0),      # omega
        (0.0, 1.0),        # alpha
        (0.0, 1.0),        # gamma
        (0.0, 1.0),        # beta
        (2.05, 200.0),     # nu
    ]

    def stationarity_penalty(alpha, beta, gamma):
        val = alpha + beta + gamma / 2.0
        if val < 0.999:
            return 0.0
        return 1e6 * (val - 0.999) ** 2

    def nll(theta):
        mu, omega, alpha, gamma, beta, nu = theta
        if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0 or nu <= 2.0:
            return 1e12

        pen = stationarity_penalty(alpha, beta, gamma)

        eps = r - mu
        sigma2 = np.zeros(T, dtype=float)
        s2_0 = np.var(eps)
        sigma2[0] = s2_0 if s2_0 > 1e-6 else 1e-6

        for i in range(1, T):
            ind = 1.0 if eps[i - 1] < 0 else 0.0
            sigma2[i] = omega + (alpha + gamma * ind) * (eps[i - 1] ** 2) + beta * sigma2[i - 1]
            if sigma2[i] <= 1e-12:
                sigma2[i] = 1e-12

        z = eps / np.sqrt(sigma2)
        ll = student_t_logpdf(z, nu) - 0.5 * np.log(sigma2)

        return -(w * ll).sum() + pen

    opt = minimize(nll, x0, method="L-BFGS-B", bounds=bnds)

    mu, omega, alpha, gamma, beta, nu = opt.x

    # sigma2 y std_resid con params finales
    eps = r - mu
    sigma2 = np.zeros(T, dtype=float)
    s2_0 = np.var(eps)
    sigma2[0] = s2_0 if s2_0 > 1e-6 else 1e-6
    for i in range(1, T):
        ind = 1.0 if eps[i - 1] < 0 else 0.0
        sigma2[i] = omega + (alpha + gamma * ind) * (eps[i - 1] ** 2) + beta * sigma2[i - 1]
        sigma2[i] = max(sigma2[i], 1e-12)

    z = eps / np.sqrt(sigma2)
    stab_val = float(alpha + beta + gamma / 2.0)

    return {
        "mu": float(mu),
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta": float(beta),
        "nu": float(nu),
        "sigma2_last": float(sigma2[-1]),
        "std_resid": z,
        "stationarity_value": stab_val,
        "stationary_ok": bool(stab_val < 1.0),
        "opt_success": bool(opt.success),
        "opt_message": str(opt.message)
    }


def params_ok(p: dict | None) -> bool:
    if p is None:
        return False

    needed = ["mu", "omega", "alpha", "beta", "gamma", "nu",
              "sigma2_last", "stationary_ok", "opt_success"]
    if any(k not in p for k in needed):
        return False

    vals = [p["omega"], p["alpha"], p["beta"], p["gamma"], p["nu"], p["sigma2_last"]]
    if any((v is None) or (isinstance(v, float) and np.isnan(v)) for v in vals):
        return False

    if p["omega"] <= 0 or p["sigma2_last"] <= 0:
        return False
    if not (0 <= p["alpha"] <= 1 and 0 <= p["beta"] <= 1 and 0 <= p["gamma"] <= 1):
        return False
    if p["nu"] <= 2.05:
        return False
    if not p["stationary_ok"]:
        return False
    if not p["opt_success"]:
        return False

    return True


# =========================
# FORECAST / VAR
# =========================
def gjr_next_sigma2(eps_t, sigma2_t, omega, alpha, gamma, beta):
    ind = 1.0 if eps_t < 0 else 0.0
    return omega + (alpha + gamma * ind) * (eps_t**2) + beta * sigma2_t


def var_1d(mu, sigma, nu, alpha_level=0.01):
    q = t.ppf(alpha_level, df=nu)
    return mu + sigma * q


# =========================
# BACKTESTS (ROBUSTOS EN LOG-SPACE)
# =========================
def kupiec_uc_test(exceptions: np.ndarray, alpha: float):
    """
    Kupiec UC robusto: calcula LR en log-space para evitar underflow/división por cero.
    """
    T = int(exceptions.size)
    x = int(exceptions.sum())

    if T == 0:
        return np.nan, np.nan

    eps = 1e-12
    phat = x / T
    phat = min(max(phat, eps), 1 - eps)
    alpha_c = min(max(alpha, eps), 1 - eps)

    logL0 = (T - x) * np.log(1 - alpha_c) + x * np.log(alpha_c)
    logL1 = (T - x) * np.log(1 - phat) + x * np.log(phat)

    LR = -2.0 * (logL0 - logL1)
    pval = 1 - chi2.cdf(LR, df=1)
    return float(LR), float(pval)


def christoffersen_independence_test(exceptions: np.ndarray):
    """
    Christoffersen IND robusto en log-space.
    """
    e = exceptions.astype(int)
    if e.size < 2:
        return np.nan, np.nan

    n00 = np.sum((e[:-1] == 0) & (e[1:] == 0))
    n01 = np.sum((e[:-1] == 0) & (e[1:] == 1))
    n10 = np.sum((e[:-1] == 1) & (e[1:] == 0))
    n11 = np.sum((e[:-1] == 1) & (e[1:] == 1))

    eps = 1e-12
    def clip(p): return min(max(p, eps), 1 - eps)

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi1  = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    pi01, pi11, pi1 = clip(pi01), clip(pi11), clip(pi1)

    logL0 = (n00 + n10) * np.log(1 - pi1) + (n01 + n11) * np.log(pi1)
    logL1 = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
        n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )

    LR = -2.0 * (logL0 - logL1)
    pval = 1 - chi2.cdf(LR, df=1)
    return float(LR), float(pval)


def christoffersen_cc_test(exceptions: np.ndarray, alpha: float):
    LR_uc, _ = kupiec_uc_test(exceptions, alpha)
    LR_ind, _ = christoffersen_independence_test(exceptions)
    if np.isnan(LR_uc) or np.isnan(LR_ind):
        return np.nan, np.nan
    LR_cc = LR_uc + LR_ind
    pval = 1 - chi2.cdf(LR_cc, df=2)
    return float(LR_cc), float(pval)


# =========================
# ROLLING: recalib cada 2 semanas + forgetting factor + retry/fallback
# =========================
def rolling_recalib_2w_forgetting(r: pd.Series) -> pd.DataFrame:
    idx = r.index
    n = len(r)

    rows = []

    params = None
    params_prev_valid = None
    sigma2_t = None
    mu_t = None

    last_lb_stat = np.nan
    last_lb_pval = np.nan

    retries_used = 0
    fallback_used_days = 0
    recalib_days = 0

    for t in range(TRADING_DAYS, n - 1):
        recalibrate_today = ((t - TRADING_DAYS) % RECALIBRATION_FREQ == 0) or (params is None)

        used_retry = False
        used_fallback = False

        if recalibrate_today:
            recalib_days += 1
            r_win = r.iloc[t - TRADING_DAYS:t].values

            # 1) intento normal
            cand = fit_gjr_garch_weighted(r_win, lam=LAMBDA, start_params=params)

            # 2) retry conservador
            if not params_ok(cand):
                used_retry = True
                retries_used += 1
                cand = fit_gjr_garch_weighted(r_win, lam=LAMBDA, start_params=params, hard_start=RETRY_DEFAULT)

            # 3) fallback si sigue mal
            if params_ok(cand):
                params = cand
                params_prev_valid = cand
                sigma2_t = params["sigma2_last"]
                mu_t = params["mu"]

                z2 = params["std_resid"] ** 2
                lb = acorr_ljungbox(z2, lags=[LB_LAGS], return_df=True)
                last_lb_stat = float(lb["lb_stat"].iloc[0])
                last_lb_pval = float(lb["lb_pvalue"].iloc[0])
            else:
                used_fallback = True
                if params_prev_valid is not None:
                    params = params_prev_valid
                    # sigma2_t y mu_t quedan como estaban

        if params is None or sigma2_t is None or mu_t is None:
            fallback_used_days += 1
            rows.append({
                "date": idx[t],
                "r_next_pct": float(r.iloc[t + 1]),
                "mu_fc_pct": np.nan,
                "sigma_fc_pct": np.nan,
                "VaR_fc_pct": np.nan,
                "exception": np.nan,
                "alpha": np.nan, "beta": np.nan, "gamma": np.nan, "nu": np.nan,
                "alpha+beta+gamma/2": np.nan,
                "stationary_ok": np.nan,
                f"lb_stat_z2_lag{LB_LAGS}": np.nan,
                f"lb_pval_z2_lag{LB_LAGS}": np.nan,
                "recalibrated_today": recalibrate_today,
                "retry_used": used_retry,
                "fallback_used": True,
                "opt_success": np.nan,
                "lambda": LAMBDA
            })
            continue

        # Forecast
        r_t = float(r.iloc[t])
        eps_t = r_t - mu_t

        sigma2_next = gjr_next_sigma2(
            eps_t, sigma2_t,
            params["omega"], params["alpha"], params["gamma"], params["beta"]
        )
        sigma_next = float(np.sqrt(max(sigma2_next, 1e-12)))
        VaR_next = var_1d(mu_t, sigma_next, params["nu"], alpha_level=ALPHA_LEVEL)

        r_next = float(r.iloc[t + 1])
        exception = (r_next < VaR_next)

        sigma2_t = sigma2_next

        if used_fallback:
            fallback_used_days += 1

        rows.append({
            "date": idx[t],
            "r_next_pct": r_next,
            "mu_fc_pct": mu_t,
            "sigma_fc_pct": sigma_next,
            "VaR_fc_pct": VaR_next,
            "exception": exception,
            "alpha": params["alpha"],
            "beta": params["beta"],
            "gamma": params["gamma"],
            "nu": params["nu"],
            "alpha+beta+gamma/2": params["stationarity_value"],
            "stationary_ok": params["stationary_ok"],
            f"lb_stat_z2_lag{LB_LAGS}": last_lb_stat,
            f"lb_pval_z2_lag{LB_LAGS}": last_lb_pval,
            "recalibrated_today": recalibrate_today,
            "retry_used": used_retry,
            "fallback_used": used_fallback,
            "opt_success": params["opt_success"],
            "lambda": LAMBDA
        })

    out = pd.DataFrame(rows).set_index("date")
    out.attrs["recalib_days"] = recalib_days
    out.attrs["retries_used"] = retries_used
    out.attrs["fallback_used_days"] = fallback_used_days
    return out


# =========================
# PLOTS / REPORT
# =========================
def plot_last_year(df: pd.DataFrame):
    last = df.last("365D").dropna(subset=["sigma_fc_pct"])
    if last.empty:
        raise ValueError("No hay datos suficientes (último año) con forecasts válidos.")
    realized = last["r_next_pct"].abs()

    plt.figure(figsize=(12, 5))
    plt.plot(last.index, realized, label="Vol realizada proxy |r(t+1)| (%)")
    plt.plot(last.index, last["sigma_fc_pct"], label="Vol pronosticada 1d (σ) (%)")
    plt.title("Volatilidad diaria: realizada vs pronosticada (último año)")
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    ticker = ACTIVE
    start = "2000-01-01"

    px = download_prices(ticker=ticker, start=start)
    r = log_returns_pct(px)

    df = rolling_recalib_2w_forgetting(r)

    dfv = df.dropna(subset=["VaR_fc_pct", "exception"]).copy()
    exceptions = dfv["exception"].astype(bool).values

    # Diagnóstico rápido de cobertura
    T = len(exceptions)
    x = int(exceptions.sum()) if T else 0
    print(f"\nExcepciones: {x} de {T} | Tasa: {x/T:.4%} | Esperada: {ALPHA_LEVEL:.2%} (~{int(ALPHA_LEVEL*T)})" if T else "Sin datos válidos.")

    frac_nonstationary = 1.0 - dfv["stationary_ok"].mean() if len(dfv) else np.nan
    lb_col = [c for c in dfv.columns if c.startswith("lb_pval_z2")][0] if len(dfv) else None
    frac_lb_reject = (dfv[lb_col] < 0.05).mean() if (len(dfv) and lb_col) else np.nan

    LR_uc, p_uc = kupiec_uc_test(exceptions, ALPHA_LEVEL)
    LR_ind, p_ind = christoffersen_independence_test(exceptions)
    LR_cc, p_cc = christoffersen_cc_test(exceptions, ALPHA_LEVEL)

    recalib_days = df.attrs.get("recalib_days", np.nan)
    retries_used = df.attrs.get("retries_used", np.nan)
    fallback_used_days = df.attrs.get("fallback_used_days", np.nan)

    print("\n=== CONFIG ===")
    print(
        f"Ticker: {ticker} | Window: {TRADING_DAYS} | Recalib freq: {RECALIBRATION_FREQ}d | "
        f"lambda: {LAMBDA} | VaR alpha: {ALPHA_LEVEL} | Ljung-Box lags: {LB_LAGS}"
    )

    print("\n=== ROBUSTEZ ESTIMACIÓN ===")
    print(f"Días de recalibración: {recalib_days}")
    print(f"Retries usados (starting conservador): {retries_used}")
    print(f"Días con fallback: {fallback_used_days}")

    print("\n=== CHECK ESTACIONARIEDAD (GJR) ===")
    print("Condición: alpha + beta + gamma/2 < 1")
    print(f"Frac. NO estacionario (en días válidos): {frac_nonstationary:.2%}" if not np.isnan(frac_nonstationary) else "N/A")

    print("\n=== Ljung-Box sobre z_t^2 ===")
    print(f"Frac. rechazos (p<0.05): {frac_lb_reject:.2%}" if not np.isnan(frac_lb_reject) else "N/A")

    print("\n=== BACKTEST VaR ===")
    print(f"Kupiec UC:             LR={LR_uc:.4f}, p-value={p_uc:.4f}")
    print(f"Christoffersen IND:    LR={LR_ind:.4f}, p-value={p_ind:.4f}")
    print(f"Christoffersen CC:     LR={LR_cc:.4f}, p-value={p_cc:.4f}")

    print("\n=== TABLA (últimas 30 filas válidas) ===")
    cols = [
        "r_next_pct", "sigma_fc_pct", "VaR_fc_pct", "exception",
        "alpha", "beta", "gamma", "nu", "alpha+beta+gamma/2",
        "stationary_ok", lb_col, "recalibrated_today", "retry_used", "fallback_used"
    ]
    print(dfv[cols].tail(30).to_string())

    plot_last_year(dfv)


if __name__ == "__main__":
    main()

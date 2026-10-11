"""Graphs of the Lipofectamine fit.

Reads the annealed parameters from the same seed as
lipoplex_simulated_annealing.py and writes three figures:

    figures/dose_response.png        plasmids/nucleus and luciferase vs dose
    figures/nuclear_fraction.png     percent of dose that reaches the nucleus
    figures/time_courses.png         cytoplasm, nucleus, and luciferase vs time

Points are the digitized Cohen et al. 2009 Figure 4 values. Curves are
Eqs. 1–20 at the annealed (p_entry, K_sat), with luciferase scaled by the
profiled assay factor kappa.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
from scipy.integrate import solve_ivp

from lipoplex_simulated_annealing import (
    A549,
    B16F10,
    HARVEST_H,
    INCUBATION_H,
    N_PACK,
    N_STATES,
    AnnealingResult,
    CellLine,
    Parameters,
    anneal,
    derivatives,
    report,
    simulate_endpoint,
)

FIGURES = Path(__file__).resolve().parent / "figures"

DATA_COLOR = "#111111"
MODEL_COLOR = "#0B6E4F"
WASH_COLOR = "#888888"


def _dose_grid(cell_line: CellLine, n: int = 48) -> np.ndarray:
    low = float(cell_line.dose_plasmids_per_cell.min()) / 1.6
    high = float(cell_line.dose_plasmids_per_cell.max()) * 1.6
    return np.logspace(math.log10(low), math.log10(high), n)


def model_curve(cell_line: CellLine, theta: Parameters, kappa: float,
                 doses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nuclear plasmids and kappa*[R*] on a dose grid."""
    nuclear = np.empty(len(doses))
    luciferase = np.empty(len(doses))
    for i, dose in enumerate(doses):
        n_hat, mature = score_endpoint(float(dose), theta, cell_line.k_div)
        nuclear[i] = n_hat
        luciferase[i] = kappa * mature
    return nuclear, luciferase


def score_endpoint(dose: float, theta: Parameters, k_div: float) -> tuple[float, float]:
    return simulate_endpoint(dose, theta, k_div)


def trajectory(dose: float, theta: Parameters, k_div: float,
               n_incubation: int = 80, n_chase: int = 160
               ) -> tuple[np.ndarray, np.ndarray]:
    """Full state from 0 to 24 h, with [L_ex] set to 0 at the 3 h wash."""
    y0 = np.zeros(N_STATES)
    y0[0] = dose / N_PACK
    args = (theta.p_entry, theta.k_TX, theta.K_sat, k_div)
    t_wash = np.linspace(0.0, INCUBATION_H, n_incubation)
    first = solve_ivp(
        derivatives, (0.0, INCUBATION_H), y0, t_eval=t_wash,
        method="LSODA", rtol=1.0e-5, atol=1.0e-6, args=args,
    )
    if not first.success:
        raise RuntimeError(first.message)
    y_wash = first.y[:, -1].copy()
    y_wash[0] = 0.0
    t_chase = np.linspace(INCUBATION_H, HARVEST_H, n_chase)
    second = solve_ivp(
        derivatives, (INCUBATION_H, HARVEST_H), y_wash, t_eval=t_chase,
        method="LSODA", rtol=1.0e-5, atol=1.0e-6, args=args,
    )
    if not second.success:
        raise RuntimeError(second.message)
    # Drop the duplicated wash sample so the Lex jump is a single step.
    t = np.concatenate([first.t, second.t[1:]])
    y = np.concatenate([first.y, second.y[:, 1:]], axis=1)
    return t, y


def _log_ticks(vmin: float, vmax: float, mantissas: tuple[float, ...]) -> list[float]:
    exp_lo = math.floor(math.log10(vmin)) - 1
    exp_hi = math.ceil(math.log10(vmax)) + 1
    ticks = []
    for exp in range(exp_lo, exp_hi + 1):
        for mantissa in mantissas:
            value = mantissa * 10.0 ** exp
            if vmin <= value <= vmax:
                ticks.append(value)
    return ticks


def _log_label(value: float, _pos: float) -> str:
    """1×10^n as 10^n, and 2×10^n / 5×10^n with an integer power."""
    if value <= 0.0:
        return ""
    exponent = math.floor(math.log10(value) + 1.0e-12)
    mantissa = value / 10.0 ** exponent
    for target in (1.0, 2.0, 5.0):
        if abs(mantissa - target) / target < 0.02:
            if target == 1.0:
                return rf"$10^{{{exponent}}}$"
            return rf"${target:.0f}\times10^{{{exponent}}}$"
    return ""


def _format_log_axis(axis) -> None:
    """Replace Matplotlib's default log labels.

    On a short range the default formatter labels every minor tick, and those
    labels collide. It also prints values such as 2×10^4 as 10^4.30.
    """
    vmin, vmax = axis.get_view_interval()
    if vmin <= 0.0 or vmax <= vmin:
        return
    decades = math.log10(vmax / vmin)
    mantissas = (1.0,) if decades >= 1.8 else (1.0, 2.0, 5.0)
    ticks = _log_ticks(vmin, vmax, mantissas)
    if len(ticks) < 2:
        ticks = _log_ticks(vmin, vmax, (1.0, 2.0, 5.0))
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(_log_label))
    axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.set_minor_formatter(NullFormatter())


def _style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
    })


def _log_panel(ax: plt.Axes, doses_obs: np.ndarray, observed: np.ndarray,
               doses_model: np.ndarray, modeled: np.ndarray,
               xlabel: str, ylabel: str, title: str) -> None:
    ax.loglog(doses_model, modeled, color=MODEL_COLOR, lw=2.0, label="Eqs. 1–20",
              zorder=2)
    ax.loglog(doses_obs, observed, "o", color=DATA_COLOR, ms=8, label="Cohen Fig. 4",
              zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    _format_log_axis(ax.xaxis)
    _format_log_axis(ax.yaxis)


def plot_dose_response(fits: list[tuple[CellLine, AnnealingResult]],
                       path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)
    for row, (cell, result) in enumerate(fits):
        theta = result.parameters
        kappa = result.score.kappa
        grid = _dose_grid(cell)
        nuclear, luciferase = model_curve(cell, theta, kappa, grid)
        doses = cell.dose_plasmids_per_cell
        _log_panel(
            axes[row, 0], doses, cell.plasmids_per_nucleus, grid, nuclear,
            "dose (plasmids/cell)", "plasmids/nucleus",
            f"{cell.name} nucleus   p_entry = {theta.p_entry:.3f}",
        )
        _log_panel(
            axes[row, 1], doses, cell.luciferase_rlu_per_mg, grid, luciferase,
            "dose (plasmids/cell)", "luciferase (RLU/mg)",
            f"{cell.name} luciferase   K_sat = {theta.K_sat:.0f}",
        )
    fig.suptitle(
        "Lipofectamine at 24 h. Curves use the annealed parameters; "
        "luciferase includes the profiled assay factor.",
        fontsize=12,
    )
    fig.savefig(path)
    plt.close(fig)


def plot_nuclear_fraction(fits: list[tuple[CellLine, AnnealingResult]],
                          path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
    for ax, (cell, result) in zip(axes, fits):
        theta = result.parameters
        grid = _dose_grid(cell)
        nuclear, _ = model_curve(cell, theta, result.score.kappa, grid)
        doses = cell.dose_plasmids_per_cell
        observed_pct = 100.0 * cell.plasmids_per_nucleus / doses
        model_pct = 100.0 * nuclear / grid
        ax.semilogx(grid, model_pct, color=MODEL_COLOR, lw=2.0, label="Eqs. 1–20")
        ax.semilogx(doses, observed_pct, "o", color=DATA_COLOR, ms=8,
                    label="Cohen Fig. 4")
        ax.set_xlabel("dose (plasmids/cell)")
        ax.set_ylabel("nuclear plasmids (% of dose)")
        ax.set_title(f"{cell.name}   p_entry = {theta.p_entry:.3f}")
        ax.legend(frameon=False)
        _format_log_axis(ax.xaxis)
    fig.suptitle(
        "Eqs. 1–11 put a constant fraction of the dose in the nucleus.",
        fontsize=12,
    )
    fig.savefig(path)
    plt.close(fig)


def plot_time_courses(fits: list[tuple[CellLine, AnnealingResult]],
                      path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.2), sharex=True,
                             constrained_layout=True)
    quantity = (
        (4, "cytoplasmic plasmids", False),
        (5, "nuclear plasmids", False),
        (8, "luciferase (RLU/mg)", True),
    )
    dose_colors = ("#4C78A8", "#F58518", "#54A24B")
    for row, (cell, result) in enumerate(fits):
        theta = result.parameters
        kappa = result.score.kappa
        for dose, color in zip(cell.dose_plasmids_per_cell, dose_colors):
            t, y = trajectory(float(dose), theta, cell.k_div)
            label = f"{dose:.3g}"
            for col, (index, _ylabel, scale) in enumerate(quantity):
                series = kappa * y[index] if scale else y[index]
                axes[row, col].plot(t, series, color=color, lw=1.8, label=label)
        for col, (_index, ylabel, _scale) in enumerate(quantity):
            ax = axes[row, col]
            ax.axvline(INCUBATION_H, color=WASH_COLOR, ls="--", lw=1.0)
            ax.set_xlim(0.0, HARVEST_H)
            ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("time (h)")
            if col == 0:
                ax.set_title(f"{cell.name}", loc="left")
            if col == 2:
                ax.legend(title="dose (plasmids/cell)", frameon=False)
    fig.suptitle(
        "Trajectories at the annealed parameters. Dashed line is the 3 h wash.",
        fontsize=12,
    )
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    _style()
    FIGURES.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    fits: list[tuple[CellLine, AnnealingResult]] = []
    for cell_line in (B16F10, A549):
        result = anneal(cell_line, rng)
        report(cell_line, result)
        fits.append((cell_line, result))

    plot_dose_response(fits, FIGURES / "dose_response.png")
    plot_nuclear_fraction(fits, FIGURES / "nuclear_fraction.png")
    plot_time_courses(fits, FIGURES / "time_courses.png")
    for name in ("dose_response.png", "nuclear_fraction.png", "time_courses.png"):
        print(f"wrote {FIGURES / name}")


if __name__ == "__main__":
    main()

"""
Simulated annealing for the reduced lipoplex transfection model.

The kinetics are Stanford iGEM 2026, "Multi-Level Kinetic Model of Plasmid
Delivery via Transfection of Lipoplexes." Equation numbers below are that
note's. Eqs. 1–20 are the streamlined system that is integrated. Eqs. 21–26
are the multi-lipoplex extension, and Eqs. 27–28 are the non-dividing limit.
Ligon et al. already fitted the lipoplex delivery rates (their Table 1,
multiple-lipoplex model with slow maturation). Those stay fixed. This fit
is for plasmid DNA, so simulated annealing searches only the three
parameters the note leaves free: p_entry, k_TX, and K_sat. The solver is
Eqs. 1–20.

    p_entry    probability a cytoplasmic plasmid is captured at mitosis
    k_TX       maximum effective transcript-production rate
    K_sat      nuclear plasmid copy number at half-maximal production

They are scored on the two 24 h Lipofectamine observables in Cohen et al.,
J. Control. Release 135:166–174 (2009), Figure 4: plasmids per nucleus, and
luciferase luminescence. Polyethylenimine (polyplex) points are not used.

The search itself is the Metropolis simulated-annealing algorithm of
Kirkpatrick, Gelatt, and Vecchi, Science 220:671–680 (1983), which is the
procedure Ligon et al., PLoS ONE 9:e107148 (2014), used for lipoplex kinetics
and which the iGEM note follows.

Requires: numpy, scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Experimental protocol and fixed rates
# ---------------------------------------------------------------------------

# Cohen et al., Methods: complexes are left on the cells for 3 h, the medium
# is replaced, and nuclei and luciferase are read 24 h after transfection
# (3 h exposure + 21 h chase).
INCUBATION_H = 3.0
HARVEST_H = 24.0

# Ligon et al., PLoS ONE 9:e107148 (2014), Table 1, column
# "multiple-lipoplex with slow maturation." Already fitted for lipoplexes.
# Not annealed. The iGEM Table 1 quotes these fitted values.
K_A = 0.27       # h^-1, attachment
K_E = 0.81       # h^-1, endocytosis
K_L = 0.11       # h^-1, endosomal escape
D_E = 0.67       # h^-1, endosome degradation
N_PACK = 3.2     # plasmids per complex, Schwake et al.; iGEM Table 1
K_TL = 170.0     # h^-1, translation
K_M = 5.5        # h^-1, maturation (Ligon slow-maturation column)
D_M = 0.062      # h^-1, transcript degradation
D_G = 0.056      # h^-1, protein degradation

# Eq. 4. Table 1: unpacking is the mRNA value, "set instantaneous", and
# pre-escape packaged loss is "negligible". Both still appear in the ODE.
K_U = 1.0e6      # h^-1
D_L = 1.0e-6     # h^-1

# Eqs. 10 and 11. Table 1 does not assign plasmid degradation, so both
# coefficients are zero and the terms remain in the balances.
D_PL = 0.0
D_PL_NUC = 0.0

# Eq. 1. Table 1 does not give k_W. The coefficient is zero; Cohen's medium
# change is the separate condition [L_ex](3 h) = 0.
K_W = 0.0        # h^-1

# Eq. 24 and the text after Eq. 26. Occupancy runs through 10. The note does
# not assign "number of pits" in Eq. 22, so that factor stays an argument of
# the multi-lipoplex system and is not used in the three-parameter fit.
OCCUPANCY_MAX = 10


# ---------------------------------------------------------------------------
# Lipofectamine dose–response (Cohen et al. 2009, Figure 4)
# ---------------------------------------------------------------------------

# Digitized from the published log-log plots. The paper has no numeric table;
# these are the values in Fig4_transfection_values.xlsx. Blank PEI cells in
# that sheet are unused.


@dataclass(frozen=True)
class CellLine:
    """One Lipofectamine series. k_div is fixed by the doubling time (Eq. 5)."""

    name: str
    doubling_time_h: float
    doubling_time_source: str
    dose_plasmids_per_cell: np.ndarray
    plasmids_per_nucleus: np.ndarray
    luciferase_rlu_per_mg: np.ndarray

    @property
    def k_div(self) -> float:
        # Eq. 5, repeated as Eq. 12. Eq. 6 (2 = exp(k_div * T_doubling)) is
        # the defining identity, not a separate check: substituting k_div
        # back in always returns exp(ln 2) = 2 to float precision for any
        # positive doubling_time_h, so there is nothing here left to verify.
        return math.log(2.0) / self.doubling_time_h


B16F10 = CellLine(
    name="B16F10",
    # Parental B16-F10, not the faster Luc2 derivative.
    doubling_time_h=17.2,
    doubling_time_source="Danciu et al., Cancer Cell Int. 13:75 (2013)",
    dose_plasmids_per_cell=np.array([2.0e4, 1.0e5, 2.0e5]),
    plasmids_per_nucleus=np.array([200.0, 500.0, 1200.0]),
    luciferase_rlu_per_mg=np.array([1.7e10, 5.0e10, 6.0e10]),
)

A549 = CellLine(
    name="A549",
    doubling_time_h=22.0,
    doubling_time_source="ATCC CCL-185, population doubling time about 22 h",
    dose_plasmids_per_cell=np.array([2.0e3, 2.0e4, 2.0e5]),
    plasmids_per_nucleus=np.array([50.0, 85.0, 1000.0]),
    luciferase_rlu_per_mg=np.array([1.8e9, 8.5e10, 1.6e11]),
)


# ---------------------------------------------------------------------------
# Free-parameter bounds and sampling measure
# ---------------------------------------------------------------------------

# Eq. 8. Dimensionless, so the search measure is uniform.
P_ENTRY_MIN = 0.0
P_ENTRY_MAX = 1.0

# Fitting section: 10^{-2} <= k_TX <= 10^3 h^-1. Numerical limits, not a
# measured promoter range. Log-uniform on a positive rate.
K_TX_MIN = 1.0e-2
K_TX_MAX = 1.0e3

# Half-saturation. The note expects ~10^3–3*10^3 plasmids/nucleus from Cohen.
# The search interval below is the one specified for this fit: log-uniform
# from 10^2 to 10^4, which covers that transition.
K_SAT_MIN = 1.0e2
K_SAT_MAX = 1.0e4

# Random-walk widths. Uniform in p_entry, log-uniform in the two rates.
SIGMA_P_ENTRY = 0.08
SIGMA_LN_K_TX = 0.30
SIGMA_LN_K_SAT = 0.20

# Geometric cooling. T starts hot enough that uphill moves are usually
# accepted, then falls by this factor after each temperature level.
INITIAL_UPHILL_ACCEPTANCE = 0.80
COOLING_RATE = 0.95
STEPS_PER_TEMPERATURE = 25
N_TEMPERATURES = 70


@dataclass(frozen=True)
class Parameters:
    """The three DNA-specific parameters. Delivery rates stay at Ligon's values."""

    p_entry: float
    k_TX: float
    K_sat: float

    def clamped(self) -> Parameters:
        return Parameters(
            p_entry=min(max(self.p_entry, P_ENTRY_MIN), P_ENTRY_MAX),
            k_TX=min(max(self.k_TX, K_TX_MIN), K_TX_MAX),
            K_sat=min(max(self.K_sat, K_SAT_MIN), K_SAT_MAX),
        )


def sample_parameters(rng: np.random.Generator) -> Parameters:
    """Draw θ: uniform p_entry, log-uniform k_TX and K_sat."""
    return Parameters(
        p_entry=float(rng.uniform(P_ENTRY_MIN, P_ENTRY_MAX)),
        k_TX=math.exp(rng.uniform(math.log(K_TX_MIN), math.log(K_TX_MAX))),
        K_sat=math.exp(rng.uniform(math.log(K_SAT_MIN), math.log(K_SAT_MAX))),
    )


def reflect(value: float, lower: float, upper: float) -> float:
    """Fold `value` into [lower, upper] by reflection.

    Reflection is its own inverse, so a Gaussian step followed by reflection
    is a symmetric proposal. The Hastings ratio is 1 and the Metropolis rule
    below needs only the energy difference.
    """
    span = upper - lower
    period = 2.0 * span
    folded = math.fmod(value - lower, period)
    if folded < 0.0:
        folded += period
    if folded > span:
        folded = period - folded
    return lower + folded


def propose(current: Parameters, rng: np.random.Generator) -> Parameters:
    """One random-walk step in the sampling coordinates.

    p'       = reflect(p + σ_p ξ_1)
    ln k'_TX = reflect(ln k_TX + σ_k ξ_2)
    ln K'    = reflect(ln K_sat + σ_K ξ_3)

    with ξ_i standard normal.
    """
    shock = rng.normal(size=3)
    p_entry = reflect(
        current.p_entry + SIGMA_P_ENTRY * shock[0],
        P_ENTRY_MIN,
        P_ENTRY_MAX,
    )
    ln_k_TX = reflect(
        math.log(current.k_TX) + SIGMA_LN_K_TX * shock[1],
        math.log(K_TX_MIN),
        math.log(K_TX_MAX),
    )
    ln_K_sat = reflect(
        math.log(current.K_sat) + SIGMA_LN_K_SAT * shock[2],
        math.log(K_SAT_MIN),
        math.log(K_SAT_MAX),
    )
    return Parameters(p_entry, math.exp(ln_k_TX), math.exp(ln_K_sat))


# ---------------------------------------------------------------------------
# Kinetics, Eqs. 1–28
# ---------------------------------------------------------------------------

# Streamlined state order (delivery phase through expression phase):
#   0 external lipoplex [L_ex]          complexes / cell
#   1 surface vesicle [V]               complexes / cell
#   2 endosome [E]                      complexes / cell
#   3 internal escaped lipoplex [L_in]  complexes / cell
#   4 cytoplasmic plasmid [PL_cyt]      plasmids / cell
#   5 nuclear plasmid [PL_nuc]          plasmids / nucleus
#   6 transcript [T]
#   7 immature luciferase [R]
#   8 mature luciferase [R*]
N_STATES = 9


def entry_bounds_hold(p_entry: float) -> bool:
    """Eq. 8. 0 ≤ p_entry ≤ 1."""
    return P_ENTRY_MIN <= p_entry <= P_ENTRY_MAX


def mitotic_entry_flux(k_div: float, p_entry: float, pl_cyt: float) -> float:
    """Eq. 7. v_nuc,mit = k_div · p_entry · [PL_cyt]."""
    return k_div * p_entry * pl_cyt


def nuclear_entry_rate(k_div: float, p_entry: float) -> float:
    """Eq. 9. k_entry,eff = k_div · p_entry."""
    return k_div * p_entry


def saturation_fraction(pl_nuc: float, K_sat: float) -> float:
    """Occupancy term of Eq. 13. Eq. 17 is this fraction in the high-copy limit."""
    return pl_nuc / (K_sat + pl_nuc)


def transcript_rate_low_copy(k_TX: float, K_sat: float, pl_nuc: float,
                             transcript: float) -> float:
    """Eqs. 14 and 15. Valid when [PL_nuc] ≪ K_sat."""
    return (k_TX / K_sat) * pl_nuc - D_M * transcript


def transcript_rate_high_copy(k_TX: float, transcript: float) -> float:
    """Eqs. 16 and 18. Valid when [PL_nuc] ≫ K_sat, where Eq. 17 sends the
    saturation fraction to 1."""
    return k_TX - D_M * transcript


def nondividing_mitotic_flux(p_entry: float, pl_cyt: float) -> float:
    """Eqs. 27 and 28. In a non-dividing population k_div → 0, so
    v_nuc,mit = k_div · p_entry · [PL_cyt] → 0."""
    k_div = 0.0
    return mitotic_entry_flux(k_div, p_entry, pl_cyt)


def derivatives(t: float, y: np.ndarray, p_entry: float, k_TX: float,
                K_sat: float, k_div: float) -> np.ndarray:
    """Streamlined lipoplex system, Eqs. 1–4, 7, 9–11, 13, 19, and 20.

    `t` is unused: the vector field is autonomous between washes.
    """
    lex, vesicle, endosome, lin, pl_cyt, pl_nuc, transcript, immature, mature = y

    # Eq. 9, then Eq. 7: v_nuc,mit = k_entry,eff · [PL_cyt].
    k_entry = nuclear_entry_rate(k_div, p_entry)
    v_nuc_mit = k_entry * pl_cyt

    # Eq. 1. External lipoplex.
    d_lex = -K_A * lex - K_W * lex

    # Eq. 2. Surface-bound vesicle.
    d_vesicle = K_A * lex - K_E * vesicle

    # Eq. 3. Endosome.
    d_endosome = K_E * vesicle - K_L * endosome - D_E * endosome

    # Eq. 4. Internal escaped lipoplex, before unpacking.
    d_lin = K_L * endosome - K_U * lin - D_L * lin

    # Eq. 10. Cytoplasmic plasmid. The entry flux is Eq. 7.
    d_pl_cyt = N_PACK * K_U * lin - D_PL * pl_cyt - v_nuc_mit

    # Eq. 11. Nuclear plasmid.
    d_pl_nuc = v_nuc_mit - D_PL_NUC * pl_nuc

    # Eq. 13. Saturating transcription. Eqs. 15 and 18 are its limits.
    d_transcript = k_TX * saturation_fraction(pl_nuc, K_sat) - D_M * transcript

    # Eq. 19. Translation to immature luciferase.
    d_immature = K_TL * transcript - K_M * immature - D_G * immature

    # Eq. 20. Maturation. Luminescence is proportional to [R*].
    d_mature = K_M * immature - D_G * mature

    return np.array([
        d_lex,
        d_vesicle,
        d_endosome,
        d_lin,
        d_pl_cyt,
        d_pl_nuc,
        d_transcript,
        d_immature,
        d_mature,
    ])


def attachment_rate_per_pit(n_pits: float) -> float:
    """Eq. 22. k_AX = k_A / (number of pits + 1)."""
    return K_A / (n_pits + 1.0)


def multi_lipoplex_derivatives(t: float, y: np.ndarray, p_entry: float,
                               k_TX: float, K_sat: float, k_div: float,
                               n_pits: float) -> np.ndarray:
    """Occupancy-resolved delivery, Eqs. 21–26, feeding Eqs. 10, 11, 13, 19, 20.

    State layout:
        [L_ex],
        P_1 … P_10,          pits of occupancy i
        E_1 … E_10,          endosomes of occupancy i
        [L_in], [PL_cyt], [PL_nuc], [T], [R], [R*]

    Eq. 26 replaces Eq. 4. The note's fitting section calibrates the
    streamlined system, so annealing does not call this function. `n_pits`
    is Eq. 22's "number of pits", which the note leaves unspecified.
    """
    if len(y) != 1 + 2 * OCCUPANCY_MAX + 6:
        raise ValueError("multi-lipoplex state has the wrong length")

    lex = float(y[0])
    pits = y[1:1 + OCCUPANCY_MAX]
    endosomes = y[1 + OCCUPANCY_MAX:1 + 2 * OCCUPANCY_MAX]
    lin = float(y[1 + 2 * OCCUPANCY_MAX])
    pl_cyt = float(y[2 + 2 * OCCUPANCY_MAX])
    pl_nuc = float(y[3 + 2 * OCCUPANCY_MAX])
    transcript = float(y[4 + 2 * OCCUPANCY_MAX])
    immature = float(y[5 + 2 * OCCUPANCY_MAX])
    mature = float(y[6 + 2 * OCCUPANCY_MAX])

    # Eq. 22.
    k_AX = attachment_rate_per_pit(n_pits)
    dy = np.zeros_like(y, dtype=float)

    # Eq. 21. Sum is i = 1 … 9, so P_10 does not bind another lipoplex.
    occupied = float(np.sum(pits[:OCCUPANCY_MAX - 1]))
    dy[0] = -k_AX * lex - k_AX * lex * occupied - K_W * lex

    # Eq. 23. Pits of occupancy 1.
    dy[1] = k_AX * lex - k_AX * lex * pits[0] - K_E * pits[0]

    # Eq. 24. i = 1 … 9 updates P_2 … P_10. P_10 (the last iteration) does
    # not bind another lipoplex (see Eq. 21's comment above), so it has no
    # k_AX*lex*pit_next outflow term; without this guard mass was draining
    # out of P_10 into nothing, leaking 88-99.5% of complexes by 24 h.
    for i in range(1, OCCUPANCY_MAX):
        pit_i = pits[i - 1]
        pit_next = pits[i]
        binds_again = i < OCCUPANCY_MAX - 1
        outflow = k_AX * lex * pit_next if binds_again else 0.0
        dy[1 + i] = k_AX * lex * pit_i - outflow - K_E * pit_next

    # Eq. 25. Endosomes of occupancy i = 1 … 10.
    endosome_origin = 1 + OCCUPANCY_MAX
    for i in range(OCCUPANCY_MAX):
        dy[endosome_origin + i] = (
            K_E * pits[i] - K_L * endosomes[i] - D_E * endosomes[i]
        )

    # Eq. 26. Released lipoplex, weighted by occupancy.
    released = 0.0
    for i in range(OCCUPANCY_MAX):
        occupancy = i + 1
        released += occupancy * K_L * endosomes[i]
    lin_index = 1 + 2 * OCCUPANCY_MAX
    dy[lin_index] = released - K_U * lin - D_L * lin

    # Eq. 7 into Eqs. 10 and 11, then the expression phase.
    v_nuc_mit = mitotic_entry_flux(k_div, p_entry, pl_cyt)
    dy[lin_index + 1] = N_PACK * K_U * lin - D_PL * pl_cyt - v_nuc_mit
    dy[lin_index + 2] = v_nuc_mit - D_PL_NUC * pl_nuc
    dy[lin_index + 3] = k_TX * saturation_fraction(pl_nuc, K_sat) - D_M * transcript
    dy[lin_index + 4] = K_TL * transcript - K_M * immature - D_G * immature
    dy[lin_index + 5] = K_M * immature - D_G * mature
    return dy


def _integrate_leg(y0: np.ndarray, t0: float, t1: float, theta: Parameters,
                   k_div: float) -> np.ndarray:
    """Integrate Eqs. 1–4, 7, and 9–20 over one uninterrupted interval."""
    solution = solve_ivp(
        derivatives,
        (t0, t1),
        y0,
        method="LSODA",
        rtol=1.0e-5,
        atol=1.0e-6,
        args=(theta.p_entry, theta.k_TX, theta.K_sat, k_div),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1]


def simulate_endpoint(dose_plasmids_per_cell: float, theta: Parameters,
                      k_div: float) -> tuple[float, float]:
    """([PL_nuc], [R*]) at 24 h from Eqs. 1–20.

    Ligon et al. set k_W = 0 during incubation and remove external lipoplex
    when the cells are washed. Cohen et al. wash at 3 h, so [L_ex](3 h) = 0.
    The initial complex count is the applied plasmid dose divided by n.
    """
    if dose_plasmids_per_cell < 0.0:
        raise ValueError("dose must be non-negative")

    y0 = np.zeros(N_STATES)
    y0[0] = dose_plasmids_per_cell / N_PACK

    at_wash = _integrate_leg(y0, 0.0, INCUBATION_H, theta, k_div).copy()
    at_wash[0] = 0.0
    at_harvest = _integrate_leg(at_wash, INCUBATION_H, HARVEST_H, theta, k_div)
    return float(at_harvest[5]), float(at_harvest[8])


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

# A failed integration, or a non-positive prediction, is treated as an
# unreachable point. The value only has to be much larger than any real
# squared log error (~10 for a factor-of-e^3 miss).
FAILED_ENERGY = 1.0e6


@dataclass(frozen=True)
class Score:
    """Concentrated log-space mismatch to one cell line."""

    total: float
    nuclear: float
    luciferase: float
    # Assay factor in the statement "luminescence = κ [R*]" (text after Eq. 20).
    # Closed form is derived in `score`.
    kappa: float
    nuclear_prediction: np.ndarray
    mature_luciferase: np.ndarray


def score(theta: Parameters, cell_line: CellLine) -> Score:
    """Mean squared log error against Figure 4.

    Nuclear plasmids are an absolute observation, so

        J_nuc = mean_d (ln N_model,d − ln N_obs,d)^2.          (plasmids)

    Luminescence is proportional to mature luciferase, not equal to it
    (text after Eq. 20). κ is an assay calibration, not a fourth biological
    parameter. For Gaussian error on ln(RLU) the maximizer is

        ln κ̂ = mean_d (ln L_obs,d − ln [R*]_d),

    and the concentrated luciferase term is

        J_luc = mean_d (ln [R*]_d + ln κ̂ − ln L_obs,d)^2.

    The annealed energy is J = J_nuc + J_luc. Both terms are mean squared
    log residuals, so a given fold-error in either observable costs the same.

    Eqs. 13, 19, and 20 are linear in k_TX, and nuclear entry does not depend
    on it. Therefore [R*] scales exactly with k_TX, ln κ̂ shifts by the
    opposite amount, and J does not depend on k_TX. A single 24 h endpoint
    identifies p_entry and K_sat. It identifies the product κ k_TX, not k_TX
    itself. The chain still proposes k_TX inside its interval, because that
    is the search written in the note; the flat direction is reported rather
    than hidden.
    """
    n_doses = len(cell_line.dose_plasmids_per_cell)
    nuclear_hat = np.empty(n_doses)
    mature_hat = np.empty(n_doses)

    try:
        for i, dose in enumerate(cell_line.dose_plasmids_per_cell):
            nuclear_hat[i], mature_hat[i] = simulate_endpoint(
                float(dose), theta, cell_line.k_div
            )
    except RuntimeError:
        return Score(FAILED_ENERGY, FAILED_ENERGY, FAILED_ENERGY, math.nan,
                     nuclear_hat, mature_hat)

    if np.any(nuclear_hat <= 0.0) or np.any(mature_hat <= 0.0):
        return Score(FAILED_ENERGY, FAILED_ENERGY, FAILED_ENERGY, math.nan,
                     nuclear_hat, mature_hat)

    log_nuclear_residual = np.log(nuclear_hat) - np.log(cell_line.plasmids_per_nucleus)
    nuclear_energy = float(np.mean(log_nuclear_residual ** 2))

    log_ratio = np.log(cell_line.luciferase_rlu_per_mg) - np.log(mature_hat)
    log_kappa = float(np.mean(log_ratio))
    luciferase_residual = np.log(mature_hat) + log_kappa - np.log(
        cell_line.luciferase_rlu_per_mg
    )
    luciferase_energy = float(np.mean(luciferase_residual ** 2))

    return Score(
        total=nuclear_energy + luciferase_energy,
        nuclear=nuclear_energy,
        luciferase=luciferase_energy,
        kappa=math.exp(log_kappa),
        nuclear_prediction=nuclear_hat,
        mature_luciferase=mature_hat,
    )


# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------

def accept(delta_energy: float, temperature: float, uniform_draw: float) -> bool:
    """Boltzmann acceptance. Kirkpatrick, Gelatt & Vecchi (1983).

    A(ΔJ, T) = 1                        if ΔJ <= 0
             = exp(−ΔJ / T)             if ΔJ > 0

    Accept the proposal when u < A, with u drawn from Uniform(0, 1).
    Downhill steps have A > 1, so they are always accepted.
    """
    if delta_energy <= 0.0:
        return True
    log_acceptance = -delta_energy / temperature
    # A draw can never be < exp(−700). Skip the overflow.
    if log_acceptance < -700.0:
        return False
    return uniform_draw < math.exp(log_acceptance)


def _initial_temperature(start: Parameters, cell_line: CellLine,
                         rng: np.random.Generator) -> float:
    """Pick T_0 so uphill proposals are accepted with the target probability.

    Kirkpatrick's hot start: the mean Boltzmann factor of uphill moves is
    about χ = 0.8. With a single temperature that matches the mean uphill
    gap, T_0 = mean(ΔJ)_uphill / ln(1/χ).
    """
    base = score(start, cell_line).total
    uphill: list[float] = []
    for _ in range(40):
        proposal = propose(start, rng)
        delta = score(proposal, cell_line).total - base
        if delta > 0.0 and math.isfinite(delta):
            uphill.append(delta)
    if not uphill:
        return 1.0
    mean_uphill = float(np.mean(uphill))
    return mean_uphill / math.log(1.0 / INITIAL_UPHILL_ACCEPTANCE)


@dataclass(frozen=True)
class AnnealingResult:
    cell_line: str
    parameters: Parameters
    score: Score
    initial_temperature: float
    final_temperature: float
    evaluations: int
    accepted_fraction: float


def anneal(cell_line: CellLine, rng: np.random.Generator,
           n_temperatures: int = N_TEMPERATURES,
           steps_per_temperature: int = STEPS_PER_TEMPERATURE) -> AnnealingResult:
    """Minimize J(θ) by simulated annealing.

    At temperature level k = 0, 1, ..., K−1:

        propose θ' from the random walk
        ΔJ = J(θ') − J(θ)
        accept θ' with probability A(ΔJ, T_k)
        after `steps_per_temperature` attempts, cool by

            T_{k+1} = α T_k,    0 < α < 1.

    The returned point is the lowest energy visited, not the last point.
    """
    current = sample_parameters(rng)
    current_score = score(current, cell_line)
    temperature = _initial_temperature(current, cell_line, rng)

    best = current
    best_score = current_score
    accepted = 0
    evaluations = 1 + 40  # initial point, plus the temperature pilot

    initial_temperature = temperature

    for _level in range(n_temperatures):
        for _step in range(steps_per_temperature):
            candidate = propose(current, rng)
            candidate_score = score(candidate, cell_line)
            evaluations += 1
            delta = candidate_score.total - current_score.total
            if accept(delta, temperature, float(rng.random())):
                current = candidate
                current_score = candidate_score
                accepted += 1
                if current_score.total < best_score.total:
                    best = current
                    best_score = current_score
        temperature *= COOLING_RATE

    attempts = n_temperatures * steps_per_temperature
    return AnnealingResult(
        cell_line=cell_line.name,
        parameters=best,
        score=best_score,
        initial_temperature=initial_temperature,
        final_temperature=temperature,
        evaluations=evaluations,
        accepted_fraction=accepted / attempts,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fold(mean_squared_log: float) -> float:
    """Typical multiplicative error, exp(sqrt(J)). 1 means an exact match."""
    return math.exp(math.sqrt(mean_squared_log))


def report(cell_line: CellLine, result: AnnealingResult) -> None:
    theta = result.parameters
    fit = result.score
    product = fit.kappa * theta.k_TX

    print(f"\n{cell_line.name} Lipofectamine, harvest at {HARVEST_H:.0f} h")
    print(f"  doubling time {cell_line.doubling_time_h:.1f} h "
          f"({cell_line.doubling_time_source})")
    print(f"  k_div = ln 2 / T_doubling = {cell_line.k_div:.4f} h^-1   (Eq. 5)")
    print(f"  evaluations {result.evaluations}, "
          f"acceptance {result.accepted_fraction:.2f}, "
          f"T_0 {result.initial_temperature:.3g} -> "
          f"T_final {result.final_temperature:.3g}")
    print(f"  J = {fit.total:.4f}   "
          f"(nuclear {_fold(fit.nuclear):.2f}-fold, "
          f"luciferase {_fold(fit.luciferase):.2f}-fold)")
    print(f"  p_entry = {theta.p_entry:.4f}")
    print(f"  K_sat   = {theta.K_sat:.1f} plasmids/nucleus")
    # J does not depend on k_TX (see score). Printing a single k_TX as
    # "the" estimate would overstate what the 24 h data can support.
    print(f"  identifiable product kappa * k_TX = {product:.4g}")
    print(f"  one point on that ridge: k_TX = {theta.k_TX:.4g} h^-1, "
          f"kappa = {fit.kappa:.4g} RLU/mg per mature luciferase")
    print("  Table 1 cross-check for p_entry (Schwake et al., LFN): ~0.37")
    print("  dose/cell     N_obs     N_model     RLU_obs        RLU_model")
    for dose, n_obs, n_hat, rlu_obs, mature in zip(
        cell_line.dose_plasmids_per_cell,
        cell_line.plasmids_per_nucleus,
        fit.nuclear_prediction,
        cell_line.luciferase_rlu_per_mg,
        fit.mature_luciferase,
    ):
        print(f"  {dose:10.3g}  {n_obs:8.1f}  {n_hat:10.1f}  "
              f"{rlu_obs:12.3g}  {fit.kappa * mature:14.3g}")


def main() -> None:
    rng = np.random.default_rng(2026)
    for cell_line in (B16F10, A549):
        # Separate fits: each line has its own doubling time and dose grid.
        # p_entry, k_TX, and K_sat are not forced to be shared across lines.
        result = anneal(cell_line, rng)
        report(cell_line, result)


if __name__ == "__main__":
    main()

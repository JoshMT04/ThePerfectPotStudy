"""
koopman_gravity.py

Modular Koopman-operator aesthetic gravity analysis for vase trajectory data.
Designed so multiple AI/game datasets can be processed with the same functions.

Typical notebook usage
----------------------
from koopman_gravity import (
    load_umap_data, add_parent_vase_mapping, build_trajectory_df,
    build_trajectory_edges, build_transition_matrices,
    get_umap_bounds, build_rbf_grid, build_eval_grid, lift_to_observables,
    build_koopman_operator, extract_dominant_eigenvector,
    choose_gravity_orientation, compute_gravity_scores,
    evaluate_gravity_on_grid,
    compute_selection_entropy,
    plot_gravity_field, plot_selection_entropy,
    plot_entropy_vs_gravity, plot_gravity_scatter, plot_vase_grid,
    attach_gravity_to_df, merge_gravity_into_xy,
    score_real_vases, find_most_beautiful_vase, plot_vase_silhouette,
)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr, shapiro
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  1. Data loading & trajectory construction
# ─────────────────────────────────────────────────────────────────────────────

def load_umap_data(path: str, source: str = "game") -> pd.DataFrame:
    """
    Load a UMAP projection parquet and attach a normalised 'generation' column.

    Parameters
    ----------
    path   : path to parquet file
    source : 'game' — extracts generation from 'vase_id'
             'ai'   — extracts from 'generation_key', keeps only phase == 'main'
    """
    df = pd.read_parquet(path)

    # PC-based projection files carry coordinates as umap_x_pc/umap_y_pc rather than
    # umap_x/umap_y; normalise so the rest of the pipeline doesn't need to know which
    # projection it's working with.
    if "umap_x" not in df.columns and "umap_x_pc" in df.columns:
        df = df.rename(columns={"umap_x_pc": "umap_x", "umap_y_pc": "umap_y"})

    # Combined/full files hold all sources (real, game, ai_game) in one table; filter
    # down to the requested source before extracting generation, otherwise rows from
    # other sources produce NaN and break the int cast downstream.
    source_col_value = {"game": "game", "ai": "ai_game"}.get(source)
    if source_col_value is not None and "source" in df.columns:
        df = df[df["source"] == source_col_value].copy()

    if source == "game":
        df["generation"] = df["vase_id"].str.extract(r"(gen_\d+)").iloc[:, 0]
    elif source == "ai":
        df["generation"] = df["generation_key"].str.extract(r"(gen_\d+)").iloc[:, 0]
        df = df[df["phase"] == "main"].copy()
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'game' or 'ai'.")
    return df


SESSION_KEYS = ["participant_id", "play_counter"]


def add_parent_vase_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a 'parent_vase_id' column: the reigning champion each vase was judged against.

    Game design
    -----------
    One session = one (participant_id, play_counter) round, five generations long.
    Generation 1 presents two vases and the participant picks the more beautiful.
    Every later generation pits the reigning champion against one freshly
    generated challenger, and the participant again picks the more beautiful.

    Storage scheme
    --------------
    Each vase is written to file exactly once, in the generation it first appears.
    A champion that survives further generations is *not* re-recorded, so
    generations 2+ contain only the challenger; the champion stays noted in the
    generation it last won. The lineage therefore has to be carried *forward* from
    that generation rather than read off generation-1, which is what the backward
    merge_asof below does.

    Result
    ------
    Generation-1 rows get parent_vase_id = NaN (no prior champion; the two starting
    vases are siblings, not a lineage step). A challenger at generation g gets the
    most recent winner from any generation strictly before g — the champion it
    actually faced, however many generations ago that champion won.
    """
    df = df.copy()
    df["generation_num"] = df["generation"].str.extract(r"(\d+)").astype(int)

    is_winner = df["is_selected_vase"].eq(True)
    winners = df.loc[is_winner, SESSION_KEYS + ["generation_num", "vase_id"]].copy()

    # Exactly one vase can win a given generation. More than one means distinct
    # sessions have been collapsed together — e.g. if 'generation' lost the round
    # index (ids look like 'round_0_gen_1') and play_counter no longer separates
    # rounds. Fail loudly rather than build a scrambled lineage.
    clashes = winners.duplicated(SESSION_KEYS + ["generation_num"]).sum()
    if clashes:
        raise ValueError(
            f"{clashes} generations have more than one winning vase — session keys "
            f"{SESSION_KEYS} + generation_num do not uniquely identify a matchup."
        )

    winners["parent_generation_num"] = winners["generation_num"]
    winners = winners.rename(columns={"vase_id": "parent_vase_id"})

    # Backward asof on generation_num: for each row, the latest champion crowned
    # strictly earlier in the same session. allow_exact_matches=False keeps a
    # generation's own winner from becoming its own parent.
    return pd.merge_asof(
        df.sort_values("generation_num", kind="mergesort"),
        winners.sort_values("generation_num", kind="mergesort"),
        on="generation_num",
        by=SESSION_KEYS,
        direction="backward",
        allow_exact_matches=False,
    )


def build_trajectory_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, minimal trajectory dataframe from the full vase dataframe."""
    cols = [
        "participant_id", "play_counter", "generation_num", "generation",
        "vase_id", "is_selected_vase", "parent_vase_id", "umap_x", "umap_y",
    ]
    return (
        df[cols]
        .copy()
        .sort_values(["participant_id", "play_counter", "generation_num", "vase_id"])
        .reset_index(drop=True)
    )


def build_trajectory_edges(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract champion→champion edge pairs (one row per lineage link) from a trajectory df.

    Only *winning* challengers become edges. A challenger the participant rejected
    is, by their own judgement, a step away from beauty; including it would feed the
    Koopman fit transitions pointing the wrong way and cancel out the preference
    signal the trajectory is supposed to carry. So an edge exists exactly where a
    challenger unseated the reigning champion, and the resulting chain is the path
    the participant walked towards beauty.
    """
    is_winner = trajectory_df["is_selected_vase"].eq(True)
    return (
        trajectory_df
        .loc[
            trajectory_df["parent_vase_id"].notna() & is_winner,
            ["participant_id", "play_counter", "parent_vase_id", "vase_id", "generation_num"],
        ]
        .copy()
        .rename(columns={"vase_id": "child_vase_id"})
        .reset_index(drop=True)
    )


def build_transition_matrices(
    trajectory_edges: pd.DataFrame,
    trajectory_df: pd.DataFrame,
):
    """
    Attach UMAP coordinates to trajectory edges and build the Z / Z_prime matrices
    required by build_koopman_operator.

    Returns
    -------
    Z        : np.ndarray, shape (2, N) — parent UMAP positions
    Z_prime  : np.ndarray, shape (2, N) — child  UMAP positions
    pairs_df : annotated edge dataframe with umap columns attached
    """
    vase_umap = (
        trajectory_df[["vase_id", "umap_x", "umap_y"]]
        .drop_duplicates("vase_id")
        .set_index("vase_id")
    )
    pairs = trajectory_edges.copy()
    pairs["parent_umap_x"] = pairs["parent_vase_id"].map(vase_umap["umap_x"])
    pairs["parent_umap_y"] = pairs["parent_vase_id"].map(vase_umap["umap_y"])
    pairs["child_umap_x"]  = pairs["child_vase_id"].map(vase_umap["umap_x"])
    pairs["child_umap_y"]  = pairs["child_vase_id"].map(vase_umap["umap_y"])
    pairs = pairs.dropna(
        subset=["parent_umap_x", "parent_umap_y", "child_umap_x", "child_umap_y"]
    ).reset_index(drop=True)

    if len(pairs) == 0:
        raise ValueError("Zero valid transition pairs — check parent/child lineage logic.")

    Z       = pairs[["parent_umap_x", "parent_umap_y"]].to_numpy().T
    Z_prime = pairs[["child_umap_x",  "child_umap_y" ]].to_numpy().T
    return Z, Z_prime, pairs

def data_loading_pipeline(umap_path: str, source: str = "game"):
    """
    Full data loading and trajectory construction pipeline.

    Parameters
    ----------
    umap_path : path to UMAP parquet file
    source    : 'game' or 'ai' — determines how generation is extracted

    Returns
    -------
    trajectory_df   : long-format vase dataframe with parent mapping and UMAP coords
    trajectory_edges: parent-child edge dataframe with UMAP coords attached
    Z, Z_prime      : (2, N) arrays of parent and child UMAP positions for Koopman analysis
    """
    df = load_umap_data(umap_path, source)
    df = add_parent_vase_mapping(df)
    trajectory_df = build_trajectory_df(df)
    trajectory_edges = build_trajectory_edges(trajectory_df)
    Z, Z_prime, pairs_df = build_transition_matrices(trajectory_edges, trajectory_df)
    return trajectory_df, trajectory_edges, pairs_df, Z, Z_prime


# ─────────────────────────────────────────────────────────────────────────────
#  2. Shared UMAP bounds
# ─────────────────────────────────────────────────────────────────────────────

def get_umap_bounds(combined_path: str, pad: float = 0.02) -> dict:
    """
    Compute shared axis bounds from the combined UMAP projection parquet.

    Returns a dict with keys:
        x_min, x_max, y_min, y_max  — raw extents
        xlim, ylim                  — padded (lo, hi) tuples ready for ax.set_xlim
    """
    tmp = pd.read_parquet(combined_path, columns=["umap_x", "umap_y"])
    x_min, x_max = tmp["umap_x"].min(), tmp["umap_x"].max()
    y_min, y_max = tmp["umap_y"].min(), tmp["umap_y"].max()
    del tmp
    pad_x = (x_max - x_min) * pad
    pad_y = (y_max - y_min) * pad
    return dict(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        xlim=(x_min - pad_x, x_max + pad_x),
        ylim=(y_min - pad_y, y_max + pad_y),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  3. RBF lifting & Koopman operator
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_grid(umap_bounds: dict, n_eval: int = 30):
    """
    Uniform n_eval × n_eval lattice for *evaluating* and plotting the gravity field.

    This is deliberately separate from the RBF basis built by build_rbf_grid. The
    two answer different questions: how many basis functions the fit can support
    (few — see build_rbf_grid) versus how finely the resulting field is rendered
    (as many as you like, since evaluation is just a matrix product and involves no
    inverse). Driving both from one n_grid, as this module used to, forces a choice
    between a well-conditioned fit and a smooth picture.

    Returns (grid_x, grid_y), both shape (n_eval, n_eval).
    """
    return np.meshgrid(
        np.linspace(umap_bounds["x_min"], umap_bounds["x_max"], n_eval),
        np.linspace(umap_bounds["y_min"], umap_bounds["y_max"], n_eval),
    )


def build_rbf_grid(umap_bounds: dict, n_grid: int = 15):
    """
    Create a uniform n_grid × n_grid lattice of RBF centres and compute sigma
    as the median nearest-neighbour distance between centres.

    Choosing n_grid
    ---------------
    This sets the size of the Koopman basis (n_grid² functions), and more is not
    better. With sigma tied to the spacing between centres, packing them closer
    makes neighbouring Gaussians increasingly indistinguishable, and the basis
    approaches linear dependence — the flat-limit failure mode of Gaussian RBFs.
    Measured on the game trajectories:

        n_grid=15 →  225 RBFs → cond(Psi) 3e13
        n_grid=20 →  400 RBFs → cond(Psi) 2e23
        n_grid=30 →  900 RBFs → cond(Psi) 2e49

    At n_grid=30 the fit is numerically meaningless without heavy regularisation:
    the operator picks up eigenvalues of order 1e8 that belong to basis directions
    the data never excites. Resolution of the *output* is set by build_eval_grid
    instead, so there is no reason to inflate n_grid for a prettier field.

    Returns (centres, grid_x, grid_y, sigma).
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(umap_bounds["x_min"], umap_bounds["x_max"], n_grid),
        np.linspace(umap_bounds["y_min"], umap_bounds["y_max"], n_grid),
    )
    centres = np.vstack([grid_x.ravel(), grid_y.ravel()]).T   # (n_grid², 2)
    nn = NearestNeighbors(n_neighbors=2).fit(centres)
    dists, _ = nn.kneighbors(centres)
    sigma = np.median(dists[:, 1])
    return centres, grid_x, grid_y, sigma


def lift_to_observables(
    z_data: np.ndarray,
    centres: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian RBF lifting: map 2D UMAP positions into a high-dimensional feature space.

    Parameters
    ----------
    z_data  : (2, N)       — input points (rows = x, y)
    centres : (n_rbf, 2)   — RBF centre locations
    sigma   : float        — RBF bandwidth

    Returns
    -------
    (n_rbf, N) feature matrix
    """
    points  = z_data.T                                              # (N, 2)
    dist_sq = np.sum((points[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    return np.exp(-dist_sq / (2 * sigma ** 2)).T                    # (n_rbf, N)


def build_koopman_operator(
    Z: np.ndarray,
    Z_prime: np.ndarray,
    centres: np.ndarray,
    sigma: float,
    chunk_size: int = 10_000,
    ridge: float = 1e-4,
    verbose: bool = True,
    progress: bool = True,
) -> np.ndarray:
    """
    Estimate the Koopman operator K such that Ψ(Z') ≈ K Ψ(Z) by solving
    the ridge-regularised least-squares problem via Gram matrices.

    Why the ridge is not optional
    ----------------------------
    A = ΣΨΨᵀ is rank-deficient in practice: RBF centres sitting over empty regions
    of UMAP space are never excited by the data, and adjacent centres overlap so
    heavily they are near-duplicates. Its smallest eigenvalues reach 1e-21, so a
    plain solve divides by numerical dust and amplifies it, yielding eigenvalues of
    order 1e8 for an operator whose spectrum belongs inside the unit disk. The
    dominant-mode search then picks out of that noise, and the resulting field is
    not reproducible between runs.

    Adding ridge·mean(diag(A)) to the diagonal lifts every eigenvalue by a constant.
    Well-determined directions (eigenvalues 1e1–1e3) are barely touched; starved
    directions are shrunk towards zero instead of exploding. The field is stable
    across ridge ∈ [1e-5, 1e-1] (pairwise Pearson r ≥ 0.78, mostly > 0.95), so the
    exact value is not load-bearing. Truncation alternatives — pinv, or lstsq on Ψ
    directly — were tested and are worse here: cond(Ψ) ≈ 1e49 means any rank
    threshold lands arbitrarily, and the field swings with it.

    Parameters
    ----------
    Z, Z_prime  : (2, N) arrays of parent and child UMAP positions
    centres     : RBF centres from build_rbf_grid
    sigma       : RBF bandwidth from build_rbf_grid
    chunk_size  : rows processed per batch (tune to available RAM)
    ridge       : regularisation strength, as a fraction of the mean diagonal of A.
                  Relative rather than absolute so it means the same thing across
                  UMAP embeddings of different scale (see umap_seed_sensitivity).
                  0 reproduces the old unregularised solve.
    verbose     : print the resulting spectral radius as a sanity check
    progress    : show the chunk progress bar (turn off when fitting many operators
                  in a loop, e.g. one per PC plane)

    Returns
    -------
    K : (n_rbf, n_rbf) Koopman operator matrix
    """
    n_rbf   = len(centres)
    n_trans = Z.shape[1]
    A = np.zeros((n_rbf, n_rbf))
    B = np.zeros((n_rbf, n_rbf))

    for start in tqdm(range(0, n_trans, chunk_size), desc="Building Koopman Gram matrices",
                      disable=not progress):
        sl    = slice(start, start + chunk_size)
        psi   = lift_to_observables(Z[:, sl],       centres, sigma)
        psi_p = lift_to_observables(Z_prime[:, sl], centres, sigma)
        A += psi   @ psi.T
        B += psi_p @ psi.T

    if ridge:
        A = A + ridge * (np.trace(A) / n_rbf) * np.eye(n_rbf)

    # K A = B  →  K = B A⁻¹
    K = np.linalg.solve(A.T, B.T).T

    if verbose:
        radius = np.abs(np.linalg.eigvals(K)).max()
        flag = "" if radius < 1.05 else "  ← non-physical, raise ridge or lower n_grid"
        print(f"Koopman operator: {n_rbf} basis functions, spectral radius {radius:.4f}{flag}")

    return K


def extract_koopman_mode(K: np.ndarray, mode: int = 1, verbose: bool = True):
    """
    Eigendecompose K and return one mode, indexed by decreasing |eigenvalue|.

    Which mode to use
    -----------------
    Recall that an eigenfunction obeys E[φ(next) | now] = λ·φ(now), so λ says what
    the mode does over a generation.

    mode=0 → λ ≈ 1. The invariant mode: its expectation does not change as the
        lineage moves. By Perron–Frobenius it is non-negative, and it amounts to the
        stationary occupancy density — where vases pile up. A conserved, single-signed
        quantity cannot express a direction of travel, so it is a poor "gravity".

    mode=1 → λ ≈ 0.93 here. The slowest *non-trivial* mode, i.e. the slow coordinate
        of the dynamics (the same object as the second eigenvector in spectral
        clustering / diffusion maps). It straddles zero: its sign splits shape space
        into two metastable basins, its zero contour is the bottleneck between them,
        and its magnitude is depth within a basin. Decay rate 1/(1-λ) gives the
        relaxation timescale in generations. This is the mode that carries the
        attractor structure "aesthetic gravity" is meant to describe, so it is
        the default.

    Note that a decaying mode drifts towards zero from both sides, since
    E[Δφ] = (λ-1)·φ. Any orientation rule based on the mean step direction is
    therefore invalid for mode >= 1 — use choose_gravity_orientation, which scores
    head-to-head matchups instead.

    Returns (eigenvalues, eigenvectors, index, eigenvector) with eigenvalues and
    eigenvectors in the original (unsorted) order, matching the older API.
    """
    evals, evecs = np.linalg.eig(K)
    order = np.argsort(-np.abs(evals))
    if mode >= len(order):
        raise ValueError(f"mode={mode} but K has only {len(order)} eigenvalues.")
    idx = int(order[mode])

    lam = evals[idx]
    if abs(lam.imag) > 1e-8 * max(abs(lam), 1e-12):
        print(
            f"Warning: mode {mode} eigenvalue {lam:.5f} is complex; taking the real "
            "part of its eigenvector, which discards the oscillatory component."
        )
    if verbose:
        tau = np.inf if abs(1 - abs(lam)) < 1e-12 else 1.0 / (1.0 - abs(lam))
        label = "invariant / occupancy density" if mode == 0 else "slow coordinate"
        print(
            f"Koopman mode {mode}: λ = {lam.real:.5f} ({label}), "
            f"relaxation ≈ {tau:.1f} generations"
        )

    return evals, evecs, idx, np.real(evecs[:, idx])


def extract_dominant_eigenvector(K: np.ndarray):
    """
    Backwards-compatible wrapper: the eigenvector whose eigenvalue is closest to 1,
    i.e. the invariant mode. Prefer extract_koopman_mode, which documents why that
    mode is usually the wrong one to build a gravity field from.

    Returns (eigenvalues, eigenvectors, dominant_index, dominant_eigenvector).
    """
    evals, evecs = np.linalg.eig(K)
    idx = np.argmin(np.abs(evals - 1.0))
    return evals, evecs, idx, np.real(evecs[:, idx])


def build_matchups(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct every head-to-head comparison a participant actually made.

    This is the unit of evidence the experiment produces: two vases side by side, one
    judgement. Scoring against it is strictly better than correlating Φ with a binary
    selected flag across the whole dataset, because it compares each chosen vase to
    the alternative that was genuinely on offer rather than to unrelated vases from
    other people's sessions.

    Both arms are recoverable thanks to the champion carry-forward in
    add_parent_vase_mapping:
      * generation 1 stores both starting vases, so the pair is explicit;
      * generations 2+ store only the challenger, and parent_vase_id supplies the
        incumbent it was shown against.

    Returns one row per matchup with columns
    ['participant_id', 'play_counter', 'generation_num', 'winner_vase_id', 'loser_vase_id'].
    """
    keys = SESSION_KEYS + ["generation_num"]
    is_winner = trajectory_df["is_selected_vase"].eq(True)

    # Generations 2+: challenger vs the champion it faced.
    later = trajectory_df.loc[trajectory_df["parent_vase_id"].notna()].copy()
    challenger_won = later["is_selected_vase"].eq(True)
    later["winner_vase_id"] = np.where(challenger_won, later["vase_id"], later["parent_vase_id"])
    later["loser_vase_id"]  = np.where(challenger_won, later["parent_vase_id"], later["vase_id"])

    # Generation 1: the selected starting vase against each rejected one.
    first = trajectory_df.loc[trajectory_df["parent_vase_id"].isna()]
    opening = (
        first.loc[first["is_selected_vase"].eq(True), keys + ["vase_id"]]
        .rename(columns={"vase_id": "winner_vase_id"})
        .merge(
            first.loc[~first["is_selected_vase"].eq(True), keys + ["vase_id"]]
            .rename(columns={"vase_id": "loser_vase_id"}),
            on=keys, how="inner",
        )
    )

    cols = keys + ["winner_vase_id", "loser_vase_id"]
    return (
        pd.concat([later[cols], opening[cols]], ignore_index=True)
        .dropna(subset=["winner_vase_id", "loser_vase_id"])
        .reset_index(drop=True)
    )


def score_vases(df: pd.DataFrame, eigenvector: np.ndarray, centres: np.ndarray,
                sigma: float, chunk_size: int = 10_000) -> pd.Series:
    """Evaluate the eigenfunction at every vase in df, returned as a vase_id-indexed Series."""
    xy = np.vstack([df["umap_x"].to_numpy(), df["umap_y"].to_numpy()])
    phi = compute_gravity_scores(xy, eigenvector, centres, sigma, chunk_size)
    return (
        pd.Series(phi, index=df["vase_id"].to_numpy())
        .groupby(level=0).first()
    )


def matchup_win_rate(matchups: pd.DataFrame, phi_by_vase: pd.Series):
    """
    Fraction of head-to-head matchups in which the winner has the higher Φ.

    0.5 is chance. Returns (win_rate, n_scored).
    """
    d = (
        phi_by_vase.reindex(matchups["winner_vase_id"]).to_numpy()
        - phi_by_vase.reindex(matchups["loser_vase_id"]).to_numpy()
    )
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, 0
    return float((d > 0).mean()), int(len(d))


def choose_gravity_orientation(
    df: pd.DataFrame,
    dominant_eigenvector: np.ndarray,
    centres: np.ndarray,
    sigma: float,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """
    The eigenvector is defined only up to sign. Orient it so that the higher-Φ vase is
    the one people actually chose, judged over every head-to-head matchup.

    Why this anchor
    ---------------
    Two earlier rules were discarded:

    * Correlating Φ with the binary is_selected_vase flag could not fail — g_neg is
      exactly -g_pos, so r_neg = -r_pos and one of the two was always >= 0 and got
      returned whether or not any relationship existed. It also compared vases across
      unrelated sessions.
    * Requiring mean ΔΦ per winning step to be positive works only for the invariant
      mode. A decaying mode satisfies E[Δφ] = (λ-1)·φ, which points towards zero from
      both sides, so the mean step direction reflects where the data started rather
      than which pole is beautiful.

    The matchup win rate has neither problem: it is a paired comparison within the
    choice the participant faced, it is bounded and interpretable against a 0.5 chance
    baseline, and it is valid for any mode regardless of decay.

    Parameters
    ----------
    df : trajectory dataframe with 'vase_id', 'umap_x', 'umap_y', 'parent_vase_id',
         'is_selected_vase' and 'generation_num'

    Returns
    -------
    oriented_eigenvector : same shape as input, possibly sign-flipped
    """
    phi_by_vase = score_vases(df, dominant_eigenvector, centres, sigma, chunk_size)
    matchups = build_matchups(df)
    win_rate, n = matchup_win_rate(matchups, phi_by_vase)

    if not n:
        print("Orientation check — no matchups available; keeping original orientation.")
        return dominant_eigenvector

    print(
        f"Orientation check — winner has higher Φ in {max(win_rate, 1 - win_rate):.1%} "
        f"of {n:,} matchups (chance = 50%)"
    )
    if win_rate < 0.5:
        print("Flipping gravity orientation.")
        return -dominant_eigenvector
    print("Keeping original orientation.")
    return dominant_eigenvector


def compute_gravity_scores(
    umap_xy: np.ndarray,
    dominant_eigenvector: np.ndarray,
    centres: np.ndarray,
    sigma: float,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """
    Evaluate the dominant eigenfunction at each point in umap_xy.

    Parameters
    ----------
    umap_xy : (2, N) array of UMAP coordinates

    Returns
    -------
    scores : (N,) array of aesthetic gravity values
    """
    n = umap_xy.shape[1]
    scores = np.empty(n)
    for start in range(0, n, chunk_size):
        sl  = slice(start, start + chunk_size)
        psi = lift_to_observables(umap_xy[:, sl], centres, sigma)
        scores[sl] = (dominant_eigenvector @ psi).real
    return scores


def evaluate_gravity_on_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    dominant_eigenvector: np.ndarray,
    centres: np.ndarray,
    sigma: float,
):
    """
    Evaluate the gravity potential on the evaluation grid and compute its spatial gradient.

    (U, V) is the gradient of the fitted potential — the direction in which Φ rises and
    how steeply — expressed in Φ per UMAP unit. It is NOT a measured flow: no observed
    displacement enters into it. It agrees with the empirical drift field only to about
    cosine 0.64. For arrows representing movement that participants actually made, build
    the drift field from the transitions instead.

    The grid spacing is passed to np.gradient explicitly. Without it np.gradient
    differentiates per grid index, and since the UMAP bounding box is ~1.8x wider in x
    than in y, the x-component is inflated by that ratio and every arrow is skewed
    towards the horizontal by a median of 11 degrees.

    Returns (phi_2D, U, V) — all shape (n_eval, n_eval).
    """
    dense = np.vstack([grid_x.ravel(), grid_y.ravel()])
    psi   = lift_to_observables(dense, centres, sigma)
    phi   = (dominant_eigenvector @ psi).reshape(grid_x.shape)

    dx = grid_x[0, 1] - grid_x[0, 0] if grid_x.shape[1] > 1 else 1.0
    dy = grid_y[1, 0] - grid_y[0, 0] if grid_y.shape[0] > 1 else 1.0
    V, U  = np.gradient(phi, dy, dx)
    return phi, U, V

def gravity_pipeline(
    trajectory_df,
    trajectory_edges,
    umap_bounds,
    n_basis=15,
    n_eval=30,
    chunk_size=10_000,
    ridge=1e-4,
    mode=1,
):
    """
    Full Koopman gravity pipeline from trajectory data to grid evaluation.

    Parameters
    ----------
    trajectory_df    : output from build_trajectory_df
    trajectory_edges : output from build_trajectory_edges
    umap_bounds      : output from get_umap_bounds
    n_basis          : RBF basis resolution — n_basis² functions are fitted.
                       Keep small; see build_rbf_grid for why more is worse.
    n_eval           : output resolution the field is rendered at. Free to raise,
                       but plot_selection_entropy and plot_entropy_vs_gravity compare
                       phi_2D against an entropy grid cell-by-cell, so it must match
                       compute_selection_entropy's n_grid if you use those.
    chunk_size       : batch size for Koopman operator estimation and gravity scoring
    ridge            : regularisation strength for the operator fit
    mode             : which Koopman mode becomes the gravity field, by decreasing
                       |eigenvalue|. Defaults to 1, the slow coordinate; mode=0 is the
                       invariant occupancy density. See extract_koopman_mode.

    Returns
    -------
    dict with keys:
        phi_2D       : (n_eval, n_eval) gravity potential
        U, V         : (n_eval, n_eval) gradient components for streamplotting
        grid_x,grid_y: (n_eval, n_eval) evaluation coordinates, for plotting
        centres      : (n_basis², 2) RBF centre coordinates
        sigma        : RBF bandwidth
        eigenvector  : the sign-oriented eigenvector. Pass this to
                       compute_gravity_scores or score_real_vases to score
                       individual vases *exactly*, rather than interpolating phi_2D.
        eigenvalue   : its eigenvalue
        win_rate     : in-sample head-to-head matchup accuracy (0.5 = chance).
                       Use validate_gravity_holdout for an out-of-sample figure.
    """
    Z, Z_prime, pairs_df = build_transition_matrices(trajectory_edges, trajectory_df)
    centres, _, _, sigma = build_rbf_grid(umap_bounds, n_basis)
    grid_x, grid_y = build_eval_grid(umap_bounds, n_eval)

    K = build_koopman_operator(Z, Z_prime, centres, sigma, chunk_size, ridge=ridge)
    evals, evecs, idx, evec = extract_koopman_mode(K, mode)
    oriented_evec = choose_gravity_orientation(trajectory_df, evec, centres, sigma, chunk_size)
    phi_2D, U, V = evaluate_gravity_on_grid(grid_x, grid_y, oriented_evec, centres, sigma)

    win_rate, _ = matchup_win_rate(
        build_matchups(trajectory_df),
        score_vases(trajectory_df, oriented_evec, centres, sigma, chunk_size),
    )

    return dict(
        phi_2D=phi_2D, U=U, V=V,
        grid_x=grid_x, grid_y=grid_y,
        centres=centres, sigma=sigma,
        eigenvector=oriented_evec, eigenvalue=evals[idx],
        win_rate=win_rate,
    )


def validate_gravity_holdout(
    trajectory_df,
    umap_bounds,
    n_basis=15,
    chunk_size=10_000,
    ridge=1e-4,
    mode=1,
    n_folds=5,
    seed=0,
):
    """
    Cross-validate the gravity field by participant: can it predict the choices of
    people whose data it never saw?

    The in-sample win rate is optimistic, because Φ is fitted on the very transitions
    it is then scored against. Each fold here refits the operator — and re-derives the
    sign — on a subset of participants, then scores only the held-out participants'
    matchups.

    Folds are split on participant_id, not on matchups. Matchups within a session are
    dependent: the same champion recurs across several of them and the lineage is a
    chain, so splitting matchups at random would leak a person's own choices into the
    data used to predict them.

    Caveat: the UMAP embedding itself was fitted on all vases before this point, so the
    held-out coordinates are not completely independent of the training data. This
    measures generalisation of the Koopman field given the embedding, not of the whole
    pipeline end to end.

    Returns a DataFrame with one row per fold plus a 'mean' row, holding the train and
    test win rates and the number of matchups scored.
    """
    rng = np.random.default_rng(seed)
    participants = np.array(sorted(trajectory_df["participant_id"].unique()))
    folds = np.array_split(rng.permutation(participants), n_folds)

    rows = []
    for k, test_ids in enumerate(folds):
        test_mask = trajectory_df["participant_id"].isin(set(test_ids))
        train_df = trajectory_df.loc[~test_mask]
        test_df  = trajectory_df.loc[test_mask]

        train_edges = build_trajectory_edges(train_df)
        Z, Z_prime, _ = build_transition_matrices(train_edges, train_df)
        centres, _, _, sigma = build_rbf_grid(umap_bounds, n_basis)

        K = build_koopman_operator(Z, Z_prime, centres, sigma, chunk_size,
                                   ridge=ridge, verbose=False)
        _, _, _, evec = extract_koopman_mode(K, mode, verbose=False)
        # Sign is derived from the training participants only.
        evec = choose_gravity_orientation(train_df, evec, centres, sigma, chunk_size)

        train_rate, n_train = matchup_win_rate(
            build_matchups(train_df),
            score_vases(train_df, evec, centres, sigma, chunk_size),
        )
        test_rate, n_test = matchup_win_rate(
            build_matchups(test_df),
            score_vases(test_df, evec, centres, sigma, chunk_size),
        )
        rows.append(dict(fold=k, n_train_matchups=n_train, train_win_rate=train_rate,
                         n_test_matchups=n_test, test_win_rate=test_rate))
        print(f"  fold {k}: train {train_rate:.4f} ({n_train:,})  |  "
              f"test {test_rate:.4f} ({n_test:,})")

    out = pd.DataFrame(rows)
    summary = out.mean(numeric_only=True).to_dict()
    summary["fold"] = "mean"
    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)
    print(
        f"\nHeld-out matchup accuracy: {summary['test_win_rate']:.4f} "
        f"(chance = 0.5) across {n_folds} participant folds"
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  4. Selection entropy
# ─────────────────────────────────────────────────────────────────────────────

def compute_selection_entropy(
    trajectory_pairs: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    umap_bounds: dict,
    n_grid: int = 30,
    n_dir_bins: int = 8,
    min_count: int = 5,
):
    """
    Compute normalised directional entropy of human-selected transitions per grid cell.

    A cell with entropy ≈ 0 means all selections point the same way (laminar / attractor).
    A cell with entropy ≈ 1 means selections are maximally random (turbulent).

    Parameters
    ----------
    trajectory_pairs : output from build_transition_matrices (with umap coords)
    trajectory_df    : from build_trajectory_df (used for is_selected_vase flag)
    umap_bounds      : from get_umap_bounds
    n_grid           : grid resolution (should match the Koopman grid)
    n_dir_bins       : number of compass direction bins (8 = N/NE/E/SE/S/SW/W/NW)
    min_count        : cells with fewer transitions are left as NaN

    Returns
    -------
    entropy_grid : (n_grid, n_grid) array, NaN where data is sparse
    count_grid   : (n_grid, n_grid) int array of transition counts
    x_edges      : (n_grid+1,) bin edges along UMAP x
    y_edges      : (n_grid+1,) bin edges along UMAP y
    """
    sel_info = (
        trajectory_df[["participant_id", "play_counter", "vase_id", "is_selected_vase"]]
        .rename(columns={"vase_id": "child_vase_id"})
    )
    sel = (
        trajectory_pairs
        .merge(sel_info, on=["participant_id", "play_counter", "child_vase_id"], how="left")
        .query("is_selected_vase == True")
        .copy()
    )
    sel["dx"]    = sel["child_umap_x"] - sel["parent_umap_x"]
    sel["dy"]    = sel["child_umap_y"] - sel["parent_umap_y"]
    sel["theta"] = np.arctan2(sel["dy"], sel["dx"])

    x_edges = np.linspace(umap_bounds["x_min"], umap_bounds["x_max"], n_grid + 1)
    y_edges = np.linspace(umap_bounds["y_min"], umap_bounds["y_max"], n_grid + 1)

    sel["grid_ix"] = np.clip(
        np.searchsorted(x_edges, sel["parent_umap_x"].to_numpy(), side="right") - 1,
        0, n_grid - 1,
    )
    sel["grid_iy"] = np.clip(
        np.searchsorted(y_edges, sel["parent_umap_y"].to_numpy(), side="right") - 1,
        0, n_grid - 1,
    )

    bin_edges    = np.linspace(-np.pi, np.pi, n_dir_bins + 1)
    H_max        = np.log(n_dir_bins)
    entropy_grid = np.full((n_grid, n_grid), np.nan)
    count_grid   = np.zeros((n_grid, n_grid), dtype=int)

    for (iy, ix), grp in sel.groupby(["grid_iy", "grid_ix"], observed=True):
        n = len(grp)
        count_grid[iy, ix] = n
        if n < min_count:
            continue
        hist, _ = np.histogram(grp["theta"], bins=bin_edges)
        p    = hist / hist.sum()
        p_nz = p[p > 0]
        entropy_grid[iy, ix] = -np.sum(p_nz * np.log(p_nz)) / H_max

    n_valid = (~np.isnan(entropy_grid)).sum()
    print(f"Cells with ≥{min_count} selected transitions: {n_valid} / {n_grid * n_grid}")
    print(f"Entropy range: {np.nanmin(entropy_grid):.3f} – {np.nanmax(entropy_grid):.3f}")

    return entropy_grid, count_grid, x_edges, y_edges


# ─────────────────────────────────────────────────────────────────────────────
#  5. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def gravity_colour_norm(values: np.ndarray):
    """
    Pick a colour normalisation and map appropriate to the range of Φ.

    Φ is the dominant Koopman eigenfunction, and nothing constrains it to straddle
    zero — with the regularised operator its eigenvalue sits at ~1.0 and the
    eigenfunction comes out non-negative. Zero is then not a meaningful midpoint,
    so a diverging map centred on it would imply a contrast that is not there (and
    TwoSlopeNorm raises outright once vmin >= 0).

    Returns (norm, cmap): diverging RdBu_r about zero when the values genuinely
    span both signs, otherwise a sequential map over the observed range.
    """
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmin < 0 < vmax:
        return plt.matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax), plt.cm.RdBu_r
    return plt.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), plt.cm.magma


def plot_gravity_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    phi_2D: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    umap_xlim: tuple,
    umap_ylim: tuple,
    title: str = "Aesthetic Gravity Field",
    save_path: str = None,
):
    """
    Contour heatmap + streamlines of the aesthetic gravity potential.

    The streamlines trace the gradient of the fitted potential, and their line width is
    |∇Φ| normalised by its own maximum — i.e. how steeply the fitted field rises, in
    arbitrary relative units. They are not measured movement and carry no per-generation
    magnitude, so a thick line means "Φ changes quickly here", not "vases move fast here".
    """
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    norm, cmap = gravity_colour_norm(phi_2D)
    cf = ax.contourf(grid_x, grid_y, phi_2D, levels=50, cmap=cmap, norm=norm, alpha=0.9, antialiased=True)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.set_label(r"$\Phi$ (Aesthetic Gravity)", fontsize=12, labelpad=15)

    speed = np.sqrt(U ** 2 + V ** 2)
    lw    = 3 * speed / (speed.max() + 1e-9)
    ax.streamplot(grid_x, grid_y, U, V, color="#C000C0",
                  linewidth=lw, density=2.5, arrowstyle="-|>", arrowsize=0.75)

    ax.set_xlim(*umap_xlim)
    ax.set_ylim(*umap_ylim)
    ax.set_xlabel("UMAP 1", fontweight="bold", labelpad=10)
    ax.set_ylabel("UMAP 2", fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", color="#EEEEEE", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()


def plot_selection_entropy(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    phi_2D: np.ndarray,
    entropy_grid: np.ndarray,
    count_grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    umap_xlim: tuple,
    umap_ylim: tuple,
    min_count: int = 5,
    save_path: str = None,
):
    """Side-by-side: gravity field (left) and directional selection entropy (right)."""
    if phi_2D.shape != entropy_grid.shape:
        raise ValueError(
            f"phi_2D {phi_2D.shape} and entropy_grid {entropy_grid.shape} must match: "
            "gravity_pipeline's n_eval and compute_selection_entropy's n_grid differ."
        )
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    ax = axes[0]
    norm, cmap = gravity_colour_norm(phi_2D)
    cf = ax.contourf(grid_x, grid_y, phi_2D, levels=50, cmap=cmap, norm=norm, alpha=0.92, antialiased=True)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.set_label(r"$\Phi$ (Aesthetic Gravity)", fontsize=11, labelpad=12)
    ax.set_title("Aesthetic Gravity Field", fontsize=13, pad=10)
    ax.set_xlim(*umap_xlim); ax.set_ylim(*umap_ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP 1", fontweight="bold")
    ax.set_ylabel("UMAP 2", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    im = ax.pcolormesh(x_edges, y_edges, entropy_grid, cmap="inferno", vmin=0, vmax=1, shading="flat")
    cb2 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb2.outline.set_visible(False)
    cb2.set_label("Normalised selection entropy  (0 = laminar, 1 = turbulent)", fontsize=10, labelpad=12)
    ax.contour(x_centres, y_centres, count_grid,
               levels=[min_count, 20, 50], colors="white", alpha=0.35,
               linewidths=[0.6, 0.9, 1.2], linestyles="--")
    ax.set_title("Selection Entropy  (laminar ↔ turbulent)", fontsize=13, pad=10)
    ax.set_xlim(*umap_xlim); ax.set_ylim(*umap_ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP 1", fontweight="bold")
    ax.set_ylabel("UMAP 2", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_entropy_vs_gravity(
    phi_2D: np.ndarray,
    entropy_grid: np.ndarray,
    count_grid: np.ndarray,
    save_path: str = None,
):
    """
    Scatter plot of gravity potential vs directional entropy, with a regression line
    and Pearson/Spearman correlation printed to stdout.
    """
    if phi_2D.shape != entropy_grid.shape:
        raise ValueError(
            f"phi_2D {phi_2D.shape} and entropy_grid {entropy_grid.shape} must match: "
            "gravity_pipeline's n_eval and compute_selection_entropy's n_grid differ."
        )
    valid    = ~np.isnan(entropy_grid.ravel())
    phi_v    = phi_2D.ravel()[valid]
    ent_v    = entropy_grid.ravel()[valid]
    cnt_v    = count_grid.ravel()[valid]

    _, p_phi = shapiro(phi_v)
    _, p_ent = shapiro(ent_v)
    if p_phi < 0.05 or p_ent < 0.05:
        print("Non-normal distribution detected (Shapiro-Wilk p < 0.05) — Spearman preferred.")

    r,   _ = pearsonr(phi_v, ent_v)
    rho, _ = spearmanr(phi_v, ent_v)
    print(f"Gravity φ vs selection entropy  (n={valid.sum()} cells)")
    print(f"  Pearson  r = {r:+.4f}  |  Spearman ρ = {rho:+.4f}")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sc = ax.scatter(phi_v, ent_v, c=cnt_v, cmap="viridis", s=20, alpha=0.75,
                    linewidths=0, norm=plt.matplotlib.colors.LogNorm())
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.set_label("Transition count per cell", fontsize=10)

    m, b = np.polyfit(phi_v, ent_v, 1)
    xr   = np.linspace(phi_v.min(), phi_v.max(), 100)
    ax.plot(xr, m * xr + b, color="#e74c3c", linewidth=1.8, zorder=5,
            label=f"r = {r:+.3f},  ρ = {rho:+.3f}")

    ax.set_xlabel(r"Aesthetic gravity $\Phi$ (cell mean)", fontsize=12)
    ax.set_ylabel("Normalised selection entropy", fontsize=12)
    ax.set_title("Do high-gravity regions have more coherent selection?", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_gravity_scatter(
    df: pd.DataFrame,
    umap_xlim: tuple,
    umap_ylim: tuple,
    title: str = "Vases coloured by aesthetic gravity",
    save_path: str = None,
):
    """Scatter plot of all vases in UMAP space, coloured by aesthetic gravity score."""
    g = df["aesthetic_gravity"]
    norm, cmap = gravity_colour_norm(g.to_numpy())
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        df["umap_x"], df["umap_y"],
        c=g, cmap=cmap, norm=norm, s=5, alpha=0.65, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.set_label(r"$\Phi$ (Aesthetic Gravity)", fontsize=12, labelpad=15)
    ax.set_xlim(*umap_xlim); ax.set_ylim(*umap_ylim)
    ax.set_xlabel("UMAP 1", fontweight="bold", labelpad=10)
    ax.set_ylabel("UMAP 2", fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", color="#EEEEEE", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_vase_grid(
    xy_df: pd.DataFrame,
    vase_meta: pd.DataFrame,
    umap_bounds: dict,
    n_grid: int = 100,
    save_path: str = None,
):
    """
    High-resolution spatial grid: one representative vase silhouette per occupied
    grid cell, filled with the RdBu_r colour mapped to aesthetic gravity.

    Parameters
    ----------
    xy_df      : contour-point dataframe with columns ['vase_id', 'x', 'y']
                 (optionally 'coord_order' for point ordering)
    vase_meta  : one row per vase with 'vase_id', 'umap_x', 'umap_y', 'aesthetic_gravity'
    umap_bounds: from get_umap_bounds
    n_grid     : grid resolution (100 gives a dense map)
    """
    # Build normalised contours (mean-centred, max-abs scaled to [-1, 1])
    contours = {}
    for vase_id, grp in xy_df.groupby("vase_id", sort=False):
        if "coord_order" in grp.columns:
            grp = grp.sort_values("coord_order")
        x = grp["x"].to_numpy(dtype=float)
        y = grp["y"].to_numpy(dtype=float)
        if len(x) < 3:
            continue
        if x[0] != x[-1] or y[0] != y[-1]:
            x, y = np.append(x, x[0]), np.append(y, y[0])
        x -= np.nanmean(x)
        y -= np.nanmean(y)
        scale = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y)), 1e-9)
        contours[str(vase_id)] = np.vstack([x / scale, y / scale])

    pad_x = (umap_bounds["x_max"] - umap_bounds["x_min"]) * 0.02
    pad_y = (umap_bounds["y_max"] - umap_bounds["y_min"]) * 0.02
    x_edges = np.linspace(umap_bounds["x_min"] - pad_x, umap_bounds["x_max"] + pad_x, n_grid + 1)
    y_edges = np.linspace(umap_bounds["y_min"] - pad_y, umap_bounds["y_max"] + pad_y, n_grid + 1)

    xs = vase_meta["umap_x"].to_numpy()
    ys = vase_meta["umap_y"].to_numpy()
    ix = np.clip(np.searchsorted(x_edges, xs) - 1, 0, n_grid - 1)
    iy = np.clip(np.searchsorted(y_edges, ys) - 1, 0, n_grid - 1)

    # Per cell: keep the vase closest to cell centre
    chosen = {}
    for i in range(len(vase_meta)):
        key = (int(ix[i]), int(iy[i]))
        cx  = 0.5 * (x_edges[key[0]] + x_edges[key[0] + 1])
        cy  = 0.5 * (y_edges[key[1]] + y_edges[key[1] + 1])
        d   = (xs[i] - cx) ** 2 + (ys[i] - cy) ** 2
        if key not in chosen or d < chosen[key][1]:
            chosen[key] = (vase_meta.iloc[i]["vase_id"], d)

    gravity_by_vase = vase_meta.set_index("vase_id")["aesthetic_gravity"].to_dict()
    norm, cmap = gravity_colour_norm(vase_meta["aesthetic_gravity"].to_numpy())
    cell_w     = x_edges[1] - x_edges[0]
    cell_h     = y_edges[1] - y_edges[0]
    draw_scale = min(cell_w, cell_h) * 0.45

    fig, ax = plt.subplots(figsize=(14, 14), dpi=600)
    ax.scatter(xs, ys, s=6, color="lightgrey", alpha=0.25, zorder=1)

    for (cx_i, cy_i), (vase_id, _) in chosen.items():
        cx      = 0.5 * (x_edges[cx_i] + x_edges[cx_i + 1])
        cy      = 0.5 * (y_edges[cy_i] + y_edges[cy_i + 1])
        contour = contours.get(str(vase_id))
        if contour is None:
            continue
        gravity = gravity_by_vase.get(vase_id, np.nan)
        colour  = (0.7, 0.7, 0.7, 0.35) if pd.isna(gravity) else cmap(norm(gravity))
        ax.fill(cx + contour[0] * draw_scale, cy + contour[1] * draw_scale,
                facecolor=colour, edgecolor="#00000000", alpha=0.9, linewidth=0.2, zorder=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.set_label(r"$\Phi$ (Aesthetic Gravity)", fontsize=12, labelpad=15)

    ax.set_aspect("equal")
    ax.set_xlim(*umap_bounds["xlim"])
    ax.set_ylim(*umap_bounds["ylim"])
    ax.set_xlabel("UMAP 1", fontweight="bold", labelpad=10)
    ax.set_ylabel("UMAP 2", fontweight="bold", labelpad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", color="#EEEEEE", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
#  6. Multi-mode basin decomposition
# ─────────────────────────────────────────────────────────────────────────────
#
#  A single eigenfunction can only ever describe a bipartition — a positive region,
#  a negative region and the bottleneck between them — which is why a mode=1 gravity
#  field shows exactly two poles no matter how many attractors the data holds. To
#  recover several attractors you need several slow eigenfunctions used jointly as
#  coordinates, then clustered; k eigenfunctions resolve up to k+1 metastable basins.
#  This is the standard spectral/PCCA+ construction, with k-means standing in for the
#  full PCCA+ simplex fit.
#
#  Beware of splitting a near-degenerate group of eigenvalues. Where two eigenvalues
#  are nearly equal, the split between their eigenfunctions is arbitrary — they must be
#  taken together or not at all. extract_koopman_modes warns when the requested set
#  cuts through such a group.


def extract_koopman_modes(K: np.ndarray, modes=(1, 2, 3), degeneracy_tol: float = 0.02,
                          verbose: bool = True):
    """
    Return (eigenvalues, eigenvectors) for several modes, indexed by decreasing |λ|.

    mode 0 is the invariant occupancy density and is excluded by default: being
    conserved and single-signed it carries no directional information, and including
    it in a clustering coordinate frame just adds a density axis.

    Returns (eigenvalues (k,), eigenvectors (n_rbf, k)).
    """
    evals, evecs = np.linalg.eig(K)
    order = np.argsort(-np.abs(evals))
    modes = list(modes)
    if max(modes) >= len(order):
        raise ValueError(f"mode {max(modes)} requested but K has {len(order)} eigenvalues.")

    idx = [int(order[m]) for m in modes]
    lam = evals[idx]

    worst = np.abs(lam.imag / np.maximum(np.abs(lam), 1e-12)).max()
    if worst > 1e-8:
        print(f"Warning: selected modes include a complex eigenvalue (max relative "
              f"imaginary part {worst:.2e}); taking real parts discards oscillation.")

    # Does the selection cut through a near-degenerate group?
    nxt = max(modes) + 1
    if nxt < len(order):
        edge, beyond = abs(evals[int(order[max(modes)])]), abs(evals[int(order[nxt])])
        if edge - beyond < degeneracy_tol:
            print(f"Warning: |λ| of mode {max(modes)} ({edge:.4f}) and mode {nxt} "
                  f"({beyond:.4f}) differ by less than {degeneracy_tol}; these modes are "
                  "near-degenerate and the split between them is arbitrary. Consider "
                  f"including mode {nxt}.")

    if verbose:
        for m, l in zip(modes, lam):
            tau = 1.0 / (1.0 - abs(l)) if abs(1 - abs(l)) > 1e-12 else np.inf
            print(f"  mode {m}: λ = {l.real:.5f}, relaxation ≈ {tau:.1f} generations")

    return lam, np.real(evecs[:, idx])


def embed_vases_in_modes(df: pd.DataFrame, eigenvectors: np.ndarray, centres: np.ndarray,
                         sigma: float, chunk_size: int = 10_000, modes=None) -> pd.DataFrame:
    """
    Evaluate each selected eigenfunction at every vase, giving a coordinate frame to
    cluster in.

    Each coordinate is standardised to unit variance. Eigenvector norms are arbitrary,
    so without this an axis could dominate the clustering purely through its scale
    rather than through any structure it carries.

    Returns a DataFrame indexed by vase_id with one column per mode, named for the
    mode's actual Koopman index (mode_1, mode_2, ...) when `modes` is supplied, plus
    umap_x / umap_y for plotting.
    """
    vases = df.drop_duplicates("vase_id")
    xy = np.vstack([vases["umap_x"].to_numpy(), vases["umap_y"].to_numpy()])

    coords = np.column_stack([
        compute_gravity_scores(xy, eigenvectors[:, k], centres, sigma, chunk_size)
        for k in range(eigenvectors.shape[1])
    ])
    coords = coords / np.maximum(coords.std(axis=0, keepdims=True), 1e-12)

    names = [f"mode_{m}" for m in (modes if modes is not None else range(coords.shape[1]))]
    out = pd.DataFrame(coords, columns=names)
    out["vase_id"] = vases["vase_id"].to_numpy()
    out["umap_x"] = vases["umap_x"].to_numpy()
    out["umap_y"] = vases["umap_y"].to_numpy()
    return out.set_index("vase_id")


def assign_basins(embedding: pd.DataFrame, n_basins: int = 4, seed: int = 0):
    """
    Cluster the eigenfunction coordinates into metastable basins via k-means.

    Basins are relabelled by descending size so the numbering is stable between runs.
    Returns a vase_id-indexed Series of integer basin labels.
    """
    from sklearn.cluster import KMeans

    cols = [c for c in embedding.columns if c.startswith("mode_")]
    labels = KMeans(n_clusters=n_basins, n_init=10, random_state=seed).fit_predict(
        embedding[cols].to_numpy()
    )
    order = pd.Series(labels).value_counts().index
    remap = {old: new for new, old in enumerate(order)}
    return pd.Series([remap[l] for l in labels], index=embedding.index, name="basin")


def basin_transition_matrix(trajectory_edges: pd.DataFrame, basins: pd.Series,
                            normalise: bool = True) -> pd.DataFrame:
    """
    Champion-to-champion flow between basins, as a parent-basin × child-basin matrix.

    A strongly diagonal matrix means the basins really are metastable: lineages that
    enter one tend to stay. Heavy off-diagonal mass means the partition is cutting
    through freely-mixing territory rather than at genuine bottlenecks.

    normalise=True gives row-stochastic transition probabilities.
    """
    src = basins.reindex(trajectory_edges["parent_vase_id"]).to_numpy()
    dst = basins.reindex(trajectory_edges["child_vase_id"]).to_numpy()
    ok = ~(pd.isna(src) | pd.isna(dst))
    mat = pd.crosstab(
        pd.Series(src[ok], name="from_basin").astype(int),
        pd.Series(dst[ok], name="to_basin").astype(int),
    )
    k = int(basins.max()) + 1
    mat = mat.reindex(index=range(k), columns=range(k), fill_value=0)
    if normalise:
        mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
    return mat


def basin_preference(matchups: pd.DataFrame, basins: pd.Series) -> pd.DataFrame:
    """
    Which basins do people actually prefer?

    Restricted to matchups whose two vases fall in different basins, so the result is a
    direct head-to-head record between basins rather than a within-basin tie. Returns a
    DataFrame with, per basin, the number of cross-basin matchups it appeared in and the
    fraction it won. 0.5 is parity.
    """
    w = basins.reindex(matchups["winner_vase_id"]).to_numpy()
    l = basins.reindex(matchups["loser_vase_id"]).to_numpy()
    ok = ~(pd.isna(w) | pd.isna(l))
    w, l = w[ok].astype(int), l[ok].astype(int)
    cross = w != l

    k = int(basins.max()) + 1
    rows = []
    for b in range(k):
        wins = int((w[cross] == b).sum())
        losses = int((l[cross] == b).sum())
        n = wins + losses
        rows.append(dict(basin=b, cross_basin_matchups=n,
                         win_rate=wins / n if n else np.nan))
    return pd.DataFrame(rows)


def decompose_basins(trajectory_df, trajectory_edges, umap_bounds, n_basis=15,
                     modes=(1, 2, 3), n_basins=4, chunk_size=10_000, ridge=1e-4,
                     seed=0):
    """
    Full multi-mode basin decomposition: fit the operator, take several slow
    eigenfunctions, cluster vases in that space and characterise the resulting basins.

    Returns a dict with keys:
        embedding    : vase_id-indexed eigenfunction coordinates (+ umap_x/umap_y)
        basins       : vase_id-indexed basin labels
        summary      : per-basin size, UMAP centroid, representative vase, win rate
        transitions  : row-stochastic basin-to-basin flow matrix
        preference   : per-basin cross-basin head-to-head win rate
        eigenvalues  : the eigenvalues used
        centres, sigma
    """
    Z, Z_prime, _ = build_transition_matrices(trajectory_edges, trajectory_df)
    centres, _, _, sigma = build_rbf_grid(umap_bounds, n_basis)
    K = build_koopman_operator(Z, Z_prime, centres, sigma, chunk_size, ridge=ridge)

    print(f"Using {len(modes)} slow modes to resolve up to {len(modes) + 1} basins:")
    evals, evecs = extract_koopman_modes(K, modes)

    embedding = embed_vases_in_modes(trajectory_df, evecs, centres, sigma, chunk_size,
                                     modes=modes)
    basins = assign_basins(embedding, n_basins, seed)

    matchups = build_matchups(trajectory_df)
    preference = basin_preference(matchups, basins)
    transitions = basin_transition_matrix(trajectory_edges, basins)

    # Representative vase per basin: nearest the basin centroid in eigenfunction space.
    cols = [c for c in embedding.columns if c.startswith("mode_")]
    rows = []
    for b in range(n_basins):
        members = embedding.loc[basins[basins == b].index]
        centroid = members[cols].mean().to_numpy()
        d = np.linalg.norm(members[cols].to_numpy() - centroid, axis=1)
        rep = members.index[int(np.argmin(d))]
        rows.append(dict(
            basin=b, n_vases=len(members),
            umap_x=members["umap_x"].mean(), umap_y=members["umap_y"].mean(),
            representative_vase_id=rep,
        ))
    summary = pd.DataFrame(rows).merge(preference, on="basin")

    print(f"\nbasin occupancy: {summary['n_vases'].tolist()}")
    print(f"basin self-retention (diagonal): "
          f"{[round(float(transitions.iloc[i, i]), 3) for i in range(n_basins)]}")

    return dict(embedding=embedding, basins=basins, summary=summary,
                transitions=transitions, preference=preference,
                eigenvalues=evals, centres=centres, sigma=sigma)


def plot_basin_map(embedding: pd.DataFrame, basins: pd.Series, umap_bounds: dict,
                   summary: pd.DataFrame = None, title: str = "Metastable basins",
                   save_path: str = None):
    """Vases in UMAP space coloured by basin, with basin centroids marked."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    k = int(basins.max()) + 1
    cmap = plt.cm.tab10
    for b in range(k):
        m = basins == b
        lab = f"basin {b}  (n={int(m.sum()):,}"
        if summary is not None and "win_rate" in summary.columns:
            lab += f", wins {summary.loc[summary.basin == b, 'win_rate'].iloc[0]:.1%}"
        ax.scatter(embedding.loc[m, "umap_x"], embedding.loc[m, "umap_y"],
                   s=4, alpha=0.45, linewidths=0, color=cmap(b % 10), label=lab + ")")
    if summary is not None:
        for _, r in summary.iterrows():
            ax.scatter(r["umap_x"], r["umap_y"], marker="X", s=220,
                       color=cmap(int(r["basin"]) % 10), edgecolor="black", linewidth=1.4,
                       zorder=5)
    ax.set_xlim(*umap_bounds["xlim"]); ax.set_ylim(*umap_bounds["ylim"])
    ax.set_xlabel("UMAP 1", fontweight="bold", labelpad=10)
    ax.set_ylabel("UMAP 2", fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", color="#EEEEEE", alpha=0.5)
    ax.legend(frameon=False, fontsize=9, markerscale=3, loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_basin_representatives(xy_df: pd.DataFrame, summary: pd.DataFrame,
                               save_path: str = None):
    """
    One representative vase silhouette per basin, ordered by head-to-head win rate so
    the visual reads left-to-right from least to most preferred.
    """
    order = summary.sort_values("win_rate").reset_index(drop=True)
    n = len(order)
    cmap = plt.cm.tab10
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.6), dpi=200)
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, order.iterrows()):
        grp = xy_df[xy_df["vase_id"] == r["representative_vase_id"]]
        if "coord_order" in grp.columns:
            grp = grp.sort_values("coord_order")
        ax.fill(grp["x"].to_numpy(), grp["y"].to_numpy(),
                facecolor=cmap(int(r["basin"]) % 10), edgecolor="#222222",
                linewidth=0.7, alpha=0.9)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"basin {int(r['basin'])}\nwins {r['win_rate']:.1%}  n={int(r['n_vases']):,}",
                     fontsize=10)
    plt.tight_layout(pad=0.4)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  7. Data export helpers
# ─────────────────────────────────────────────────────────────────────────────

def attach_gravity_to_df(df: pd.DataFrame, gravity_scores: np.ndarray) -> pd.DataFrame:
    """
    Return a copy of df with an 'aesthetic_gravity' column added.
    gravity_scores must align row-for-row with df.
    """
    df = df.copy()
    df["aesthetic_gravity"] = gravity_scores
    return df


def merge_gravity_into_xy(xy_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge vase-level mean gravity + UMAP from df into xy_df on 'vase_id'.
    Removes any pre-existing gravity/umap columns to avoid merge suffixes.
    """
    vase_meta = (
        df[["vase_id", "aesthetic_gravity", "umap_x", "umap_y"]]
        .dropna(subset=["aesthetic_gravity", "umap_x", "umap_y"])
        .groupby("vase_id", as_index=False)
        .agg(
            aesthetic_gravity=("aesthetic_gravity", "mean"),
            umap_x=("umap_x", "mean"),
            umap_y=("umap_y", "mean"),
        )
    )
    drop_cols = [
        c for c in xy_df.columns
        if c in {"aesthetic_gravity", "umap_x", "umap_y"}
        or c.startswith(("aesthetic_gravity_", "umap_x_", "umap_y_"))
    ]
    return (
        xy_df.drop(columns=drop_cols, errors="ignore")
        .merge(vase_meta, on="vase_id", how="left", validate="many_to_one")
    )


# ─────────────────────────────────────────────────────────────────────────────
#  7. Real (museum) vase scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_real_vases(
    real_vases_df: pd.DataFrame,
    dominant_eigenvector: np.ndarray,
    centres: np.ndarray,
    sigma: float,
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """
    Apply the trained gravity eigenfunction to real museum vase UMAP positions.
    Returns a copy of real_vases_df with 'aesthetic_gravity' column added.
    """
    missing = {"umap_x", "umap_y"} - set(real_vases_df.columns)
    if missing:
        raise ValueError(f"real_vases_df missing columns: {sorted(missing)}")

    xy     = np.vstack([real_vases_df["umap_x"].to_numpy(), real_vases_df["umap_y"].to_numpy()])
    scores = compute_gravity_scores(xy, dominant_eigenvector, centres, sigma, chunk_size)
    df     = real_vases_df.copy()
    df["aesthetic_gravity"] = scores
    return df


def find_most_beautiful_vase(
    real_vases_df: pd.DataFrame,
    real_vases_xy_df: pd.DataFrame,
    id_col: str = "vase_id",
) -> tuple:
    """
    Return (vase_row, contour_df) for the real vase with the highest aesthetic gravity.

    Parameters
    ----------
    id_col : the column name used as the vase identifier in real_vases_xy_df
    """
    row        = real_vases_df.loc[real_vases_df["aesthetic_gravity"].idxmax()]
    contour_df = real_vases_xy_df[real_vases_xy_df[id_col] == row["vase_id"]].copy()
    return row, contour_df


def plot_vase_silhouette(
    contour_df: pd.DataFrame,
    title: str = "",
    save_path: str = None,
):
    """Clean filled silhouette of a single vase from its contour coordinates."""
    x, y = contour_df["x"].to_numpy(), contour_df["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.fill(x, y, facecolor="#1f3b70", edgecolor="#0d2448", linewidth=0.6, alpha=0.92)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

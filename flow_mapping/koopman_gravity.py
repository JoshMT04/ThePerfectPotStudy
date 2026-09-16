"""
koopman_gravity.py

Modular Koopman-operator aesthetic gravity analysis for vase trajectory data.
Designed so multiple AI/game datasets can be processed with the same functions.

Typical notebook usage
----------------------
from koopman_gravity import (
    load_umap_data, add_parent_vase_mapping, build_trajectory_df,
    build_trajectory_edges, build_transition_matrices,
    get_umap_bounds, build_rbf_grid, lift_to_observables,
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


def add_parent_vase_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a 'parent_vase_id' column by joining each vase to the selected vase
    from the previous generation within the same (participant, play_counter) session.
    """
    df = df.copy()
    df["generation_num"] = df["generation"].str.extract(r"(\d+)").astype(int)
    selected = (
        df.loc[
            df["is_selected_vase"],
            ["participant_id", "play_counter", "generation_num", "vase_id"],
        ]
        .rename(columns={
            "generation_num": "parent_generation_num",
            "vase_id": "parent_vase_id",
        })
    )
    df["parent_generation_num"] = df["generation_num"] - 1
    df = df.merge(
        selected,
        on=["participant_id", "play_counter", "parent_generation_num"],
        how="left",
    )
    return df


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
    Extract parent→child edge pairs (one row per lineage link) from a trajectory df.
    """
    return (
        trajectory_df
        .loc[
            trajectory_df["parent_vase_id"].notna(),
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

def build_rbf_grid(umap_bounds: dict, n_grid: int = 30):
    """
    Create a uniform n_grid × n_grid lattice of RBF centres and compute sigma
    as the median nearest-neighbour distance between centres.

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
) -> np.ndarray:
    """
    Estimate the Koopman operator K such that Ψ(Z') ≈ K Ψ(Z) by solving
    the least-squares problem via Gram matrices.

    Parameters
    ----------
    Z, Z_prime  : (2, N) arrays of parent and child UMAP positions
    centres     : RBF centres from build_rbf_grid
    sigma       : RBF bandwidth from build_rbf_grid
    chunk_size  : rows processed per batch (tune to available RAM)

    Returns
    -------
    K : (n_rbf, n_rbf) Koopman operator matrix
    """
    n_rbf   = len(centres)
    n_trans = Z.shape[1]
    A = np.zeros((n_rbf, n_rbf))
    B = np.zeros((n_rbf, n_rbf))

    for start in tqdm(range(0, n_trans, chunk_size), desc="Building Koopman Gram matrices"):
        sl    = slice(start, start + chunk_size)
        psi   = lift_to_observables(Z[:, sl],       centres, sigma)
        psi_p = lift_to_observables(Z_prime[:, sl], centres, sigma)
        A += psi   @ psi.T
        B += psi_p @ psi.T

    # K A = B  →  K = B A⁻¹
    return np.linalg.solve(A.T, B.T).T


def extract_dominant_eigenvector(K: np.ndarray):
    """
    Eigendecompose K and extract the eigenvector whose eigenvalue is closest to 1.
    Eigenvalue ≈ 1 corresponds to the slowest-decaying (most persistent) mode.

    Returns (eigenvalues, eigenvectors, dominant_index, dominant_eigenvector).
    """
    evals, evecs = np.linalg.eig(K)
    idx = np.argmin(np.abs(evals - 1.0))
    return evals, evecs, idx, np.real(evecs[:, idx])


def choose_gravity_orientation(
    df: pd.DataFrame,
    dominant_eigenvector: np.ndarray,
    centres: np.ndarray,
    sigma: float,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """
    The dominant eigenvector is defined only up to sign. This function evaluates
    both orientations on all vases and returns whichever sign correlates more
    positively with human selection rates (Pearson correlation).

    Parameters
    ----------
    df                  : full vase dataframe with 'umap_x', 'umap_y', 'is_selected_vase'
    dominant_eigenvector: raw output from extract_dominant_eigenvector

    Returns
    -------
    oriented_eigenvector : same shape as input, possibly sign-flipped
    """
    vase_xy = np.vstack([df["umap_x"].to_numpy(), df["umap_y"].to_numpy()])
    n = vase_xy.shape[1]
    g_pos = np.empty(n)
    g_neg = np.empty(n)

    for start in range(0, n, chunk_size):
        sl  = slice(start, start + chunk_size)
        psi = lift_to_observables(vase_xy[:, sl], centres, sigma)
        g_pos[sl] = ( dominant_eigenvector @ psi).real
        g_neg[sl] = (-dominant_eigenvector @ psi).real

    check = df[["vase_id", "is_selected_vase"]].copy()
    check["selected"] = check["is_selected_vase"].astype(float)
    check["g_pos"] = g_pos
    check["g_neg"] = g_neg

    vase_check = (
        check.groupby("vase_id", as_index=False)
        .agg(
            selected_rate=("selected", "mean"),
            g_pos=("g_pos", "mean"),
            g_neg=("g_neg", "mean"),
        )
        .dropna()
    )

    r_pos = vase_check["g_pos"].corr(vase_check["selected_rate"])
    r_neg = vase_check["g_neg"].corr(vase_check["selected_rate"])
    print(f"Orientation check — Pearson(+): {r_pos:.4f}  |  Pearson(−): {r_neg:.4f}")

    if r_neg > r_pos:
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
    Evaluate the gravity potential on the dense RBF grid and compute its spatial gradient.

    Returns (phi_2D, U, V) — all shape (n_grid, n_grid).
    phi_2D is the potential; (U, V) is the gradient vector field for streamplots.
    """
    dense = np.vstack([grid_x.ravel(), grid_y.ravel()])
    psi   = lift_to_observables(dense, centres, sigma)
    phi   = (dominant_eigenvector @ psi).reshape(grid_x.shape)
    V, U  = np.gradient(phi)
    return phi, U, V

def gravity_pipeline(trajectory_df, trajectory_edges, umap_bounds, n_grid=30, chunk_size=10_000):
    """
    Full Koopman gravity pipeline from trajectory data to grid evaluation.

    Parameters
    ----------
    trajectory_df    : output from build_trajectory_df
    trajectory_edges : output from build_trajectory_edges
    umap_bounds      : output from get_umap_bounds
    n_grid           : grid resolution for RBF centres and gravity evaluation
    chunk_size       : batch size for Koopman operator estimation and gravity scoring

    Returns
    -------
    phi_2D : (n_grid, n_grid) array of gravity potential values on the grid
    U, V   : (n_grid, n_grid) arrays of gradient components for streamplotting
    centres : (n_grid², 2) array of RBF centre coordinates
    sigma   : float, RBF bandwidth
    """
    Z, Z_prime, pairs_df = build_transition_matrices(trajectory_edges, trajectory_df)
    centres, grid_x, grid_y, sigma = build_rbf_grid(umap_bounds, n_grid)
    K = build_koopman_operator(Z, Z_prime, centres, sigma, chunk_size)
    evals, evecs, idx_dom, dom_evec = extract_dominant_eigenvector(K)
    oriented_evec = choose_gravity_orientation(trajectory_df, dom_evec, centres, sigma, chunk_size)
    phi_2D, U, V = evaluate_gravity_on_grid(grid_x, grid_y, oriented_evec, centres, sigma)
    return phi_2D, U, V, centres, sigma


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
    """Contour heatmap + streamlines of the aesthetic gravity potential."""
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    norm = plt.matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=phi_2D.min(), vmax=phi_2D.max())
    cf = ax.contourf(grid_x, grid_y, phi_2D, levels=50, cmap="RdBu_r", norm=norm, alpha=0.9, antialiased=True)
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
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    ax = axes[0]
    norm = plt.matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=phi_2D.min(), vmax=phi_2D.max())
    cf = ax.contourf(grid_x, grid_y, phi_2D, levels=50, cmap="RdBu_r", norm=norm, alpha=0.92, antialiased=True)
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
    norm = plt.matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=g.min(), vmax=g.max())
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        df["umap_x"], df["umap_y"],
        c=g, cmap="RdBu_r", norm=norm, s=5, alpha=0.65, linewidths=0,
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
    gmin, gmax = vase_meta["aesthetic_gravity"].min(), vase_meta["aesthetic_gravity"].max()
    norm       = plt.matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=gmin, vmax=gmax)
    cmap       = plt.cm.RdBu_r
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
#  6. Data export helpers
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

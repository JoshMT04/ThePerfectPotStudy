"""
pc_flow.py

Koopman flow mapping in PC space rather than UMAP space.

Why PC space
------------
The PCs are the coordinates the experiment was actually run in: the generator sampled
PC values to synthesise each vase, and those values were recorded. So a PC plane is the
real parameter space participants explored, it is linear, its axes have fixed meaning,
and it needs no embedding to be fitted. UMAP, by contrast, is a non-linear embedding
estimated from the data, whose axes have no intrinsic meaning and whose geometry shifts
with the random seed.

What the data gives us here
---------------------------
file_processing/processed/pc_data.parquet stores BOTH vases at every generation — the
incumbent champion is re-recorded rather than carried forward, and is_parent_vase marks
it. The parent at generation n is exactly the winner at generation n-1 (verified: 54,480
of 54,480 matches, maximum PC difference 0.0), so the lineage is recorded rather than
inferred, and every matchup has both arms present.

That also makes champion *retentions* visible: of 54,480 champion transitions, 33,888
(62.2%) are the champion holding its place and 20,592 are a challenger taking over. The
UMAP export could not represent the retentions at all. They are included here by default,
because a retained champion is still the selected vase and because a process that stops
moving is precisely what an attractor looks like — mean displacement per generation falls
2.68 -> 1.68 -> 1.05 -> 0.66 across a session.

This module handles PC data loading and pairwise plane analysis, and defers the operator
machinery to koopman_gravity. koopman_gravity keys its coordinates off columns literally
named 'umap_x'/'umap_y'; pair_frame() below populates those columns with the chosen PC
pair, so treat that name as a generic 2D coordinate slot rather than anything UMAP-ish.
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from koopman_gravity import (
    build_rbf_grid, build_eval_grid, build_transition_matrices,
    build_koopman_operator, extract_koopman_mode, evaluate_gravity_on_grid,
    compute_gravity_scores, matchup_win_rate, gravity_colour_norm,
)

PC_COLUMNS = [f"PC{i}" for i in range(1, 7)]


# ─────────────────────────────────────────────────────────────────────────────
#  1. Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pc_wide(path: str = "../file_processing/processed/pc_data.parquet") -> pd.DataFrame:
    """
    Load the long-format PC export and pivot it to one row per vase.

    A small number of vase records (650 of 136,850) carry duplicated PC rows; the
    duplicated values are byte-identical, so they are dropped rather than averaged.

    Returns a frame with one row per vase holding PC1..PC6, vase_id, generation_num,
    is_selected_vase and is_parent_vase.
    """
    df = pd.read_parquet(path)
    wide = (
        df.drop_duplicates(["participant_id", "generation_key", "vase", "pc_name"])
        .pivot_table(
            index=["participant_id", "play_counter", "generation_key", "vase",
                   "is_selected_vase", "is_parent_vase"],
            columns="pc_name", values="pc_value",
        )
        .reset_index()
    )
    wide["generation_num"] = wide["generation_key"].str.extract(r"gen_(\d+)").astype(int)
    wide["vase_id"] = (
        wide["participant_id"] + "::" + wide["generation_key"] + "::" + wide["vase"]
    )
    return wide.sort_values(
        ["participant_id", "play_counter", "generation_num", "vase"]
    ).reset_index(drop=True)


def build_champion_transitions(wide: pd.DataFrame, include_stays: bool = True) -> pd.DataFrame:
    """
    Champion-to-champion edges: the selected vase at generation n to the selected vase
    at generation n+1, within one session.

    include_stays=True keeps the transitions where the incumbent was retained. These are
    genuine dynamics — the state was observed not to move — and dropping them biases the
    operator towards motion and inflates the apparent drift. They are identifiable
    because the retained champion is re-recorded under a new vase_id with identical PC
    values, so the edge has exactly zero displacement.

    Returns edges with the column names build_transition_matrices expects.
    """
    keys = ["participant_id", "play_counter"]
    win = (
        wide.loc[wide["is_selected_vase"], keys + ["generation_num", "vase_id"]]
        .sort_values(keys + ["generation_num"])
    )
    nxt = win.rename(columns={"vase_id": "child_vase_id"}).copy()
    nxt["generation_num"] = nxt["generation_num"] - 1

    edges = (
        win.rename(columns={"vase_id": "parent_vase_id"})
        .merge(nxt, on=keys + ["generation_num"], how="inner")
        .reset_index(drop=True)
    )
    # generation_num currently labels the source; make it label the step's destination
    edges["generation_num"] = edges["generation_num"] + 1

    if not include_stays:
        stayed = wide.set_index("vase_id")["is_parent_vase"]
        edges = edges.loc[
            ~stayed.reindex(edges["child_vase_id"]).to_numpy()
        ].reset_index(drop=True)
    return edges


def build_pc_matchups(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Every head-to-head comparison, taken directly from the file.

    Both arms are recorded at every generation here, so unlike the UMAP export nothing
    has to be reconstructed: within each (session, generation) the selected vase is the
    winner and the other is the loser.
    """
    keys = ["participant_id", "play_counter", "generation_num"]
    win = (wide.loc[wide["is_selected_vase"], keys + ["vase_id"]]
           .rename(columns={"vase_id": "winner_vase_id"}))
    lose = (wide.loc[~wide["is_selected_vase"], keys + ["vase_id"]]
            .rename(columns={"vase_id": "loser_vase_id"}))
    return win.merge(lose, on=keys, how="inner").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  2. One PC plane
# ─────────────────────────────────────────────────────────────────────────────

def pair_frame(wide: pd.DataFrame, pc_x: str, pc_y: str) -> pd.DataFrame:
    """
    Project the vases onto one PC plane, exposing the pair as 'umap_x'/'umap_y' so the
    koopman_gravity functions can consume it unchanged. Nothing UMAP is involved; the
    column names are simply that module's generic coordinate slot.
    """
    out = wide[["vase_id", "participant_id", "play_counter", "generation_num",
                "is_selected_vase", "is_parent_vase"]].copy()
    out["umap_x"] = wide[pc_x].to_numpy()
    out["umap_y"] = wide[pc_y].to_numpy()
    return out


def pc_bounds(wide: pd.DataFrame, pc_x: str, pc_y: str, q: float = 0.005,
              pad: float = 0.02) -> dict:
    """
    Axis bounds for one PC plane, clipped to the central quantile range.

    PC scores have long tails — PC1 spans -21.6 to 27.7 while its standard deviation is
    5.7 — so bounds taken from the extremes would spend most of the RBF grid on a handful
    of outliers and leave the populated centre under-resolved. Clipping at the 0.5th and
    99.5th percentiles by default keeps the grid where the data is.
    """
    x, y = wide[pc_x], wide[pc_y]
    x_min, x_max = float(x.quantile(q)), float(x.quantile(1 - q))
    y_min, y_max = float(y.quantile(q)), float(y.quantile(1 - q))
    pad_x, pad_y = (x_max - x_min) * pad, (y_max - y_min) * pad
    return dict(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        xlim=(x_min - pad_x, x_max + pad_x), ylim=(y_min - pad_y, y_max + pad_y),
    )


def run_pc_pair(wide, edges, matchups, pc_x, pc_y, n_basis=15, n_eval=40,
                ridge=1e-3, mode=1, chunk_size=10_000):
    """
    Fit the Koopman operator on one PC plane and return its gravity field.

    The sign of the eigenfunction is fixed by the matchup win rate — the higher-Φ vase
    should be the one people chose — which is valid whether or not the mode decays.

    ridge defaults to 1e-3 rather than koopman_gravity's 1e-4: PC planes need slightly
    more regularisation before the spectrum becomes physical (on PC1-PC2, 1e-4 leaves a
    spectral radius of 1.0187 against 0.9834 at 1e-3). The spectral_radius returned
    below is the check — anything above ~1.05 means the fit is not trustworthy.

    Note that including retentions pulls K towards the identity — 62% of transitions do
    not move — so the spectrum is compressed towards 1 and the gaps between modes are
    small (0.983, 0.964, 0.958, 0.954 on PC1-PC2). Mode 1 is still the slowest
    non-trivial coordinate, but it is less cleanly separated than in UMAP space. Fitting
    on moves only (include_stays=False) spreads the spectrum out again, at the cost of
    describing movement conditional on a move having happened.

    Returns a dict with phi_2D, U, V, grid_x, grid_y, eigenvalue, win_rate and the
    per-plane bounds.
    """
    frame = pair_frame(wide, pc_x, pc_y)
    bounds = pc_bounds(wide, pc_x, pc_y)

    Z, Z_prime, _ = build_transition_matrices(edges, frame)
    centres, _, _, sigma = build_rbf_grid(bounds, n_basis)
    grid_x, grid_y = build_eval_grid(bounds, n_eval)

    K = build_koopman_operator(Z, Z_prime, centres, sigma, chunk_size,
                               ridge=ridge, verbose=False, progress=False)
    evals, _, idx, evec = extract_koopman_mode(K, mode, verbose=False)

    xy = np.vstack([frame["umap_x"].to_numpy(), frame["umap_y"].to_numpy()])
    phi = pd.Series(compute_gravity_scores(xy, evec, centres, sigma, chunk_size),
                    index=frame["vase_id"].to_numpy()).groupby(level=0).first()
    win_rate, n_matchups = matchup_win_rate(matchups, phi)
    if win_rate < 0.5:
        evec, win_rate = -evec, 1 - win_rate

    phi_2D, U, V = evaluate_gravity_on_grid(grid_x, grid_y, evec, centres, sigma)
    return dict(
        pc_x=pc_x, pc_y=pc_y, bounds=bounds,
        phi_2D=phi_2D, U=U, V=V, grid_x=grid_x, grid_y=grid_y,
        centres=centres, sigma=sigma, eigenvector=evec,
        eigenvalue=evals[idx], win_rate=win_rate, n_matchups=n_matchups,
        spectral_radius=float(np.abs(np.linalg.eigvals(K)).max()),
    )


def drift_on_plane(edges, frame, bounds, n_grid=16, min_count=25):
    """
    Measured mean displacement per cell — where champions actually moved, as opposed to
    the gradient of a fitted potential.

    Arrow length is in PC units per generation, so it carries a real scale. Cells with
    fewer than min_count transitions are left as NaN: sparse cells at the edge of the
    occupied region show spurious inward flow simply because their populated neighbours
    all lie inward.
    """
    coords = frame.drop_duplicates("vase_id").set_index("vase_id")[["umap_x", "umap_y"]]
    p = coords.reindex(edges["parent_vase_id"]).to_numpy()
    c = coords.reindex(edges["child_vase_id"]).to_numpy()
    ok = ~(np.isnan(p).any(1) | np.isnan(c).any(1))
    p, c = p[ok], c[ok]

    x_edges = np.linspace(bounds["x_min"], bounds["x_max"], n_grid + 1)
    y_edges = np.linspace(bounds["y_min"], bounds["y_max"], n_grid + 1)
    ix = np.clip(np.searchsorted(x_edges, p[:, 0]) - 1, 0, n_grid - 1)
    iy = np.clip(np.searchsorted(y_edges, p[:, 1]) - 1, 0, n_grid - 1)

    U = np.zeros((n_grid, n_grid)); V = np.zeros((n_grid, n_grid))
    C = np.zeros((n_grid, n_grid))
    np.add.at(U, (iy, ix), c[:, 0] - p[:, 0])
    np.add.at(V, (iy, ix), c[:, 1] - p[:, 1])
    np.add.at(C, (iy, ix), 1)

    keep = C >= min_count
    U = np.where(keep, U / np.maximum(C, 1), np.nan)
    V = np.where(keep, V / np.maximum(C, 1), np.nan)
    cx = 0.5 * (x_edges[:-1] + x_edges[1:])
    cy = 0.5 * (y_edges[:-1] + y_edges[1:])
    gx, gy = np.meshgrid(cx, cy)
    return dict(U=U, V=V, count=C, grid_x=gx, grid_y=gy)


def drift_split_half(edges, frame, bounds, seed=0, **kwargs):
    """
    Split-half reproducibility of a measured drift field, over disjoint participants.

    Participants rather than transitions, because transitions within a session are
    dependent — they share champions along one chain — so a random split of transitions
    would leak a person's own behaviour across the two halves.

    Returns (r_U, r_V, n_cells_compared).
    """
    rng = np.random.default_rng(seed)
    people = np.array(sorted(edges["participant_id"].unique()))
    rng.shuffle(people)
    half = set(people[: len(people) // 2])

    a = drift_on_plane(edges[edges["participant_id"].isin(half)], frame, bounds, **kwargs)
    b = drift_on_plane(edges[~edges["participant_id"].isin(half)], frame, bounds, **kwargs)
    k = ~np.isnan(a["U"]) & ~np.isnan(b["U"])
    if k.sum() < 5:
        return np.nan, np.nan, int(k.sum())
    return (float(np.corrcoef(a["U"][k], b["U"][k])[0, 1]),
            float(np.corrcoef(a["V"][k], b["V"][k])[0, 1]),
            int(k.sum()))


# ─────────────────────────────────────────────────────────────────────────────
#  3. All planes
# ─────────────────────────────────────────────────────────────────────────────

def all_pc_pairs(pcs=PC_COLUMNS):
    """Every unordered PC pair, in order: (PC1,PC2), (PC1,PC3), ... 15 for six PCs."""
    return list(itertools.combinations(pcs, 2))


def run_all_pc_pairs(wide, edges, matchups, pcs=PC_COLUMNS, verbose=True, **kwargs):
    """Fit every PC plane. Returns {(pc_x, pc_y): result}."""
    results = {}
    for pc_x, pc_y in all_pc_pairs(pcs):
        r = run_pc_pair(wide, edges, matchups, pc_x, pc_y, **kwargs)
        results[(pc_x, pc_y)] = r
        if verbose:
            print(f"  {pc_x} vs {pc_y}:  lambda = {r['eigenvalue'].real:.4f}   "
                  f"matchup win rate = {r['win_rate']:.4f}")
    return results


def _fit_drift_spline(x, d, groups, n_knots="cv", alpha=1e-2,
                      knot_grid=(4, 5, 6, 8, 10, 12, 14)):
    """
    Penalised cubic B-spline of drift against position — a GAM with a single smooth term,
    built from sklearn's SplineTransformer plus a ridge penalty.

    With n_knots="cv" the number of knots is chosen by 5-fold cross-validation grouped on
    participant, so the flexibility is set by how well the curve predicts people it has
    not seen rather than by how closely it traces the sample. Grouping matters: the
    transitions within one session share champions along a chain, so an ungrouped split
    would leak and would licence far more wiggle than the data supports.

    Selection uses the one-standard-error rule — the fewest knots whose mean CV score is
    within one standard error of the best — rather than the outright maximum. Neighbouring
    knot counts here score within noise of each other, so picking the argmax chases that
    noise and yields curves with visible wiggles that carry no support; the 1-SE rule
    returns the simplest curve the data cannot distinguish from the best one.

    Returns (fitted_model, chosen_n_knots).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    def build(k):
        return make_pipeline(
            SplineTransformer(n_knots=k, degree=3, extrapolation="linear"),
            Ridge(alpha=alpha),
        )

    X = x.reshape(-1, 1)
    if n_knots == "cv":
        cv = GroupKFold(n_splits=5)
        folds = [cross_val_score(build(k), X, d, cv=cv, groups=groups, scoring="r2")
                 for k in knot_grid]
        means = np.array([f.mean() for f in folds])
        best = int(np.argmax(means))
        threshold = means[best] - folds[best].std(ddof=1) / np.sqrt(len(folds[best]))
        n_knots = int(knot_grid[int(np.argmax(means >= threshold))])
    return build(n_knots).fit(X, d), int(n_knots)


def _drift_roots(xs, ys, x_all):
    """
    Every zero crossing of a sampled drift curve, classified by the sign of the slope
    there: negative means the dynamics restore towards that value (a stable fixed point,
    i.e. an attractor), positive means they push away (unstable, a tipping point).

    Returns a list of (position, kind, percentile) sorted by position.
    """
    roots = []
    for j in range(len(xs) - 1):
        if ys[j] == 0 or ys[j] * ys[j + 1] < 0:
            step = (xs[j + 1] - xs[j])
            root = xs[j] - ys[j] * step / (ys[j + 1] - ys[j])
            slope = (ys[j + 1] - ys[j]) / step
            roots.append((float(root), "stable" if slope < 0 else "unstable",
                          float((x_all < root).mean() * 100)))
    return roots


def per_pc_drift(edges, wide, pcs=PC_COLUMNS, n_bins=25, q=0.02, n_knots="cv") -> pd.DataFrame:
    """
    One-dimensional drift along each PC: mean displacement along PC_i as a function of
    position on PC_i. This is the marginal description of movement, and it needs one
    panel per PC rather than one per pair.

    A spline, not a line or a parabola
    ----------------------------------
    Selection is not a straight line, and it is not symmetric either. PC1 inverts at the
    bottom and flattens at the top — the drift stops steepening once a champion is far
    out, because there is less design space left to move into — and a parabola cannot
    represent an asymmetric shape like that. The fit is therefore a penalised cubic
    B-spline (a GAM with one smooth term), with the number of knots chosen by
    cross-validation grouped on participant.

    On held-out participants the spline beats both simpler fits for every PC, and the
    chosen flexibility is modest (4-8 knots; CV declines beyond that). Against the binned
    means the improvement is large:

        PC1 R^2 lin 0.74 -> quad 0.90 -> spline 0.97
        PC3         0.85 ->      0.89 ->        0.96
        PC5         0.87 ->      0.90 ->        0.97

    (Against raw single transitions every R^2 is ~0.01, because 62% of transitions are
    zero-displacement retentions and individual steps are dominated by noise. The binned
    comparison is the one that answers "does the curve track the trend".)

    A flexible curve can cross zero any number of times, so it is free to report several
    fixed points: **stable** ones where drift crosses zero going downwards and the
    dynamics pull back — attractors — and **unstable** ones where they push away. That
    freedom is the point of using it. In this data it finds exactly one stable fixed
    point per PC and no unstable ones in range, each within ~0.3 of the quadratic
    estimate, so the single-attractor result holds across all three fit families rather
    than being an artefact of assuming a simple shape.

    Read `fixed_point` together with `fixed_point_percentile`, which says where it sits
    among the champion positions actually visited. A raw min/max test is useless — the
    long PC tails stretch far enough that almost any value falls "inside" — whereas the
    percentile separates a genuine interior optimum from one pinned against the edge of
    the design space the generator could produce.

    Returns one row per (pc, bin): position, mean drift, standard error and count, plus
    the quadratic and linear fits, the stable fixed point with its percentile, any
    in-range unstable root, the local restoring slope at the fixed point, and the binned
    R^2 of both fits — each repeated across that PC's rows.
    """
    coords = wide.drop_duplicates("vase_id").set_index("vase_id")[list(pcs)]
    parent = coords.reindex(edges["parent_vase_id"]).to_numpy()
    child = coords.reindex(edges["child_vase_id"]).to_numpy()
    delta = child - parent

    groups = edges["participant_id"].to_numpy()
    rows, curves, chosen_knots = [], {}, {}
    for i, pc in enumerate(pcs):
        x, d = parent[:, i], delta[:, i]
        lin = np.polyfit(x, d, 1)
        quad = np.polyfit(x, d, 2)

        cuts = np.quantile(x, np.linspace(q, 1 - q, n_bins - 1))
        which = np.digitize(x, cuts)
        binned = []
        for b in range(n_bins):
            k = which == b
            if k.sum() < 20:
                continue
            binned.append(dict(
                pc=pc, bin=b, position=float(x[k].mean()),
                drift=float(d[k].mean()),
                sem=float(d[k].std(ddof=1) / np.sqrt(k.sum())),
                n=int(k.sum()),
            ))
        if not binned:
            continue

        bx = np.array([r["position"] for r in binned])
        bd = np.array([r["drift"] for r in binned])
        ss = np.sum((bd - bd.mean()) ** 2)
        r2 = lambda pred: float(1 - np.sum((bd - pred) ** 2) / ss) if ss else np.nan

        model, knots = _fit_drift_spline(x, d, groups, n_knots=n_knots)
        lo, hi = bx.min(), bx.max()
        xs = np.linspace(lo, hi, 600)
        ys = model.predict(xs.reshape(-1, 1))
        curves[pc] = (xs, ys)
        chosen_knots[pc] = knots

        roots = _drift_roots(xs, ys, x)
        stable_roots = [r for r in roots if r[1] == "stable"]
        unstable_roots = [r for r in roots if r[1] == "unstable"]
        # If several attractors appear, report the one holding the most champions.
        if stable_roots:
            fp, _, pct = max(
                stable_roots,
                key=lambda r: ((x >= (r[0] - 1)) & (x <= (r[0] + 1))).sum(),
            )
        else:
            fp, pct = np.nan, np.nan

        # Local restoring strength: the slope of the fitted curve at the fixed point.
        if not np.isnan(fp):
            h = (hi - lo) / 400
            k_local = float((model.predict(np.array([[fp + h]]))[0]
                             - model.predict(np.array([[fp - h]]))[0]) / (2 * h))
        else:
            k_local = np.nan

        for r in binned:
            r.update(
                n_knots=knots,
                quad_a=float(quad[0]), quad_b=float(quad[1]), quad_c=float(quad[2]),
                linear_slope=float(lin[0]), linear_intercept=float(lin[1]),
                fixed_point=float(fp) if not np.isnan(fp) else np.nan,
                fixed_point_percentile=float(pct) if not np.isnan(pct) else np.nan,
                interior=bool(5.0 <= pct <= 95.0) if not np.isnan(pct) else False,
                n_stable_fixed_points=len(stable_roots),
                n_unstable_fixed_points=len(unstable_roots),
                unstable_fixed_point=unstable_roots[0][0] if unstable_roots else np.nan,
                restoring_slope=k_local,
                r2_binned_linear=r2(np.polyval(lin, bx)),
                r2_binned_quad=r2(np.polyval(quad, bx)),
                r2_binned_spline=r2(model.predict(bx.reshape(-1, 1))),
            )
        rows.extend(binned)

    out = pd.DataFrame(rows)
    # The dense fitted curves do not fit the one-row-per-bin shape, so they travel
    # alongside the frame for plot_per_pc_drift to draw.
    out.attrs["curves"] = curves
    out.attrs["n_knots"] = chosen_knots
    return out


def plot_per_pc_drift(drift_df, pcs=PC_COLUMNS, n_cols=3, save_path=None):
    """
    Figure: one panel per PC showing 1-D drift against position.

    Each point is a bin of champion transitions — position is the mean PC value of the
    champions in that bin, drift is their mean displacement along that PC over the next
    generation, and the error bar is one standard error of that mean. The curve is the
    cross-validated spline from per_pc_drift, fitted to *all* raw transitions rather than
    to the binned points, so it is not expected to pass exactly through them; the bins are
    a readable summary of a cloud of 54,480 individual steps.

    The dotted line marks the stable fixed point — where drift crosses zero downwards,
    i.e. the value that axis pulls towards. A faint dashed line marks an unstable root
    when one falls inside the sampled range.
    """
    n_rows = int(np.ceil(len(pcs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.3 * n_rows),
                             dpi=150, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, pc in zip(axes, pcs):
        g = drift_df[drift_df["pc"] == pc]
        if g.empty:
            ax.axis("off"); continue
        fp = g["fixed_point"].iloc[0]
        pct = g["fixed_point_percentile"].iloc[0]
        inside = bool(g["interior"].iloc[0])
        unstable = g["unstable_fixed_point"].iloc[0]
        k = g["restoring_slope"].iloc[0]
        r2s = g["r2_binned_spline"].iloc[0]
        r2q = g["r2_binned_quad"].iloc[0]
        knots = int(g["n_knots"].iloc[0])

        ax.axhline(0, color="#999999", lw=0.8, ls="--")
        ax.errorbar(g["position"], g["drift"], yerr=g["sem"], fmt="o", ms=4,
                    color="#1f3b70", ecolor="#8fa8c8", capsize=2, lw=1)
        curve = (drift_df.attrs.get("curves") or {}).get(pc)
        if curve is not None:
            ax.plot(curve[0], curve[1], color="#c0392b", lw=1.8)

        if not np.isnan(fp):
            ax.axvline(fp, color="#c0392b", lw=1.0, ls=":")
        if not np.isnan(unstable):
            ax.axvline(unstable, color="#c0392b", lw=0.8, ls="--", alpha=0.45)

        where = "interior" if inside else "at edge of range"
        note = (f"fixed point {fp:+.2f} (percentile {pct:.0f}, {where})\n"
                f"restoring slope {k:+.3f}   spline $R^2$ {r2s:.2f} "
                f"({knots} knots; quad {r2q:.2f})")
        ax.annotate(note, xy=(0.03, 0.04), xycoords="axes fraction", fontsize=7.5,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
        ax.set_xlabel(f"{pc} (position)", fontweight="bold", fontsize=9)
        ax.set_ylabel(f"mean Δ{pc} per generation", fontsize=9)
        ax.set_title(pc, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(pcs):]:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def pc_signal_table(results, wide, matchups, drift_df, pcs=PC_COLUMNS) -> pd.DataFrame:
    """
    Per-PC summary: how much preference that PC carries on its own, its Bradley-Terry
    weight in the full six-PC model, and the stable fixed point of its 1-D drift with the
    local restoring slope there.

    Table companion to plot_per_pc_drift; the per-plane numbers live in pair_summary.
    """
    from sklearn.linear_model import LogisticRegression

    coords = wide.drop_duplicates("vase_id").set_index("vase_id")[list(pcs)]
    W = coords.reindex(matchups["winner_vase_id"]).to_numpy()
    L = coords.reindex(matchups["loser_vase_id"]).to_numpy()
    X = np.vstack([W - L, L - W])
    y = np.r_[np.ones(len(W)), np.zeros(len(W))]
    weights = LogisticRegression(fit_intercept=False).fit(X, y).coef_[0]

    rows = []
    for i, pc in enumerate(pcs):
        g = drift_df[drift_df["pc"] == pc]
        best_plane = max(
            (r for r in results.values() if pc in (r["pc_x"], r["pc_y"])),
            key=lambda r: r["win_rate"],
        )
        rows.append(dict(
            PC=pc,
            win_rate_alone=max((W[:, i] > L[:, i]).mean(), (W[:, i] < L[:, i]).mean()),
            bt_weight=weights[i],
            restoring_slope=g["restoring_slope"].iloc[0] if not g.empty else np.nan,
            fixed_point=g["fixed_point"].iloc[0] if not g.empty else np.nan,
            fixed_point_pct=g["fixed_point_percentile"].iloc[0] if not g.empty else np.nan,
            interior=bool(g["interior"].iloc[0]) if not g.empty else False,
            best_plane=f"{best_plane['pc_x']} vs {best_plane['pc_y']}",
            best_plane_win_rate=best_plane["win_rate"],
        ))
    return pd.DataFrame(rows)


def pair_summary(results) -> pd.DataFrame:
    """
    One row per PC plane, sorted by matchup win rate.

    The win rate says how much aesthetic preference that plane carries: 0.5 means the
    plane's coordinates tell you nothing about which vase a person picked.
    """
    rows = [
        dict(pc_x=r["pc_x"], pc_y=r["pc_y"], eigenvalue=float(r["eigenvalue"].real),
             spectral_radius=r["spectral_radius"], win_rate=r["win_rate"],
             n_matchups=r["n_matchups"])
        for r in results.values()
    ]
    return (pd.DataFrame(rows).sort_values("win_rate", ascending=False)
            .reset_index(drop=True))


def plot_pc_pair(result, drift=None, ax=None, show_streamlines=True, title=None):
    """
    One PC plane: gravity potential as filled contours, with either the measured drift
    field (quiver, arrow length in PC units per generation) or the potential's gradient
    (streamlines) on top.

    Pass drift= to show measured movement. Streamlines show only which way the fitted
    potential rises and how steeply, in relative units — they are not observed motion.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6.2, 5.4), dpi=150)

    norm, cmap = gravity_colour_norm(result["phi_2D"])
    cf = ax.contourf(result["grid_x"], result["grid_y"], result["phi_2D"],
                     levels=40, cmap=cmap, norm=norm, alpha=0.92, antialiased=True)

    if drift is not None:
        m = ~np.isnan(drift["U"])
        ax.quiver(drift["grid_x"][m], drift["grid_y"][m], drift["U"][m], drift["V"][m],
                  color="#111111", angles="xy", scale_units="xy", scale=1.0,
                  width=0.004, headwidth=3.5, alpha=0.85)
    elif show_streamlines:
        speed = np.sqrt(result["U"] ** 2 + result["V"] ** 2)
        ax.streamplot(result["grid_x"], result["grid_y"], result["U"], result["V"],
                      color="#C000C0", linewidth=2.2 * speed / (speed.max() + 1e-9),
                      density=1.6, arrowstyle="-|>", arrowsize=0.7)

    ax.set_xlim(*result["bounds"]["xlim"]); ax.set_ylim(*result["bounds"]["ylim"])
    ax.set_xlabel(result["pc_x"], fontweight="bold")
    ax.set_ylabel(result["pc_y"], fontweight="bold")
    ax.set_title(title or f"{result['pc_x']} vs {result['pc_y']}   "
                          f"(wins {result['win_rate']:.1%})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    if created:
        plt.tight_layout()
    return ax, cf


def plot_pc_pair_grid(results, drifts=None, n_cols=5, save_path=None,
                      suptitle="Koopman gravity across PC planes"):
    """All PC planes on one sheet, ordered by descending matchup win rate."""
    order = [(r["pc_x"], r["pc_y"]) for _, r in pair_summary(results).iterrows()]
    n = len(order)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.6 * n_rows), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, order):
        plot_pc_pair(results[key], drift=None if drifts is None else drifts.get(key), ax=ax)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=15, y=1.002)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  4. Synthesising vases from PC values
# ─────────────────────────────────────────────────────────────────────────────
#
#  The PCA was fitted on profiles encoded as [x_1..x_250, y_1..y_250], so a vase is
#  recovered from its PC scores by the inverse transform, X = scores @ components + mean.
#  Checked against a stored profile this reproduces the outline to an RMSE of 0.031 on an
#  x-range of 3.0, i.e. about 1% — six components capture the shape well enough to draw.
#
#  This is what makes the maps legible: instead of asking a reader to imagine what
#  "PC3 = +5" looks like, the figure shows the vase.

PCA_DIR = "../file_processing/pca_csv"


def load_pca_basis(pca_dir: str = PCA_DIR):
    """
    Load the saved PCA basis used to generate the vases in the experiment.

    Returns (components, mean, n_points): components is (6, 500), mean is (500,), and
    n_points is 250 — the first half of a feature vector is x, the second half y.
    """
    components = pd.read_csv(f"{pca_dir}/pc_vals.csv", index_col=0).values
    mean = pd.read_csv(f"{pca_dir}/pc_val_means.csv")["pc_mean"].values
    return components, mean, len(mean) // 2


def synthesise_vase(pc_values, components, mean, n_points=None):
    """
    Reconstruct a vase outline from PC scores — the inverse PCA transform.

    pc_values may be shorter than the number of components; missing entries are treated
    as zero, i.e. held at the corpus mean shape.

    Returns (x, y), each of length n_points.
    """
    n_points = n_points or len(mean) // 2
    scores = np.zeros(components.shape[0])
    scores[: len(pc_values)] = np.asarray(pc_values, dtype=float)
    flat = scores @ components + mean
    return flat[:n_points], flat[n_points:]


def fit_beauty_model(wide, matchups, pcs=PC_COLUMNS, quadratic=True):
    """
    A Bradley-Terry model of which vase wins a head-to-head, as a function of PC values.

    The symmetric design (each matchup entered both ways, no intercept) makes this a
    proper paired-comparison model rather than a classifier with an arbitrary baseline.
    Quadratic terms are included by default because preference peaks at an interior
    optimum rather than rising without bound — they lift matchup accuracy from 0.620 to
    0.634.

    Returns a callable score(P) mapping an (n, 6) array of PC values to a scalar
    "beauty" utility, with the fitted accuracy attached as score.accuracy.
    """
    from sklearn.linear_model import LogisticRegression

    coords = wide.drop_duplicates("vase_id").set_index("vase_id")[list(pcs)]
    W = coords.reindex(matchups["winner_vase_id"]).to_numpy()
    L = coords.reindex(matchups["loser_vase_id"]).to_numpy()

    design = (lambda A: np.hstack([A, A ** 2])) if quadratic else (lambda A: A)
    X = np.vstack([design(W) - design(L), design(L) - design(W)])
    y = np.r_[np.ones(len(W)), np.zeros(len(W))]
    model = LogisticRegression(fit_intercept=False, max_iter=2000).fit(X, y)

    def score(P):
        P = np.atleast_2d(np.asarray(P, dtype=float))
        return design(P) @ model.coef_[0]

    score.accuracy = float(model.score(X, y))
    score.coef = model.coef_[0]
    return score


def _draw_vase(ax, x, y, cx, cy, scale, facecolor, lw=0.5):
    """Draw one closed outline centred on (cx, cy) at a given scale."""
    x = x - x.mean()
    y = y - y.mean()
    ax.fill(cx + x * scale, cy + y * scale, facecolor=facecolor,
            edgecolor="#333333", linewidth=lw, zorder=3)


def _standardise(values, reference):
    """
    Express a Koopman field in standard deviations of itself over the vases that exist.

    An eigenvector has arbitrary norm, so raw Φ values from two different fits are not on
    the same scale and cannot share a colourbar. Dividing by the spread of Φ across the
    real champion positions puts every field in comparable units — "how unusual is this
    position for this field" — without pretending the absolute numbers mean anything.
    """
    sd = float(np.std(reference))
    return (np.asarray(values) - float(np.mean(reference))) / (sd if sd > 1e-12 else 1.0)


def plot_pc_pair_vases(result, wide, components, mean, baseline=None, n_show=5,
                       ax=None, cell_fill=0.8, norm=None, cmap=None, annotate=None):
    """
    One PC plane drawn as a lattice of synthesised vases, coloured by the gravity field.

    At each lattice point the two PCs of this plane are set to that position and the
    other four are held at `baseline`, so every shape difference visible across the panel
    is caused by those two PCs alone.

    Colour is that plane's Koopman potential Φ, evaluated exactly at the lattice point
    from the plane's own eigenfunction rather than interpolated from the plotted grid, and
    standardised over the plane's real vase positions so panels are comparable. Red is
    high gravity, blue low. Note this is the *dynamical* field — where the selection
    process dwells and flows — which is not the same thing as a direct model of which
    vase wins a comparison; the two diverge, and pair_summary's win rate is how much they
    agree on each plane.

    All vases in a panel share one scale factor, so genuine size differences between
    shapes stay visible rather than being normalised away.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.4, 4.8), dpi=150)

    b = result["bounds"]
    xs = np.linspace(b["x_min"], b["x_max"], n_show)
    ys = np.linspace(b["y_min"], b["y_max"], n_show)
    base = np.zeros(len(PC_COLUMNS)) if baseline is None else np.asarray(baseline, float)
    ix, iy = PC_COLUMNS.index(result["pc_x"]), PC_COLUMNS.index(result["pc_y"])

    keys = [(px, py) for px in xs for py in ys]
    shapes = {}
    for px, py in keys:
        pcs = base.copy()
        pcs[ix], pcs[iy] = px, py
        shapes[(px, py)] = synthesise_vase(pcs, components, mean)

    lattice = np.array(keys).T                      # (2, n) of (pc_x, pc_y) positions
    phi_lattice = compute_gravity_scores(lattice, result["eigenvector"],
                                         result["centres"], result["sigma"])
    vases = wide.drop_duplicates("vase_id")
    phi_real = compute_gravity_scores(
        np.vstack([vases[result["pc_x"]].to_numpy(), vases[result["pc_y"]].to_numpy()]),
        result["eigenvector"], result["centres"], result["sigma"])
    phi_lattice = _standardise(phi_lattice, phi_real)
    utilities = dict(zip(keys, phi_lattice))

    extent = max(max(np.ptp(sx), np.ptp(sy)) for sx, sy in shapes.values())
    cell = min((xs[1] - xs[0]) if n_show > 1 else 1.0,
               (ys[1] - ys[0]) if n_show > 1 else 1.0)
    scale = cell_fill * cell / max(extent, 1e-9)

    if norm is None or cmap is None:
        lim = max(np.abs(phi_lattice).max(), 1e-9)
        norm = plt.matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
        cmap = plt.cm.RdBu_r
    for key, (sx, sy) in shapes.items():
        _draw_vase(ax, sx, sy, key[0], key[1], scale, cmap(norm(utilities[key])))

    pad_x, pad_y = 0.6 * cell, 0.6 * cell
    ax.set_xlim(xs[0] - pad_x, xs[-1] + pad_x)
    ax.set_ylim(ys[0] - pad_y, ys[-1] + pad_y)
    # The bold axis labels name both components, so a "PC1 vs PC2" title would only
    # repeat them and, in a tight grid, collide with the row above.
    ax.set_xlabel(result["pc_x"], fontweight="bold", fontsize=10)
    ax.set_ylabel(result["pc_y"], fontweight="bold", fontsize=10)
    if annotate:
        ax.annotate(annotate, xy=(0.98, 0.97), xycoords="axes fraction", ha="right",
                    va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc",
                              alpha=0.85), zorder=5)
    # Equal data aspect so a vase keeps its true proportions; without it each panel
    # stretches the outlines by the ratio of its PC ranges to its box shape.
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", color="#EEEEEE", alpha=0.6, zorder=0)
    if created:
        plt.tight_layout()
    return ax, phi_lattice


def plot_pc_pair_vase_grid(results, wide, components, mean, baseline=None, n_show=5,
                           n_cols=5, save_path=None, panel_size=None,
                           suptitle="Vase shape and gravity across every PC plane"):
    """
    All 15 PC planes as lattices of synthesised vases, ordered by matchup win rate.

    The companion to the gravity-field sheet in section 7: that one draws Φ as contours,
    this one draws the pots that Φ is defined over, coloured by the same field.

    Colour is normalised once across all 15 panels so a red vase means the same thing in
    each, after each plane's Φ has been put into units of its own spread (eigenvectors
    have arbitrary norm, so raw values are not comparable between fits).

    Matchup win rates are not annotated here; they live in pair_summary. Their floor sits
    near 0.502 rather than 0.5, because the sign convention forces the rate upwards, so a
    bare number on a panel would read as a small positive effect where there is none.
    Pass `annotate` to plot_pc_pair_vases directly to label an individual panel.

    `panel_size` is the width in inches allocated to each panel. It defaults to about
    0.55 inches per vase, so raising `n_show` enlarges the sheet rather than shrinking
    the pots — a 10x10 lattice in a panel sized for 5x5 leaves each outline a quarter of
    the area. Axes are held at equal data aspect either way, so the shapes are never
    distorted; only their size on the page changes.
    """
    order = [(r["pc_x"], r["pc_y"]) for _, r in pair_summary(results).iterrows()]
    base = np.zeros(len(PC_COLUMNS)) if baseline is None else np.asarray(baseline, float)
    vases = wide.drop_duplicates("vase_id")

    # First pass: the global colour range, on the same lattice the panels will draw.
    all_phi = []
    for key in order:
        res = results[key]
        b = res["bounds"]
        lattice = np.array([(px, py)
                            for px in np.linspace(b["x_min"], b["x_max"], n_show)
                            for py in np.linspace(b["y_min"], b["y_max"], n_show)]).T
        phi_real = compute_gravity_scores(
            np.vstack([vases[res["pc_x"]].to_numpy(), vases[res["pc_y"]].to_numpy()]),
            res["eigenvector"], res["centres"], res["sigma"])
        all_phi.append(_standardise(
            compute_gravity_scores(lattice, res["eigenvector"], res["centres"],
                                   res["sigma"]), phi_real))
    all_phi = np.concatenate(all_phi)

    # Clip at the 95th percentile: the far corners of the wider planes are combinations
    # almost nobody visited, and letting them set the scale flattens every panel.
    lim = max(float(np.quantile(np.abs(all_phi), 0.95)), 1e-9)
    norm = plt.matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    cmap = plt.cm.RdBu_r

    if panel_size is None:
        panel_size = max(4.0, 0.55 * n_show)
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_size * n_cols, 0.95 * panel_size * n_rows),
                             dpi=150, layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, order):
        plot_pc_pair_vases(results[key], wide, components, mean, baseline=baseline,
                           n_show=n_show, ax=ax, norm=norm, cmap=cmap)
    for ax in axes[len(order):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cbar.outline.set_visible(False)
    cbar.set_label("Koopman gravity Φ, in SDs of each plane's own field "
                   "(clipped at the 95th percentile)", fontsize=9)

    fig.suptitle(suptitle, fontsize=15)
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_pc_1d_vase_scan(wide, axis_fields, components, mean, pcs=PC_COLUMNS,
                         baseline=None, n_show=9, q=0.02, n_cols=2, save_path=None,
                         panel_width=None, panel_height=None,
                         suptitle="Shape and gravity along each PC"):
    """
    One row per PC: a scan along that axis alone, with the other five held at `baseline`.

    Every shape difference along a row is therefore caused by that single PC. Vases are
    coloured by that PC's own one-dimensional Koopman gravity field from run_pc_axis —
    red where gravity is high, blue where it is low — so each row reads as "what does this
    component do to the pot, and where along it does the selection process settle".

    A 1-D field is used rather than a slice through a plane fit, because a slice would
    depend on which partner PC happened to be chosen. Each field is expressed in standard
    deviations of itself across the real champion positions, since eigenvectors carry an
    arbitrary norm and the six fits would otherwise not share a colour scale.

    The scan spans the 2nd to 98th percentile of the champion positions actually visited,
    so it covers the design space participants explored rather than extrapolating.

    `n_show` sets how many vases appear per row. The axes are held at equal data aspect
    and the panel is sized from the drawn extent, so the outlines keep their true
    proportions at any n_show — raising it makes the vases smaller and the strip
    narrower, never squashed. By default the panel widens at about 0.85 inches per vase
    so they stay a readable size; pass `panel_width` to override. To scan one component
    in detail, pass a single PC and one column, e.g.

        plot_pc_1d_vase_scan(wide, axis_fields, components, mean,
                             pcs=["PC1"], n_show=21, n_cols=1)
    """
    base = np.zeros(len(PC_COLUMNS)) if baseline is None else np.asarray(baseline, float)
    vases = wide.drop_duplicates("vase_id")

    rows = []
    for pc in pcs:
        i = PC_COLUMNS.index(pc)
        grid = np.linspace(wide[pc].quantile(q), wide[pc].quantile(1 - q), n_show)
        shapes = []
        for v in grid:
            p = base.copy(); p[i] = v
            shapes.append(synthesise_vase(p, components, mean))
        field = axis_fields[pc]
        phi = _standardise(field["phi"](grid), field["phi"](vases[pc].to_numpy()))
        rows.append((pc, grid, shapes, phi))

    lim = max(float(np.abs(np.concatenate([r[3] for r in rows])).max()), 1e-9)
    norm = plt.matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    cmap = plt.cm.RdBu_r

    # Size the panel from the geometry so equal aspect costs no distortion: the tallest
    # vase sets the vertical extent, the scan range the horizontal one.
    widest = 0.0
    for pc, grid, shapes, phi in rows:
        extent = max(max(np.ptp(sx), np.ptp(sy)) for sx, sy in shapes)
        step = grid[1] - grid[0] if len(grid) > 1 else 1.0
        scale = 0.85 * step / max(extent, 1e-9)
        half = 0.5 * max(np.ptp(sy) for _, sy in shapes) * scale
        widest = max(widest, 2.4 * half / (grid[-1] - grid[0] + 2 * step))
    if panel_width is None:
        panel_width = max(7.4, 0.85 * n_show)
    if panel_height is None:
        panel_height = panel_width * widest + 0.9      # + room for the axis label

    n_rows = int(np.ceil(len(pcs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_width * n_cols, panel_height * n_rows),
                             dpi=150, layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    for ax, (pc, grid, shapes, phi) in zip(axes, rows):
        extent = max(max(np.ptp(sx), np.ptp(sy)) for sx, sy in shapes)
        step = grid[1] - grid[0] if len(grid) > 1 else 1.0
        scale = 0.85 * step / max(extent, 1e-9)
        for v, (sx, sy), u in zip(grid, shapes, phi):
            _draw_vase(ax, sx, sy, v, 0.0, scale, cmap(norm(u)), lw=0.45)
        half = 0.5 * max(np.ptp(sy) for _, sy in shapes) * scale
        ax.set_xlim(grid[0] - step, grid[-1] + step)
        ax.set_ylim(-1.2 * half, 1.2 * half)
        # Equal data aspect: without it the axes stretch one PC unit differently in x and
        # y, and the outlines are distorted by whatever shape the panel happens to be.
        ax.set_aspect("equal", adjustable="box")
        ax.set_yticks([])
        # The bold axis label names the component; a title as well would just repeat it
        # and, in a short panel, collide with the row above.
        ax.set_xlabel(pc, fontweight="bold", fontsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)

    for ax in axes[len(rows):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.012)
    cbar.outline.set_visible(False)
    cbar.set_label("Koopman gravity Φ, in SDs of each axis's own field", fontsize=10)

    fig.suptitle(suptitle, fontsize=15)
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def run_pc_axis(wide, edges, matchups, pc, n_basis=24, ridge=1e-3, mode=1,
                chunk_size=10_000, q=0.005):
    """
    Fit the Koopman operator along a *single* PC — the one-dimensional counterpart of
    run_pc_pair, giving that axis its own gravity field.

    The plane fits cannot supply this: each is a 2-D field, and reading a line out of one
    would make the answer depend on which partner PC was chosen. Here the state is the
    champion's position on this PC alone, so the field describes how the selection process
    moves along that component with everything else marginalised out.

    Implemented by reusing the 2-D machinery with the second coordinate pinned to zero,
    which reduces the Gaussian RBFs to a 1-D basis; that keeps the ridge regularisation,
    mode selection and sign convention identical to the plane fits.

    The sign is fixed the same way as everywhere else — so that the higher-Φ vase is the
    one people chose — because an eigenvector is only defined up to sign. That is the
    only role the matchups play; the values themselves are the Koopman field.

    Returns a dict with eigenvector, centres, sigma, bounds, eigenvalue, win_rate and a
    `phi(values)` callable evaluating the field at arbitrary positions on the axis.
    """
    coords = wide.drop_duplicates("vase_id").set_index("vase_id")[pc]
    parent = coords.reindex(edges["parent_vase_id"]).to_numpy()
    child = coords.reindex(edges["child_vase_id"]).to_numpy()
    ok = ~(np.isnan(parent) | np.isnan(child))
    parent, child = parent[ok], child[ok]

    lo = float(wide[pc].quantile(q))
    hi = float(wide[pc].quantile(1 - q))
    grid = np.linspace(lo, hi, n_basis)
    centres = np.column_stack([grid, np.zeros(n_basis)])
    sigma = float(np.median(np.diff(grid)))

    Z = np.vstack([parent, np.zeros_like(parent)])
    Z_prime = np.vstack([child, np.zeros_like(child)])
    K = build_koopman_operator(Z, Z_prime, centres, sigma, chunk_size,
                               ridge=ridge, verbose=False, progress=False)
    evals, _, idx, evec = extract_koopman_mode(K, mode, verbose=False)

    def phi(values):
        v = np.atleast_1d(np.asarray(values, dtype=float))
        xy = np.vstack([v, np.zeros_like(v)])
        return compute_gravity_scores(xy, evec, centres, sigma, chunk_size)

    scores = pd.Series(phi(coords.to_numpy()), index=coords.index)
    win_rate, n_matchups = matchup_win_rate(matchups, scores)
    if win_rate < 0.5:
        evec, win_rate = -evec, 1 - win_rate

        def phi(values, _e=evec):
            v = np.atleast_1d(np.asarray(values, dtype=float))
            xy = np.vstack([v, np.zeros_like(v)])
            return compute_gravity_scores(xy, _e, centres, sigma, chunk_size)

    return dict(pc=pc, eigenvector=evec, centres=centres, sigma=sigma,
                bounds=(lo, hi), eigenvalue=evals[idx], phi=phi,
                win_rate=win_rate, n_matchups=n_matchups,
                spectral_radius=float(np.abs(np.linalg.eigvals(K)).max()))


def run_all_pc_axes(wide, edges, matchups, pcs=PC_COLUMNS, verbose=True, **kwargs):
    """Fit a 1-D Koopman gravity field for every PC. Returns {pc: result}."""
    out = {}
    for pc in pcs:
        out[pc] = run_pc_axis(wide, edges, matchups, pc, **kwargs)
        if verbose:
            r = out[pc]
            print(f"  {pc}: lambda = {r['eigenvalue'].real:.4f}   "
                  f"radius = {r['spectral_radius']:.4f}   win rate = {r['win_rate']:.4f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  5. How strong is selection on each PC?
# ─────────────────────────────────────────────────────────────────────────────

def selection_strength(wide, matchups, pcs=PC_COLUMNS, n_boot=300, seed=0) -> pd.DataFrame:
    """
    Quantify selection on each PC, separating what participants chose from what the
    generator happened to offer.

    Three quantities, all in standard deviations of the PC among the vases on offer so
    that components with different spreads are comparable:

    selection_i
        The standardised selection differential — the classical intensity of selection.
        Each matchup offers two vases and one is kept, so the differential for that
        matchup is (winner - loser) / 2, i.e. the winner's departure from the mean of
        what was available. Averaged over all matchups this is how far selection moves
        the population per generation. This is the honest measure of "how strongly do
        people select on this component".

    generator_bias
        The mean (challenger - incumbent) among the offers. The challenger is drawn by
        the generator, not chosen by anyone, so any systematic offset here pushes the
        champion population around regardless of preference.

    realised_change
        The actual shift in the champion population from generation 1 to generation 5.
        It is the *sum* of the two effects above, so it must not be read as selection:
        on PC2 the realised change is -0.150 SD while selection is +0.008, because the
        generator offers challengers 0.110 SD below the incumbent. The pool drifts down
        PC2 while participants are indifferent to it.

    Confidence intervals are bootstrapped over participants, since each contributes 25
    correlated comparisons.

    Returns one row per PC, sorted by the magnitude of the selection differential.
    """
    rng = np.random.default_rng(seed)
    coords = wide.drop_duplicates("vase_id").set_index("vase_id")[list(pcs)]
    W = coords.reindex(matchups["winner_vase_id"]).to_numpy()
    L = coords.reindex(matchups["loser_vase_id"]).to_numpy()
    sd = np.vstack([W, L]).std(axis=0, ddof=1)

    differential = (W - L) / 2
    people = matchups["participant_id"].to_numpy()
    unique = np.array(sorted(set(people)))
    index_of = {p: np.where(people == p)[0] for p in unique}
    boots = np.array([
        differential[np.concatenate(
            [index_of[p] for p in rng.choice(unique, len(unique), replace=True)]
        )].mean(axis=0) / sd
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    later = wide[wide["generation_num"] >= 2]
    keys = ["participant_id", "play_counter", "generation_num"]
    offers = (later[~later["is_parent_vase"]].set_index(keys)[list(pcs)]
              .join(later[later["is_parent_vase"]].set_index(keys)[list(pcs)],
                    lsuffix="_c", rsuffix="_i").dropna())
    bias = (offers[[c + "_c" for c in pcs]].to_numpy()
            - offers[[c + "_i" for c in pcs]].to_numpy()).mean(axis=0) / sd

    champions = wide[wide["is_selected_vase"]]
    first = champions[champions["generation_num"] == 1][list(pcs)].mean().to_numpy()
    last = champions[champions["generation_num"] == champions["generation_num"].max()]
    realised = (last[list(pcs)].mean().to_numpy() - first) / sd

    out = pd.DataFrame({
        "PC": list(pcs), "sd": sd,
        "selection_i": differential.mean(axis=0) / sd,
        "ci_lo": lo, "ci_hi": hi,
        "generator_bias": bias,
        "realised_change": realised,
    })
    out["selection_dominates"] = out["selection_i"].abs() > out["generator_bias"].abs()
    return out.reindex(out["selection_i"].abs().sort_values(ascending=False).index
                       ).reset_index(drop=True)

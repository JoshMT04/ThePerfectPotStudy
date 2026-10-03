import numpy as np
import cv2


def compute_base_kinematics(xy):
    """
    Computes the foundational calculus and spatial states for the vase coordinate array.
    Parameters:
    - xy: A numpy array of shape (N, 2) containing the x and y coordinates of the vase contour points.
    Returns:
    A dictionary containing the computed values:
    - 'x': The x coordinates of the contour points.
    - 'y': The y coordinates of the contour points.
    - 'dx': The first derivative of x with respect to the contour points.
    - 'dy': The first derivative of y with respect to the contour points.
    - 'k': The curvature of the contour at each point.
    - 'x_max': The maximum x coordinate (bounding box).
    - 'x_min': The minimum x coordinate (bounding box).
    - 'y_max': The maximum y coordinate (bounding box).
    - 'y_min': The minimum y coordinate (bounding box).
    - 'area': The area of the contour computed using the shoelace formula.
    - 'perimeter': The perimeter of the contour.
    - 'cx': The x coordinate of the centroid of the contour.
    - 'cy': The y coordinate of the centroid of the contour.
    - 'moments': The image moments computed from the contour points.
    """
    x = xy[:, 0]
    y = xy[:, 1]

    # Bounding Box
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    width = x_max - x_min
    height = y_max - y_min

    # 1st and 2nd Derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature (k)
    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator == 0] = 1e-8
    k = (dx * ddy - dy * ddx) / denominator

    # Shoelace Area & Contour Perimeter
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    perimeter = np.sum(np.sqrt(np.diff(np.append(x, x[0]))**2 + np.diff(np.append(y, y[0]))**2))

    # 2D Centroid (Image Moments)
    # Shift to positive coordinates for cv2 processing
    xy_shifted = xy - np.min(xy, axis=0)
    moments = cv2.moments(xy_shifted.astype(np.float32))
    cx = (moments['m10'] / moments['m00']) + x_min if moments['m00'] != 0 else 0
    cy = (moments['m01'] / moments['m00']) + y_min if moments['m00'] != 0 else 0

    return {
        'x': x, 'y': y, 'dx': dx, 'dy': dy, 'k': k,
        'x_max': x_max, 'x_min': x_min, 'y_max': y_max, 'y_min': y_min,
        'area': area, 'perimeter': perimeter,
        'cx': cx, 'cy': cy, 'moments': moments, 'height': height, 'width': width
    }


def compute_half_profile(xy):
    """
    Isolates the right-hand half-profile (base to lip) from a closed, mirrored
    vase contour and computes its own derivatives/curvature.

    Every landmark-based feature below (lip, belly, neck, base widths; bending
    energy; inflection count; etc.) is defined on this single silhouette curve,
    not the full closed contour, so it needs its own derivatives: the closed
    contour's curvature is contaminated at the two mirror seams (base and lip),
    where left and right halves join.

    Parameters:
    - xy: A numpy array of shape (N, 2), the full closed, mirrored contour
      (x spans negative-to-positive; this is the raw per-vase point cloud).

    Returns a dictionary:
    - 'x', 'y': the half-profile coordinates, ordered base (y_min) -> lip (y_max).
    - 'dx', 'dy', 'k': first derivative and curvature along that ordering.
    - 'x_base', 'x_lip', 'x_max', 'y_at_max': the base width, lip width, belly
      (max) width, and the y-position at which the belly occurs.
    - 'x_neck', 'y_neck': the narrowest point in the upper half (above the
      vertical midpoint) and its height.
    """
    x = xy[:, 0]
    y = xy[:, 1]

    # The raw contour is a trace (lip -> down the right side -> base -> up the
    # left side -> back to the lip); the right half (x >= 0) is a single
    # contiguous block in valid curve order and must be used as-is for
    # gradient/curvature purposes — re-sorting by y silently scrambles
    # point-adjacency right where the curve rounds off at the cap and foot
    # (the only places y can locally wiggle across the x=0 crossing),
    # corrupting curvature exactly where it concentrates most.
    #
    # That same wiggle means the true base/lip extrema are not reliably at
    # the array boundaries of this block (the crossing can land a point or
    # two before/after the actual y-extremum), so landmarks are found with
    # argmin/argmax on y rather than assumed to sit at index 0 / -1.
    mask = x >= 0
    xh, yh = x[mask], y[mask]

    # Integrals below (volume, surface area, centre of mass) need a known,
    # consistent direction of travel in y — np.trapezoid silently flips sign
    # if the array runs backward — so the whole block is reversed to run
    # base -> lip when the trace visits them in the opposite order. This is
    # a pure reversal (adjacency, and therefore curvature, is unaffected by
    # which end you start from), decided from the true extrema positions
    # rather than boundary values, since those are the same wobble-prone
    # points landmark-finding below has to work around.
    if np.argmin(yh) > np.argmax(yh):
        xh, yh = xh[::-1], yh[::-1]

    half_base = compute_base_kinematics(np.column_stack([xh, yh]))
    dx, dy, k = half_base['dx'], half_base['dy'], half_base['k']

    idx_base = np.argmin(yh)
    idx_lip = np.argmax(yh)
    x_base, y_min = xh[idx_base], yh[idx_base]
    x_lip, y_max = xh[idx_lip], yh[idx_lip]

    idx_belly = np.argmax(xh)
    x_max = xh[idx_belly]
    y_at_max = yh[idx_belly]

    # The "wall" region (cap and foot excluded) — see _wall_bounds for the
    # full rationale. The neck (narrowest point) search is restricted to it:
    # without this, "narrowest point in the upper half" can land right at
    # the lip rim itself for a monotonically-tapering profile, collapsing
    # neck == lip (observed directly: this degenerately zeroed out rim flare
    # angle for such vases). Restricting to the wall forces "neck" to mean a
    # genuine mid-body pinch, which is what waist ratio, rim flare angle, and
    # Birkhoff's landmark chain all actually intend by that word.
    wall_lo, wall_hi = _wall_bounds(dx, dy)

    y_mid = y_min + (y_max - y_min) / 2
    upper_mask = yh > y_mid
    in_wall = np.zeros(len(xh), dtype=bool)
    in_wall[wall_lo:wall_hi + 1] = True
    upper_wall_mask = upper_mask & in_wall
    if not np.any(upper_wall_mask):
        upper_wall_mask = upper_mask  # fallback if the wall doesn't reach the upper half

    if np.any(upper_wall_mask):
        neck_idx_local = np.argmin(xh[upper_wall_mask])
        x_neck = xh[upper_wall_mask][neck_idx_local]
        y_neck = yh[upper_wall_mask][neck_idx_local]
        idx_neck = np.where(upper_wall_mask)[0][neck_idx_local]
    else:
        x_neck, y_neck = x_lip, y_max
        idx_neck = idx_lip

    return {
        'x': xh, 'y': yh, 'dx': dx, 'dy': dy, 'k': k,
        'x_base': x_base, 'x_lip': x_lip, 'x_max': x_max, 'y_at_max': y_at_max,
        'x_neck': x_neck, 'y_neck': y_neck,
        'idx_neck': idx_neck, 'idx_belly': idx_belly, 'idx_lip': idx_lip, 'idx_base': idx_base,
        'wall_lo': wall_lo, 'wall_hi': wall_hi,
    }


# ---------------------------------------------------------------------------
# Macro-Proportions
# ---------------------------------------------------------------------------

def height_width_ratio(height, width):
    """Y_range / X_range — tall and thin vs. short and fat."""
    return height / width


def top_bottom_area_ratio(half, y_min, y_max):
    """
    top_area / (area - top_area), i.e. top_area / bottom_area, split at the
    vertical midpoint. Both half-areas are computed as the definite integral
    of width over height (trapezoidal), which is proportional to true 2D area
    on each side since the contour is left-right symmetric.
    """
    x, y = half['x'], half['y']
    y_mid = y_min + (y_max - y_min) / 2

    top_mask = y > y_mid
    bottom_mask = ~top_mask

    top_area = np.trapezoid(x[top_mask], y[top_mask]) if np.sum(top_mask) > 1 else 0.0
    bottom_area = np.trapezoid(x[bottom_mask], y[bottom_mask]) if np.sum(bottom_mask) > 1 else 1e-8

    return top_area / bottom_area if bottom_area != 0 else 0.0


def lip_to_belly_ratio(half):
    """X_lip / X_max_width — is the opening tightly closed or widely flared?"""
    return half['x_lip'] / half['x_max'] if half['x_max'] > 0 else 0.0


def vase_height(base):
    """Y_range — absolute height of the vase."""
    return base['height']


def maximum_width(base):
    """X_range — the widest point of the vase."""
    return base['width']


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

def critical_tipping_angle(half):
    """
    Arctan( Base_Radius / Height_of_3D_Centre_of_Mass ), in degrees.
    The 3D centre of mass height is found by revolving the half-profile
    around the vertical axis: CoM_y = integral(y * x^2 dy) / integral(x^2 dy).
    """
    x, y = half['x'], half['y']
    weight = x**2
    com_y_abs = np.trapezoid(y * weight, y) / np.trapezoid(weight, y)
    com_height = com_y_abs - y[0]
    base_radius = half['x_base']
    if com_height <= 0:
        return 0.0
    return np.degrees(np.arctan(base_radius / com_height))


def rotational_volume(half):
    """∫ (π * x² * dy) — total volume swept by revolving the profile."""
    x, y = half['x'], half['y']
    return np.pi * np.trapezoid(x**2, y)


# ---------------------------------------------------------------------------
# Complexity and Curvature
# ---------------------------------------------------------------------------

def _wall_slice(half, key):
    """Slices `half[key]` down to the wall region (cap/foot excluded)."""
    return half[key][half['wall_lo']:half['wall_hi'] + 1]


def total_bending_energy(half):
    """
    Σ(k²) over the wall region only — the forced closing curvature at the
    cap and foot (every closed profile must curve back to the centreline
    there) isn't a stylistic choice, so including it would dilute the
    genuine wall-styling signal this feature is meant to capture.
    """
    return np.sum(_wall_slice(half, 'k')**2)


def inflection_count(half):
    """
    COUNT(k_i * k_i+1 < 0) over the wall region — number of curvature sign
    changes (S-curves) in the wall styling, excluding the cap/foot's forced
    inflections (consistent with why this feature exists: counting the
    classical typology's S-curves, which describe the body, not the rim or
    foot ring).
    """
    k = _wall_slice(half, 'k')
    return int(np.sum((k[:-1] * k[1:]) < 0))


def contour_autocorrelation(half):
    """
    ∫( k(t) * k(t+τ) * dt ) over the wall region, reported as the lag τ (in
    profile-points) at which the normalised curvature autocorrelation first
    drops below 0.5 — how quickly the wall "forgets" its current bend moving
    up the vase.
    """
    k = _wall_slice(half, 'k')
    k = k - np.mean(k)
    n = len(k)
    if n < 3 or np.all(k == 0):
        return 0

    acf_full = np.correlate(k, k, mode='full')
    acf = acf_full[n - 1:]
    acf = acf / acf[0] if acf[0] != 0 else acf

    below = np.where(acf < 0.5)[0]
    return int(below[0]) if len(below) > 0 else n - 1


def fractal_dimension(half, n_scales=10):
    """
    LOG(N(ε)) / LOG(1/ε) — 1D box-counting dimension of the wall region,
    estimated as the slope of log(N(eps)) vs log(1/eps). Restricted to the
    wall for the same reason as the other curvature features: the cap/foot's
    forced rounding would otherwise inflate apparent jaggedness regardless
    of how jagged the actual wall styling is.

    x and y are independently normalised to [0, 1] first so the grid scales
    symmetrically regardless of the vase's absolute size (matching the rest
    of this feature set, where size is already captured separately by Height
    and Maximum width). eps ranges over logspace(-2.5, -0.5, n_scales) in
    these normalised units.
    """
    x, y = _wall_slice(half, 'x'), _wall_slice(half, 'y')
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)

    def box_count(eps):
        xbins = np.floor(x_norm / eps)
        ybins = np.floor(y_norm / eps)
        hashed = xbins * 100000 + ybins
        return len(np.unique(hashed))

    epsilons = np.logspace(-2.5, -0.5, n_scales)
    counts = [box_count(eps) for eps in epsilons]
    slope, _ = np.polyfit(np.log(1 / epsilons), np.log(counts), 1)
    return slope


# ---------------------------------------------------------------------------
# Typicality
# ---------------------------------------------------------------------------

def pc_centroid_distance(pc_values):
    """
    sqrt( Σ(PC_i²) ), i = 1..6 — Euclidean distance from the origin of the
    6-PC space fit on real-world pottery profiles.

    This is NOT computable from xy alone: pc_values must already be the
    vase's projection onto the pre-fit PCA basis (see pca_space/pca_func_new.py
    and file_processing/pca_csv/), produced upstream of this module. Pass the
    6 PC scores (PC1..PC6) in; this function only takes their norm.
    """
    pc_values = np.asarray(pc_values, dtype=float)
    return float(np.sqrt(np.sum(pc_values**2)))


# ---------------------------------------------------------------------------
# Typological / Skeleton
# ---------------------------------------------------------------------------

def position_of_maximum_width(half, y_min, y_max):
    """Y_at_max_width / Y_range — how high up the belly sits."""
    height = y_max - y_min
    return (half['y_at_max'] - y_min) / height if height > 0 else 0.0


def waist_ratio(half):
    """
    X_min_width / X_max_width — how pinched the narrowest point is relative
    to the belly. Defined against the belly (max width), not the base, so it
    is free to capture a genuine hourglass/cinched-waist silhouette rather
    than just tracking base width.
    """
    return half['x_neck'] / half['x_max'] if half['x_max'] > 0 else 0.0


def convexity_ratio(xy, base):
    """Area / Convex_Hull_Area — departure from a smooth convex outline."""
    hull = cv2.convexHull(xy.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    return base['area'] / hull_area if hull_area > 0 else 0.0


def rim_flare_angle(half):
    """
    Arctan( dX/dY at Y_lip ) − Arctan( dX/dY at Y_neck ), in degrees.
    Whether the lip curls inward, flares outward, or stays straight relative
    to the neck, using the instantaneous tangent direction at each landmark
    (matching the established pipeline's lip_flare_angle convention).
    """
    dx, dy = half['dx'], half['dy']
    idx_neck, idx_lip = half['idx_neck'], half['idx_lip']

    theta_neck = np.arctan2(dy[idx_neck], dx[idx_neck])
    theta_lip = np.arctan2(dy[idx_lip], dx[idx_lip])

    angle_diff = np.abs(theta_lip - theta_neck) * (180 / np.pi)
    return min(angle_diff, 360 - angle_diff)


def surface_to_volume_ratio(half):
    """
    Surface_Area_3D / Rotational_Volume — 3D efficiency of the form.
    Surface area of revolution: 2*pi * integral( x * sqrt(1 + (dx/dy)^2) dy ).
    """
    x, y, dx, dy = half['x'], half['y'], half['dx'], half['dy']
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.where(dy != 0, dx / dy, 0.0)
    ds = np.sqrt(1 + slope**2)
    surface_area = 2 * np.pi * np.trapezoid(x * ds, y)
    volume = rotational_volume(half)
    return surface_area / volume if volume > 0 else 0.0


# ---------------------------------------------------------------------------
# Birkhoff's Aesthetic Measure
# ---------------------------------------------------------------------------
# Following Post & Van Gelder-style operationalisations of Birkhoff (1933) for
# vase stimuli (as in the Frontiers in Psychology account of O and V): O
# counts simple 1:1 / 1:2 ratio relations among characteristic distances —
# horizontal (H, widths), vertical (V, heights), and cross (HV) — and C counts
# characteristic points (endpoints, width extrema, inflections) doubled for
# both profile sides. M = O / C.
#
# Our generated profiles are far noisier than Birkhoff's idealised vases, so a
# literal point-by-point inflection/extrema count here would just reproduce
# the existing Inflection count and Local-width-extrema signal almost exactly
# (both are dominated by high-frequency wiggle on these shapes). To keep C a
# genuinely different, perceptually-motivated count rather than a relabelling
# of those features, it is computed on a *smoothed* curvature/width signal —
# counting only the macro-scale bends a viewer would actually register.

# Birkhoff restricted vase analysis specifically to 1:1 and 1:2 relations —
# not the golden ratio, which he reserved for other object categories in his
# 1933 book. 0.5 is included only so a ratio can match 1:2 from either side.
_RATIO_TARGETS = (1.0, 2.0, 0.5)


def _order_score(ratio, k=10):
    """
    Continuous closeness (0-1) of `ratio` to 1:1 or 1:2, decaying smoothly
    with log-distance rather than a hard threshold — there is no principled
    tolerance to draw a binary cutoff at (checked empirically: the
    distribution of real landmark ratios has no natural gap away from exact
    ties), so a soft score avoids resting the whole measure on an arbitrary
    number.
    """
    ratio = max(abs(ratio), 1e-6)
    log_r = np.log(ratio)
    min_dist = min(abs(log_r - np.log(t)) for t in _RATIO_TARGETS)
    return np.exp(-k * min_dist)


def _chain_scores(values):
    """
    Order-score across *consecutive* ratios in a sequence, not all pairwise
    combinations — matching the paper's "independent relations" framing
    (their 3-width vase example has 2 independent ratios from 3 widths, not
    the 3 that all-pairs would give, because equalities propagate along the
    chain rather than needing to be checked between every pair).
    """
    return [
        _order_score(values[i] / values[i + 1])
        for i in range(len(values) - 1)
        if values[i + 1] != 0
    ]


def _smooth(arr, frac=1 / 10):
    """Moving-average smoothing, window scaled to the array's own length."""
    n = len(arr)
    window = max(3, int(n * frac) | 1)  # odd window >= 3
    if window >= n:
        return arr
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:n]


def _count_sign_changes(arr, tol_frac=0.02):
    """Sign changes in `arr`, ignoring near-zero noise around a flat region."""
    scale = np.max(np.abs(arr)) if len(arr) > 0 else 0.0
    tol = tol_frac * scale
    mask = np.abs(arr) > tol
    vals = arr[mask]
    if len(vals) < 2:
        return 0
    signs = np.sign(vals)
    return int(np.sum(signs[:-1] != signs[1:]))


def birkhoff_order(half, base):
    """
    O = H + V + HV: continuous 1:1 / 1:2 closeness scores among characteristic
    horizontal distances (H), vertical distances (V), and cross
    horizontal-vertical distances (HV) — summing consecutive-chain scores
    rather than counting all pairwise combinations, per the paper's
    "independent relations" framing.
    """
    # H: widths in profile sequence (base -> neck -> belly -> lip) — 3
    # consecutive ratios.
    widths = [half['x_base'], half['x_neck'], half['x_max'], half['x_lip']]
    H = sum(_chain_scores(widths))

    # V: segment heights between the four landmarks, sorted by their actual
    # vertical order (which need not match base/neck/belly/lip) — 2
    # consecutive ratios between the 3 resulting segments.
    landmark_ys = sorted([base['y_min'], half['y_neck'], half['y_at_max'], base['y_max']])
    segments = list(np.diff(landmark_ys))
    V = sum(_chain_scores(segments))

    # HV: total height:width, and base-to-belly height:belly width.
    total_height = base['y_max'] - base['y_min']
    total_width = 2 * half['x_max'] if half['x_max'] > 0 else 1e-8
    hv1 = total_height / total_width
    hv2 = abs(half['y_at_max'] - base['y_min']) / half['x_max'] if half['x_max'] > 0 else 0.0
    HV = _order_score(hv1) + _order_score(hv2)

    return H + V + HV


def birkhoff_complexity(half):
    """
    C = 2 * (characteristic points per side), where characteristic points are
    the 2 endpoints (base, lip) plus macro-scale width extrema and inflection
    points, counted on smoothed signals so C reflects perceptually salient
    bends rather than the same point-level noise already captured by
    Inflection count / Local width extrema.

    Extrema/inflections are counted on the wall region only: the endpoints
    are already credited explicitly via the "+2" below (matching Birkhoff's
    own base/top landmark points), so counting the cap/foot's forced closing
    curvature again through sign-changes would double-count the same
    structural necessity as a second "characteristic point".
    """
    x, k = _wall_slice(half, 'x'), _wall_slice(half, 'k')
    x_smooth = _smooth(x)
    k_smooth = _smooth(k)

    n_width_extrema = _count_sign_changes(np.diff(x_smooth))
    n_inflections = _count_sign_changes(k_smooth)

    per_side = 2 + n_width_extrema + n_inflections  # 2 endpoints + extrema + inflections
    return 2 * per_side


def birkhoff_measure(half, base):
    """M = O / C, Birkhoff's 1933 aesthetic measure (order over complexity)."""
    O = birkhoff_order(half, base)
    C = birkhoff_complexity(half)
    return O / C if C > 0 else 0.0


# ---------------------------------------------------------------------------
# Hübner Line 4 — SRVF Elastic Shape Distance
# ---------------------------------------------------------------------------
# Matches the established pipeline's own convention (srv_elastic_distance.py):
# the target's x-values are paired with the VASE's own y-grid, not the
# target's — index position stands for shared relative position along the
# profile, and the SRVF framework's own dynamic-programming re-parametrisation
# absorbs any residual phase/speed misalignment this introduces.
#
# The vase is truncated at each end before comparison, because the rounded
# cap (near the lip) and foot (near the base) are a different kind of
# geometry from the wall — comparing them against the target line's own
# shape there would inject differences that have nothing to do with the
# side profile, distorting the distance. Where to cut is found
# geometrically rather than as a fixed point-count fraction: the wall is
# where the tangent runs closer to vertical than horizontal
# (|angle from vertical| < 45 deg); the cap/foot is where it doesn't. The
# cutoff is required to hold for 3 consecutive points so one noisy sample
# can't flip it, and the search is capped at 35% of the profile per side
# (see _wall_bounds) so a vase that's wider than it is tall — a real ~13%
# of this generation space, not a rare edge case — never has its whole
# flared belly mistaken for "cap". Checked across 200 real vases, this
# adapts per vase from ~10% to the 35% cap (median ~19-21%) — nowhere close
# to a single fixed fraction, which is why a flat guess was the wrong tool
# here. The Hübner line4 target itself needs no truncation: checked
# directly, its tangent never exceeds 38.6 deg from vertical anywhere along
# its length — it's already an idealised wall shape with no cap/foot
# artifact to trim.

_HUBNER_LINE4_PATH = 'feature_extraction/hubner_lines_0326/line4_standardised_125.csv'
_WALL_ANGLE_DEG = 45.0
_WALL_RUN_LENGTH = 3
_MAX_TRIM_FRAC = 0.35


def load_hubner_line4(path=_HUBNER_LINE4_PATH):
    """Loads the Hübner line4 standardised target curve's x-values (125 points)."""
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        return np.array([float(row['x']) for row in reader])


def _wall_bounds(dx, dy, angle_deg=_WALL_ANGLE_DEG, run_length=_WALL_RUN_LENGTH,
                  max_trim_frac=_MAX_TRIM_FRAC):
    """
    Finds (lo, hi) indices bounding the contiguous "wall" region: where the
    tangent stays within `angle_deg` of vertical for at least `run_length`
    consecutive points. Returns indices such that arr[lo:hi+1] is the wall.

    The search is capped at `max_trim_frac` of the profile per side. Without
    this, a vase that's wider than it is tall — about 13% of this generation
    space, checked directly, not a rare edge case — can have a tangent more
    horizontal than vertical across most of its *body*, not just its cap and
    foot; an uncapped walk-until-vertical search mistook one such vase's
    entire flared belly for "cap", trimming 70% of one side. 35% was chosen
    from the actual distribution, not guessed: across 200 real vases there's
    a clear elbow between cap=0.25 (56% of vases hit it — too tight, clipping
    normal shapes) and cap=0.35 (9% hit it — only the genuine pathological
    tail). A well-behaved vase gets its own adaptive trim under that cap
    (median ~19-21% per side); a pathological one hits the cap and stops,
    rather than consuming the profile.
    """
    n = len(dx)
    max_trim = max(1, int(n * max_trim_frac))
    with np.errstate(divide='ignore', invalid='ignore'):
        angle = np.degrees(np.arctan(np.abs(np.where(dy != 0, dx / dy, np.inf))))
    is_wall = angle < angle_deg

    def _first_run_start(seq, window):
        run = 0
        for i, v in enumerate(seq[:window + run_length]):
            run = run + 1 if v else 0
            if run >= run_length:
                return i - run_length + 1
        return window  # no stable wall found within the allowed window: trim the cap and stop

    lo = _first_run_start(is_wall, max_trim)
    hi = n - 1 - _first_run_start(is_wall[::-1], max_trim)
    if hi <= lo:
        lo, hi = 0, n - 1
    return lo, hi


def hubner_line4_srvf_distance(half, target_x):
    """
    SRV elastic shape distance between the vase's main side-wall profile and
    the Hübner line4 standardised target curve (the target needs no
    truncation of its own — see module-level note above).
    """
    import fdasrsf.curve_functions as curve_funcs

    x_v, y_v, dx, dy = half['x'], half['y'], half['dx'], half['dy']
    n_target = len(target_x)

    # The x=0 crossing at the cap/foot can land a point on either side, so
    # the half-profile is occasionally 124 or 126 points rather than exactly
    # 125 — resample onto the target's own point count (by fractional
    # position along the profile) so both curves always have a matching
    # length to pass to the SRVF framework.
    if len(x_v) != n_target:
        frac = np.linspace(0, 1, len(x_v))
        frac_target = np.linspace(0, 1, n_target)
        x_v = np.interp(frac_target, frac, x_v)
        y_v = np.interp(frac_target, frac, y_v)
        dx = np.gradient(x_v)
        dy = np.gradient(y_v)
        lo, hi = _wall_bounds(dx, dy)
    else:
        # Reuse the wall bounds already found in compute_half_profile rather
        # than recomputing them on an identical array.
        lo, hi = half['wall_lo'], half['wall_hi']

    curve_v = np.vstack([x_v[lo:hi + 1], y_v[lo:hi + 1]])
    curve_t = np.vstack([target_x[lo:hi + 1], y_v[lo:hi + 1]])

    try:
        elastic_dist, _ = curve_funcs.elastic_distance_curve(curve_v, curve_t)
        return float(elastic_dist)
    except Exception:
        # Fallback for degenerate/collapsed curves, matching the established
        # pipeline's own fallback behaviour.
        return 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_all_features(xy, pc_values=None, hubner_line4_x=None):
    """
    Computes every feature in vase_features_master.xlsx for a single vase.

    Parameters:
    - xy: (N, 2) array, the full closed, mirrored contour for one vase.
    - pc_values: optional array-like of 6 floats (PC1..PC6), the vase's
      projection onto the fitted real-world-pottery PCA basis. If omitted,
      'distance_from_pc_centroid' is left out of the result.
    - hubner_line4_x: optional pre-loaded Hübner line4 target x-values (125
      points, from load_hubner_line4()). Loaded fresh from disk if omitted —
      pass it explicitly when extracting features for many vases in a loop,
      to avoid re-reading the same small file every time.

    Note: "Armand curvature measure" is still marked "To be added" in the
    master spreadsheet and has no agreed definition yet, so it is not
    computed here.

    Returns a flat dict of {feature_name: value}.
    """
    base = compute_base_kinematics(xy)
    half = compute_half_profile(xy)
    if hubner_line4_x is None:
        hubner_line4_x = load_hubner_line4()

    features = {
        'height_to_width_ratio': height_width_ratio(base['height'], base['width']),
        'top_to_bottom_area_ratio': top_bottom_area_ratio(half, base['y_min'], base['y_max']),
        'lip_to_belly_ratio': lip_to_belly_ratio(half),
        'height': vase_height(base),
        'maximum_width': maximum_width(base),
        'critical_tipping_angle': critical_tipping_angle(half),
        'rotational_volume': rotational_volume(half),
        'total_bending_energy': total_bending_energy(half),
        'inflection_count': inflection_count(half),
        'contour_autocorrelation': contour_autocorrelation(half),
        'fractal_dimension': fractal_dimension(half),
        'position_of_maximum_width': position_of_maximum_width(half, base['y_min'], base['y_max']),
        'waist_ratio': waist_ratio(half),
        'convexity_ratio': convexity_ratio(xy, base),
        'rim_flare_angle': rim_flare_angle(half),
        'surface_to_volume_ratio': surface_to_volume_ratio(half),
        'birkhoff_measure': birkhoff_measure(half, base),
        'hubner_line4_srvf_distance': hubner_line4_srvf_distance(half, hubner_line4_x),
    }

    if pc_values is not None:
        features['distance_from_pc_centroid'] = pc_centroid_distance(pc_values)

    return features

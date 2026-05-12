
import logging
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend strictly
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from shapely.geometry import Polygon

# Load PCA Data — convert to numpy arrays at load time to avoid
# repeated DataFrame-to-ndarray conversion on every vase attempt
_BASE = "app/vase/pca_csv"

_eigenvectors_df = pd.read_csv(f"{_BASE}/pc_vals.csv", index_col=0)
eigenvectors = _eigenvectors_df.values  # (n_components, n_features)

means = pd.read_csv(f"{_BASE}/pc_val_means.csv")["pc_mean"].values

# Pre-compute PC ranges as a dict for O(1) lookup instead of DataFrame filtering per attempt
_ranges_df = pd.read_csv(f"{_BASE}/pc_val_ranges.csv")
pc_ranges = {row["PC"]: (row["min"], row["max"]) for _, row in _ranges_df.iterrows()}
del _eigenvectors_df, _ranges_df


def pc_processing(random_pc_dict, num_vase_points):
    """
    Validates a generated PC dictionary using strictly standardised Y-coordinates.
    Returns (True, DataFrame) if valid, or (False, None) if the resulting polygon
    is self-intersecting or out of bounds.
    """
    # Sort by key to guarantee consistent row order relative to the eigenvector matrix
    pc_names = sorted(random_pc_dict.keys())
    pc_arr = np.array([random_pc_dict[k] for k in pc_names])

    # Select only the eigenvector rows for the PCs being used
    row_idx = [int(pc[2:]) - 1 for pc in pc_names]  # 'PC1' -> 0, 'PC2' -> 1, ...
    original_shape = np.dot(pc_arr, eigenvectors[row_idx, :]) + means

    x_vals = original_shape[:num_vase_points]
    y_vals = original_shape[num_vase_points:2 * num_vase_points]

    polygon = Polygon(zip(x_vals, y_vals))
    if not polygon.is_valid:
        return False, None

    idx = np.arange(1, num_vase_points + 1)
    df = pd.DataFrame({"x": x_vals, "y": y_vals}, index=idx)
    df.index.name = "variable"

    return True, df


def generate_vase_shape(PCs, vases_per_gen, num_vase_points, gen_track=0):
    vases = []
    vases_dict = {}
    max_attempts = 500

    for i in range(vases_per_gen):
        attempts = 0
        valid_vase = False

        while not valid_vase and attempts < max_attempts:
            random_pc_dict = {}
            # Generate uniformly across the global PC bounds (O(1) dict lookup)
            for pc in PCs:
                min_val, max_val = pc_ranges[pc]
                random_pc_dict[pc] = np.random.uniform(min_val, max_val)

            is_valid, df = pc_processing(random_pc_dict, num_vase_points)

            if is_valid:
                valid_vase = True
                vases.append(df)
                vases_dict[f"vase_{i+1}"] = {"PCs": random_pc_dict}

            attempts += 1

        if not valid_vase:
            logging.critical(f"Max attempts reached for initial vase {i+1}. Data compromised.")

    gen_track += 1
    return vases, vases_dict, gen_track


def calculate_decaying_std(current_gen, total_gens=5, max_prop=0.25, min_prop=0.05):
    """Return the range proportion for a given generation (0-indexed from first mutation gen).
    Exponentially decays from max_prop (gen 2) to min_prop (final gen)."""
    if total_gens <= 2:
        return max_prop
    decay_rate = np.log(min_prop / max_prop) / (total_gens - 2)
    return max_prop * np.exp(decay_rate * current_gen)


def generate_similar_vases(vases_per_gen, num_vase_points, selected_vase_PCs, range_proportion, PCs, gen_track, include_previous_vase):
    vases = []
    vases_dict = {}
    max_attempts = 500

    for i in range(vases_per_gen):
        attempts = 0
        valid_vase = False

        while not valid_vase and attempts < max_attempts:
            random_pc_dict = {}

            if include_previous_vase and i == 0:
                random_pc_dict = selected_vase_PCs.copy()
            else:
                # Generate via normal distribution with per-PC SD proportional to each PC's range
                for pc in PCs:
                    old_val = selected_vase_PCs[pc]
                    pc_min, pc_max = pc_ranges[pc]
                    pc_std = max(range_proportion * (pc_max - pc_min), 0)
                    random_pc_dict[pc] = np.random.normal(old_val, pc_std)

            is_valid, df = pc_processing(random_pc_dict, num_vase_points)

            if is_valid:
                valid_vase = True
                vases.append(df)
                vases_dict[f"vase_{i+1}"] = {"PCs": random_pc_dict}

            attempts += 1

        if not valid_vase:
            logging.critical(f"Max attempts reached for mutated vase {i+1}. Lineage compromised.")

    gen_track += 1
    return vases, vases_dict, gen_track


def vase_image(vases, user_id="default"):
    """Render vases as base64-encoded PNG data URIs. No disk I/O."""
    vase_id_dict = {}

    # Instantiate Figure and Canvas ONCE per function call to prevent server thread blocking
    fig = Figure(figsize=(4, 4), facecolor='white', dpi=200)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    for i, df in enumerate(vases):
        vase_name = f"{user_id}_vase_{i+1}"

        try:
            ax.clear()

            if df is None or df.empty or 'x' not in df.columns or 'y' not in df.columns:
                raise ValueError(f"Invalid data structure for vase {i+1}")

            x_full = df["x"].values
            y_full = df["y"].values

            if np.any(np.isnan(x_full)) or np.any(np.isnan(y_full)):
                raise ValueError(f"NaN values detected in vase {i+1} coordinates")

            ax.plot(x_full, y_full, marker="o", markersize=1, linestyle="-", color="black")
            ax.fill(x_full, y_full, color="darkgrey", alpha=0.5)
            ax.set_aspect("equal")
            ax.axis('off')

            buf = BytesIO()
            canvas.print_png(buf)
            png_bytes = buf.getvalue()
            buf.close()

            if len(png_bytes) < 1000:
                raise ValueError(f"Output PNG too small for {vase_name}")

            b64 = base64.b64encode(png_bytes).decode('ascii')
            vase_id_dict[vase_name] = f"data:image/png;base64,{b64}"

        except Exception as e:
            logging.error(f"Rendering failed for {vase_name}: {str(e)}.")
            raise RuntimeError(f"Critical pipeline failure: Unrecoverable rendering error for {vase_name}")

    # Explicitly clear memory manually to avoid GC bottlenecks
    del ax, fig, canvas

    return vase_id_dict


def make_json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj


def get_standardised_xy(pc_dict, num_vase_points=250):
    """
    Deterministically reconstructs the standardised X and Y arrays for CSV export
    without applying destructive interpolation.
    """
    pc_names = sorted(pc_dict.keys())
    pc_arr = np.array([pc_dict[k] for k in pc_names])
    row_idx = [int(pc[2:]) - 1 for pc in pc_names]  # 'PC1' -> 0, 'PC2' -> 1, ...
    original_shape = np.dot(pc_arr, eigenvectors[row_idx, :]) + means

    x_vals = original_shape[:num_vase_points]
    y_vals = original_shape[num_vase_points:2 * num_vase_points]

    return x_vals.tolist(), y_vals.tolist()

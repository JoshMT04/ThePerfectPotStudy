"""
PCA-based vase selection experiment for LLM participants.

Mirrors the logic of the human web experiment (potgen.py / routes.py) but replaces
the human with an LLM judge. Vases are represented as 6-dimensional PC vectors;
only those values are stored (not the 250-point XY contour), keeping output tiny.
"""

import os
import re
import time
import uuid
import random
import logging
import base64
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
from shapely.geometry import Polygon

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from claude import claude
from deepseek import deepseek
from chatgpt import chatgpt

# ---------------------------------------------------------------------------
# PCA data — paths relative to this file so the script works from anywhere
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(_HERE, "..", "file_processing", "pca_csv")

_eigenvectors_df = pd.read_csv(os.path.join(_BASE, "pc_vals.csv"), index_col=0)
_eigenvectors = _eigenvectors_df.values  # (n_components, n_features)
_means = pd.read_csv(os.path.join(_BASE, "pc_val_means.csv"))["pc_mean"].values
_ranges_df = pd.read_csv(os.path.join(_BASE, "pc_val_ranges.csv"))
_pc_ranges = {row["PC"]: (row["min"], row["max"]) for _, row in _ranges_df.iterrows()}
del _eigenvectors_df, _ranges_df

PCs = [f"PC{i}" for i in range(1, 7)]
_NUM_VASE_POINTS = 250

# ---------------------------------------------------------------------------
# Vase generation (adapted from potgen.py)
# ---------------------------------------------------------------------------

def _pc_processing(pc_dict: dict) -> tuple[bool, pd.DataFrame | None]:
    """Reconstruct vase contour from PC values; validate polygon."""
    pc_names = sorted(pc_dict.keys())
    pc_arr = np.array([pc_dict[k] for k in pc_names])
    row_idx = [int(pc[2:]) - 1 for pc in pc_names]
    shape = np.dot(pc_arr, _eigenvectors[row_idx, :]) + _means

    x_vals = shape[:_NUM_VASE_POINTS]
    y_vals = shape[_NUM_VASE_POINTS:2 * _NUM_VASE_POINTS]

    if not Polygon(zip(x_vals, y_vals)).is_valid:
        return False, None

    idx = np.arange(1, _NUM_VASE_POINTS + 1)
    df = pd.DataFrame({"x": x_vals, "y": y_vals}, index=idx)
    df.index.name = "variable"
    return True, df


def _generate_initial_vases() -> tuple[list[pd.DataFrame], list[dict]]:
    """Sample 2 random vases uniformly from the PC ranges."""
    vases, pc_dicts = [], []
    for _ in range(2):
        for _ in range(500):
            pc_dict = {pc: np.random.uniform(*_pc_ranges[pc]) for pc in PCs}
            ok, df = _pc_processing(pc_dict)
            if ok:
                vases.append(df)
                pc_dicts.append(pc_dict)
                break
        else:
            logging.critical("Max attempts reached for initial vase.")
    return vases, pc_dicts


def _generate_mutated_vases(parent_pcs: dict, range_prop: float) -> tuple[list[pd.DataFrame], list[dict]]:
    """
    Return [parent_df, mutant_df] and their PC dicts.
    The parent is always retained at index 0 (display order is shuffled later).
    """
    ok, parent_df = _pc_processing(parent_pcs)
    if not ok:
        raise RuntimeError("Parent PC dict produces invalid polygon.")

    for _ in range(500):
        mutant_pcs = {
            pc: np.random.normal(parent_pcs[pc], range_prop * (_pc_ranges[pc][1] - _pc_ranges[pc][0]))
            for pc in PCs
        }
        ok, mutant_df = _pc_processing(mutant_pcs)
        if ok:
            return [parent_df, mutant_df], [parent_pcs, mutant_pcs]

    logging.critical("Max attempts reached for mutated vase.")
    return [parent_df, parent_df], [parent_pcs, parent_pcs]


def _calculate_decaying_std(current_gen: int, total_gens: int = 5,
                             max_prop: float = 0.25, min_prop: float = 0.05) -> float:
    """Exponentially decay mutation range from gen 2 onward."""
    if total_gens <= 2:
        return max_prop
    decay_rate = np.log(min_prop / max_prop) / (total_gens - 2)
    return max_prop * np.exp(decay_rate * current_gen)


def _render_vases(vases: list[pd.DataFrame], tag: str) -> dict[str, str]:
    """Render vase DataFrames to base64 PNG data URIs."""
    fig = Figure(figsize=(4, 4), facecolor='white', dpi=200)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    result = {}

    for i, df in enumerate(vases):
        ax.clear()
        ax.plot(df["x"].values, df["y"].values, marker="o", markersize=1, linestyle="-", color="black")
        ax.fill(df["x"].values, df["y"].values, color="darkgrey", alpha=0.5)
        ax.set_aspect("equal")
        ax.axis('off')

        buf = BytesIO()
        canvas.print_png(buf)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        buf.close()
        result[f"{tag}_vase_{i+1}"] = f"data:image/png;base64,{b64}"

    del ax, fig, canvas
    return result

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _get_llm(model: str):
    if "deepseek" in model:
        return deepseek
    if "gpt" in model:
        return chatgpt
    return claude


def _call_with_retry(fn, *args, max_retries: int = 5, delay: int = 5, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            logging.warning(f"API call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s…")
            time.sleep(delay)


def _parse_selection(text: str, max_val: int) -> int:
    """Extract integer from <selection> tags; fall back to first digit found."""
    match = re.search(r'<selection>\s*(\d+)\s*</selection>', text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 1 <= val <= max_val:
            return val
    fallback = re.search(fr'[1-{max_val}]', text)
    return int(fallback.group()) if fallback else 1

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def experiment_run(
    num_rounds: int,
    n_generations: int,
    model: str,
    budget_tokens: int,
    prompt: str,
    final_prompt: str,
) -> tuple[pd.DataFrame, str]:
    """
    Run the full experiment for one LLM 'participant'.

    Returns a single DataFrame (main rounds + final tournament combined via
    a 'phase' column) and the session UUID. One row per vase per generation —
    stores PC values only, not XY coordinates.
    """
    print(f"  Session start — model={model}, rounds={num_rounds}, gens={n_generations}")
    session_id = str(uuid.uuid4())
    session_started_at = datetime.now().isoformat()
    llm = _get_llm(model)
    rows = []

    # Winning PC dict from the final generation of each round, for the tournament
    round_winners_pcs = []

    for round_idx in range(num_rounds):
        selected_pcs = None

        for gen in range(1, n_generations + 1):
            if gen == 1:
                vases, pc_dicts = _generate_initial_vases()
            else:
                range_prop = _calculate_decaying_std(gen - 2, total_gens=n_generations)
                vases, pc_dicts = _generate_mutated_vases(selected_pcs, range_prop)

            # Shuffle display order to avoid position bias
            order = [0, 1]
            random.shuffle(order)
            vases_disp = [vases[i] for i in order]
            pcs_disp = [pc_dicts[i] for i in order]

            tag = f"r{round_idx+1}g{gen}"
            img_urls = {f"vase{i+1}": v for i, v in enumerate(_render_vases(vases_disp, tag).values())}

            api = _call_with_retry(llm, model, prompt, budget_tokens, img_urls)
            chosen_display = _parse_selection(api["response_text"], 2)

            # Map display choice back to the original (pre-shuffle) index
            chosen_original_idx = order[chosen_display - 1]
            selected_pcs = pc_dicts[chosen_original_idx]

            print(f"    Round {round_idx+1}, Gen {gen}: chose display vase {chosen_display} "
                  f"(original idx {chosen_original_idx}, parent={'yes' if chosen_original_idx == 0 and gen > 1 else 'no'})")

            for disp_idx in range(2):
                orig_idx = order[disp_idx]
                rows.append({
                    "phase": "main",
                    "session_id": session_id,
                    "session_started_at": session_started_at,
                    "model": model,
                    "round": round_idx + 1,
                    "gen": gen,
                    "display_position": disp_idx + 1,
                    "is_selected": (disp_idx + 1 == chosen_display),
                    "is_parent": (gen > 1 and orig_idx == 0),
                    **pcs_disp[disp_idx],
                    "rt_tokens": api["rt_tokens"],
                    "budget_tokens": api["budget_tokens"],
                    "budget_utilisation": api["budget_utilisation"],
                    "wall_time_s": api["wall_time_s"],
                    "input_tokens": api["input_tokens"],
                    "output_tokens": api["output_tokens"],
                    "stop_reason": api["stop_reason"],
                    "timestamp": api["timestamp"],
                })

        round_winners_pcs.append(selected_pcs)

    # --- Final tournament: all round winners presented simultaneously ---
    print(f"  Final tournament ({num_rounds} vases)…")
    t_order = list(range(num_rounds))
    random.shuffle(t_order)
    t_pcs = [round_winners_pcs[i] for i in t_order]

    t_vases = []
    for pc_dict in t_pcs:
        _, df = _pc_processing(pc_dict)
        t_vases.append(df)

    t_imgs = {f"vase{i+1}": v for i, v in enumerate(_render_vases(t_vases, "tournament").values())}
    api = _call_with_retry(llm, model, final_prompt, budget_tokens, t_imgs)
    chosen_final = _parse_selection(api["response_text"], num_rounds)
    winning_round = t_order[chosen_final - 1] + 1

    print(f"  Tournament: chose display vase {chosen_final} (from round {winning_round})")

    for disp_idx in range(num_rounds):
        rows.append({
            "phase": "tournament",
            "session_id": session_id,
            "session_started_at": session_started_at,
            "model": model,
            "round": t_order[disp_idx] + 1,  # original round lineage
            "gen": None,
            "display_position": disp_idx + 1,
            "is_selected": (disp_idx + 1 == chosen_final),
            "is_parent": None,
            **t_pcs[disp_idx],
            "rt_tokens": api["rt_tokens"],
            "budget_tokens": api["budget_tokens"],
            "budget_utilisation": api["budget_utilisation"],
            "wall_time_s": api["wall_time_s"],
            "input_tokens": api["input_tokens"],
            "output_tokens": api["output_tokens"],
            "stop_reason": api["stop_reason"],
            "timestamp": api["timestamp"],
        })

    return pd.DataFrame(rows), session_id

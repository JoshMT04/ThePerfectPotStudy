"""
Runner for the PCA-based LLM vase selection experiment.

Each participant produces ~55 rows (50 main + 5 tournament), storing only
6 PC values per vase — no XY coordinates. All participants for a given model
are collected into a single gzip-compressed CSV, so the entire output for
N participants across M models is just M files.

Usage (from the ai_beauty/ directory):
    python pca_llm_run.py
"""

import os
import pandas as pd
from pca_llm_experiment import experiment_run

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_ROUNDS = 5
N_GENERATIONS = 5
BUDGET_TOKENS = 5000

PROMPT = (
    "From these two images of vases, please select the one that is most beautiful. "
    "Please take time to think very carefully before responding. "
    "Return a single integer, either '1' or '2', strictly enclosed within <selection></selection> tags."
)
FINAL_PROMPT = (
    f"From these {NUM_ROUNDS} images of vases, please select the one that is most beautiful. "
    "Please take time to think very carefully before responding. "
    f"Return a single integer between 1 and {NUM_ROUNDS}, strictly enclosed within <selection></selection> tags."
)

# Number of simulated participants per model
N_PARTICIPANTS = 300

# Models to run — comment out any you don't need
MODELS = [
    # ("claude", "claude-sonnet-4-6"),
    # ("deepseek", "deepseek-reasoner"),
    ("gpt",     "gpt-5.4-nano-2026-03-17"),
]

RESULT_DIR = "ai_beauty/pca_llm_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for provider, model in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model}  ({N_PARTICIPANTS} participant(s))")
    print(f"{'='*60}")

    all_dfs = []

    for i in range(N_PARTICIPANTS):
        print(f"\nParticipant {i+1}/{N_PARTICIPANTS}")
        df, session_id = experiment_run(
            num_rounds=NUM_ROUNDS,
            n_generations=N_GENERATIONS,
            model=model,
            budget_tokens=BUDGET_TOKENS,
            prompt=PROMPT,
            final_prompt=FINAL_PROMPT,
        )
        all_dfs.append(df)
        print(f"  Done — session {session_id}, {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)

    base = os.path.join(RESULT_DIR, f"{provider}_results")
    out_path = f"{base}.csv.gz"
    version = 1
    while os.path.exists(out_path):
        out_path = f"{base}_v{version}.csv.gz"
        version += 1

    combined.to_csv(out_path, index=False, compression="gzip")
    print(f"\nSaved {len(combined)} rows → {out_path}")

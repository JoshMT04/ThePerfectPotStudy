from vae_potgen import generate_initial_vases, generate_mutated_vase, vase_image, calculate_decaying_std
from claude import claude
from deepseek import deepseek
from chatgpt import chatgpt
import re
import time
import uuid
import random
import pandas as pd
from datetime import datetime
import logging

def _call_with_retry(fn, *args, max_retries=5, delay=5, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            logging.warning(f"API call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)

num_rounds = 5
n_generations = 5

prompt = "From these two images of vases, please select the one that is most beautiful. Return a single integer, either '1' or '2', strictly enclosed within <selection></selection> tags."
final_prompt = f"From these {num_rounds} images of vases, please select the one that is most beautiful. Return a single integer between 1 and {num_rounds}, strictly enclosed within <selection></selection> tags."

budget_tokens = 5000

def _parse_xml_selection(response_text: str, max_val: int) -> int:
    """Robustly extracts the integer from XML tags to prevent chain-of-thought contamination."""
    match = re.search(r'<selection>\s*(\d+)\s*</selection>', response_text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 1 <= val <= max_val:
            return val
    fallback = re.search(fr'[1-{max_val}]', response_text)
    return int(fallback.group()) if fallback else 1


def experiment_run(num_rounds: int, n_generations: int,
                   model: str, budget_tokens: int, prompt: str, final_prompt: str,
                   folder_id: str = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:

    print(f"Initiating VAE experiment: {num_rounds} rounds, {n_generations} generations.")

    session_id = str(uuid.uuid4())
    session_started_at = datetime.now().isoformat()

    all_rows = []
    final_tournament_vases = []
    final_tournament_latents = []
    final_tournament_original_rounds = []

    for round_idx in range(num_rounds):
        selected_z = None
        selected_df = None

        for gen in range(1, n_generations + 1):

            if gen == 1:
                vases, latents = generate_initial_vases(num_vases=2)
            else:
                std = calculate_decaying_std(gen - 2, total_gens=n_generations)
                result = generate_mutated_vase(selected_z, std)
                if result is None:
                    logging.critical(f"Round {round_idx+1}, Gen {gen}: Failed to generate mutated vase.")
                    break
                new_df, new_z = result
                # Pair: index 0 = retained parent, index 1 = new mutant
                vases = [selected_df, new_df]
                latents = [selected_z, new_z]

            # Randomly shuffle which is vase 1 vs vase 2
            vase_order = [0, 1]
            random.shuffle(vase_order)

            vases_shuffled = [vases[i] for i in vase_order]
            latents_shuffled = [latents[i] for i in vase_order]

            vase_id_dict = vase_image(vases_shuffled, user_id=f"round{round_idx+1}_gen{gen}", folder_id=folder_id)

            keys = list(vase_id_dict.keys())
            img_urls = {
                "vase1": vase_id_dict[keys[0]],
                "vase2": vase_id_dict[keys[1]],
            }

            if "deepseek" in model:
                _call = deepseek
            elif "gpt" in model:
                _call = chatgpt
            else:
                _call = claude
            api_result = _call_with_retry(_call, model, prompt, budget_tokens, img_urls)
            selected = _parse_xml_selection(api_result["response_text"], 2)

            # Map display selection back to the original order
            selected_shuffled_idx = selected - 1
            selected_original_idx = vase_order[selected_shuffled_idx]

            selected_df = vases[selected_original_idx]
            selected_z = latents[selected_original_idx]

            print(f"Round {round_idx+1}, Gen {gen}: Selected vase {selected} (original index {selected_original_idx})")

            if gen == n_generations:
                final_tournament_vases.append(selected_df)
                final_tournament_latents.append(selected_z)
                final_tournament_original_rounds.append(round_idx + 1)

            generation_key = f"round_{round_idx+1}_gen_{gen}"

            for display_idx, df in enumerate(vases_shuffled):
                vase_num = display_idx + 1
                original_idx = vase_order[display_idx]
                is_selected = (vase_num == selected)
                is_parent = (gen > 1 and original_idx == 0)
                z_vals = latents_shuffled[display_idx]

                for _, row in df.iterrows():
                    row_dict = {
                        "generation_key": generation_key,
                        "round": round_idx + 1,
                        "gen": gen,
                        "vase": vase_num,
                        "x": row["x"],
                        "y": row["y"],
                        "is_selected_vase": is_selected,
                        "is_parent_vase": is_parent,
                        "rt_tokens": api_result["rt_tokens"],
                        "budget_tokens": api_result["budget_tokens"],
                        "budget_utilisation": api_result["budget_utilisation"],
                        "wall_time_s": api_result["wall_time_s"],
                        "input_tokens": api_result["input_tokens"],
                        "output_tokens": api_result["output_tokens"],
                        "stop_reason": api_result["stop_reason"],
                        "model": model,
                        "session_id": session_id,
                        "timestamp": api_result["timestamp"],
                    }
                    for zi in range(len(z_vals)):
                        row_dict[f"z_{zi}"] = z_vals[zi]
                    all_rows.append(row_dict)

    # --- FINAL TOURNAMENT PHASE ---
    print(f"\nExecuting final tournament across {num_rounds} rounds...")

    tournament_rows = []
    tournament_order = list(range(num_rounds))
    random.shuffle(tournament_order)
    vases_tournament_shuffled = [final_tournament_vases[i] for i in tournament_order]

    vase_id_dict = vase_image(vases_tournament_shuffled, user_id="final_tournament", folder_id=folder_id)
    keys = list(vase_id_dict.keys())
    img_urls = {f"vase{i+1}": vase_id_dict[keys[i]] for i in range(num_rounds)}

    if "deepseek" in model:
        _call = deepseek
    elif "gpt" in model:
        _call = chatgpt
    else:
        _call = claude
    api_result = _call_with_retry(_call, model, final_prompt, budget_tokens, img_urls)
    selected_final = _parse_xml_selection(api_result["response_text"], num_rounds)

    winning_vase_shuffled_idx = selected_final - 1
    winning_round = final_tournament_original_rounds[tournament_order[winning_vase_shuffled_idx]]

    print(f"Tournament Result: Selected Vase {selected_final} (Origin: round {winning_round})")

    latents_tournament_shuffled = [final_tournament_latents[i] for i in tournament_order]

    for display_idx, df in enumerate(vases_tournament_shuffled):
        vase_num = display_idx + 1
        original_round = final_tournament_original_rounds[tournament_order[display_idx]]
        is_selected = (vase_num == selected_final)
        z_vals = latents_tournament_shuffled[display_idx]

        for _, row in df.iterrows():
            row_dict = {
                "tournament_display_index": vase_num,
                "original_round_lineage": original_round,
                "x": row["x"],
                "y": row["y"],
                "is_tournament_winner": is_selected,
                "rt_tokens": api_result["rt_tokens"],
                "budget_tokens": api_result["budget_tokens"],
                "budget_utilisation": api_result["budget_utilisation"],
                "wall_time_s": api_result["wall_time_s"],
                "model": model,
                "session_id": session_id,
                "timestamp": api_result["timestamp"],
            }
            for zi in range(len(z_vals)):
                row_dict[f"z_{zi}"] = z_vals[zi]
            tournament_rows.append(row_dict)

    df_out = pd.DataFrame(all_rows)
    df_tournament = pd.DataFrame(tournament_rows)

    return df_out, df_tournament, session_id

'''
Master Execution Pipeline: main_extractor.py
This script ingests the flattened 62.5M row game dataset, isolates the 250-point mathematical 
half-profiles, and parallelises the extraction of all 164 intrinsic and relational features.
'''

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import os
import time
import re
from datetime import datetime

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# NumPy 2 compatibility: np.trapz was removed in favor of np.trapezoid.
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# -----------------------------------------
# 1. IMPORT INTRINSIC MODULES
# -----------------------------------------
from function_modules.core_engine import compute_base_kinematics
from function_modules.proportions_and_mass import extract_proportions_and_mass
from function_modules.kinematics import extract_kinematics
from function_modules.entropy_and_moments import extract_entropy_and_moments
from function_modules.vertical_asymmetry_and_binning import extract_vertical_asymmetry_and_binning
from function_modules.typological_skeleton import extract_typological_skeleton

# -----------------------------------------
# 2. IMPORT RELATIONAL MODULE
# -----------------------------------------
from function_modules.main_curve_matcher import compare_vase_to_target

# -----------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------
def load_standardised_targets(target_dir):
    '''
    Loads the 5 theoretically perfect S-curves into memory exactly once.
    '''
    targets = {}

    # Load whatever standardised target files are present instead of hard-coding line1..line5.
    # This supports partial target directories (e.g., only line4 and line5 available).
    try:
        target_files = [f for f in os.listdir(target_dir) if re.match(r'^line\d+_standardised\.csv$', f)]
    except FileNotFoundError:
        print(f"WARNING: Target directory not found: {target_dir}")
        return targets

    def _line_number(filename):
        match = re.search(r'line(\d+)_standardised\.csv$', filename)
        return int(match.group(1)) if match else float('inf')

    for file_name in sorted(target_files, key=_line_number):
        file_path = os.path.join(target_dir, file_name)
        df = pd.read_csv(file_path)
        if 'x' not in df.columns:
            print(f"WARNING: Target {file_path} has no 'x' column. Skipping.")
            continue

        line_num = _line_number(file_name)
        targets[f'SCurve_{line_num}'] = df['x'].values

    if not targets:
        print(f"WARNING: No target files found in {target_dir} matching line*_standardised.csv")

    return targets

def extract_stratified_prototype_subset(df_raw, user_fraction=0.05, random_seed=42):
    '''
    Isolates a mathematically sound subset of the dataset by sampling entire 
    participant lineages rather than isolated, orphaned vases.
    '''
    np.random.seed(random_seed)
    unique_users = df_raw['user_id'].unique()
    if len(unique_users) == 0:
        print("   [Prototyping Mode] No users found in input CSV.")
        return df_raw.copy()

    # Ensure at least one user is sampled when a non-zero fraction is requested.
    n_users = int(len(unique_users) * user_fraction)
    if user_fraction > 0 and n_users == 0:
        n_users = 1
    n_users = min(n_users, len(unique_users))
    
    sampled_users = np.random.choice(unique_users, size=n_users, replace=False)
    df_subset = df_raw[df_raw['user_id'].isin(sampled_users)].copy()
    
    print(f"   [Prototyping Mode] Stratified Sample Generated: {len(sampled_users)} users.")
    print(f"   [Prototyping Mode] Total Rows Preserved: {len(df_subset)} (approx. {len(df_subset)//250} vases).")
    
    return df_subset


def _choose_first_existing_column(df, candidate_columns):
    '''
    Returns the first matching column name from candidates that exists in df.
    '''
    for col in candidate_columns:
        if col in df.columns:
            return col
    return None


def _load_metadata_dataframe(metadata_dir):
    '''
    Loads and concatenates participant metadata CSV files if a directory is provided.
    '''
    if not metadata_dir:
        return None

    if not os.path.isdir(metadata_dir):
        print(f"WARNING: Metadata directory not found: {metadata_dir}. Continuing without metadata merge.")
        return None

    metadata_files = sorted(
        [
            os.path.join(metadata_dir, f)
            for f in os.listdir(metadata_dir)
            if f.lower().endswith('.csv')
        ]
    )

    if not metadata_files:
        print(f"WARNING: No metadata CSV files found in {metadata_dir}. Continuing without metadata merge.")
        return None

    metadata_frames = []
    for metadata_path in metadata_files:
        try:
            metadata_frames.append(pd.read_csv(metadata_path))
        except Exception as exc:
            print(f"WARNING: Failed to read metadata file {metadata_path}: {exc}")

    if not metadata_frames:
        print("WARNING: No readable metadata files. Continuing without metadata merge.")
        return None

    df_meta = pd.concat(metadata_frames, ignore_index=True)
    return df_meta.drop_duplicates()


def _normalise_xy_schema(df_raw, metadata_df=None):
    '''
    Normalises legacy and new XY schemas into the canonical columns expected by feature extraction.
    '''
    df = df_raw.copy()

    rename_map = {}
    generation_col = _choose_first_existing_column(df, ['generation', 'generation_key', 'gen'])
    user_col = _choose_first_existing_column(df, ['user_id', 'participant_id', 'session_id'])
    selected_col = _choose_first_existing_column(df, ['gen_selected_vase', 'is_selected_vase'])

    if generation_col and generation_col != 'generation':
        rename_map[generation_col] = 'generation'
    if user_col and user_col != 'user_id':
        rename_map[user_col] = 'user_id'
    if selected_col and selected_col != 'gen_selected_vase':
        rename_map[selected_col] = 'gen_selected_vase'

    if rename_map:
        df = df.rename(columns=rename_map)

    # Merge participant metadata when available.
    if metadata_df is not None and not metadata_df.empty:
        meta_work = metadata_df.copy()
        if 'participant_id' in meta_work.columns and 'user_id' not in meta_work.columns:
            meta_work = meta_work.rename(columns={'participant_id': 'user_id'})

        # Prefer joining on both user_id and session_id when available.
        join_candidates = ['user_id', 'session_id']
        join_keys = [key for key in join_candidates if key in df.columns and key in meta_work.columns]

        if join_keys:
            meta_cols = [c for c in meta_work.columns if c not in df.columns or c in join_keys]
            df = df.merge(
                meta_work[meta_cols].drop_duplicates(),
                on=join_keys,
                how='left',
                suffixes=('', '_meta'),
            )

    required_columns = ['user_id', 'generation', 'vase', 'x', 'y', 'gen_selected_vase']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "CRITICAL ERROR: Input XY schema missing required columns after normalization: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    return df

# -----------------------------------------
# ATOMIC EXECUTION UNIT
# -----------------------------------------
def process_single_vase(vase_group, targets_dict, passthrough_cols=None):
    '''
    Processes exactly one 250-point half-profile.
    '''
    # 1. Administrative Identifiers
    user_id = vase_group['user_id'].iloc[0]
    generation = vase_group['generation'].iloc[0]
    vase_id = vase_group['vase'].iloc[0]
    
    selected_vase = vase_group['gen_selected_vase'].iloc[0]
    if isinstance(selected_vase, (bool, np.bool_)):
        is_selected = int(selected_vase)
    elif isinstance(selected_vase, str) and selected_vase.strip().lower() in {'true', 'false'}:
        is_selected = int(selected_vase.strip().lower() == 'true')
    elif pd.isna(selected_vase):
        is_selected = 0
    else:
        is_selected = 1 if vase_id == selected_vase else 0
    
    # 2. Geometric Purification
    # Sort strictly by the Y-axis from bottom (base) to top (lip)
    vase_group = vase_group.sort_values(by='y', ascending=True)
    y_coords = vase_group['y'].values
    # Enforce Absolute Right-Hand Domain
    x_coords = np.abs(vase_group['x'].values) 
    
    # Validation constraint: ensure strict dimensional integrity
    if len(x_coords) != 250:
        return None
        
    # 3. Base Intrinsic Mathematics
    xy_array = np.column_stack((x_coords, y_coords))
    base_state = compute_base_kinematics(xy_array)
    
    # 4. Execute the 84-Dimensional Intrinsic Extraction
    features = {
        'user_id': user_id,
        'generation': generation,
        'vase': vase_id,
        'is_selected': is_selected
    }

    # Preserve metadata/context fields for each vase using the first row in the group.
    passthrough_cols = passthrough_cols or []
    for col in passthrough_cols:
        if col in features:
            continue
        if col in vase_group.columns:
            features[col] = vase_group[col].iloc[0]
    
    features.update(extract_proportions_and_mass(base_state))
    features.update(extract_kinematics(base_state))
    features.update(extract_entropy_and_moments(base_state))
    features.update(extract_vertical_asymmetry_and_binning(base_state))
    features.update(extract_typological_skeleton(base_state))
    
    # 5. Execute the Relational S-Curve Matching Matrix
    for target_name, target_x in targets_dict.items():
        relational_features = compare_vase_to_target(
            y_coords=y_coords, 
            vase_x=x_coords, 
            target_x=target_x, 
            target_name=target_name
        )
        features.update(relational_features)
        
    return features


def _build_vase_key(source_csv, user_id, generation, vase_id):
    '''
    Creates a stable deduplication key for a single vase profile.
    '''
    return (
        '' if source_csv is None else str(source_csv),
        str(user_id),
        str(generation),
        str(vase_id),
    )


def _load_processed_keys(output_csv_path, require_source_csv=False):
    '''
    Loads already processed vase keys from an existing output file.
    '''
    if not os.path.exists(output_csv_path):
        return set()

    try:
        preview = pd.read_csv(output_csv_path, nrows=1)
    except Exception as exc:
        print(f"WARNING: Could not inspect existing output {output_csv_path}: {exc}")
        return set()

    required_cols = {'user_id', 'generation', 'vase'}
    if not required_cols.issubset(set(preview.columns)):
        return set()

    usecols = ['user_id', 'generation', 'vase']
    has_source_csv = 'source_csv' in preview.columns
    if require_source_csv and not has_source_csv:
        print(
            f"WARNING: {output_csv_path} has no 'source_csv' column. "
            "Safe resume is unavailable for directory mode."
        )
        return set()

    if has_source_csv:
        usecols = ['source_csv'] + usecols

    try:
        existing_df = pd.read_csv(output_csv_path, usecols=usecols)
    except Exception as exc:
        print(f"WARNING: Could not read existing processed keys from {output_csv_path}: {exc}")
        return set()

    keys = set()
    if has_source_csv:
        for _, row in existing_df.iterrows():
            keys.add(_build_vase_key(row['source_csv'], row['user_id'], row['generation'], row['vase']))
    else:
        for _, row in existing_df.iterrows():
            keys.add(_build_vase_key(None, row['user_id'], row['generation'], row['vase']))

    return keys


def _append_checkpoint(df_rows, output_csv_path):
    '''
    Appends checkpoint rows to the output CSV.
    '''
    if df_rows.empty:
        return

    write_header = not os.path.exists(output_csv_path)
    df_rows.to_csv(output_csv_path, mode='a', header=write_header, index=False)


def _append_process_log(log_csv_path, event_type, details):
    '''
    Appends one process event row to the process log CSV.
    '''
    row = {
        'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds'),
        'event_type': event_type,
    }
    row.update(details)
    df_log = pd.DataFrame([row])
    write_header = not os.path.exists(log_csv_path)
    df_log.to_csv(log_csv_path, mode='a', header=write_header, index=False)


def _extract_features_from_dataframe(
    df_raw,
    targets_dict,
    n_jobs=-1,
    prototyping_mode=True,
    source_csv=None,
    already_processed_keys=None,
    checkpoint_output_path=None,
    checkpoint_interval_seconds=120,
    batch_size=50,
    process_log_path=None,
    use_tqdm=True,
    passthrough_cols=None,
):
    '''
    Shared extraction routine from an already-loaded raw dataframe.
    '''
    if already_processed_keys is None:
        already_processed_keys = set()

    if prototyping_mode:
        print("   >> PROTOTYPING MODE ENGAGED. Executing stratified cohort subsetting...")
        df_raw = extract_stratified_prototype_subset(df_raw, user_fraction=0.05)

    print("3. Partitioning dataset into individual vase profiles...")
    grouped = [group for _, group in df_raw.groupby(['user_id', 'generation', 'vase'])]
    total_vases = len(grouped)
    print(f"   Successfully isolated {total_vases} unique mathematical arrays.")

    pending_groups = []
    skipped_count = 0
    for group in grouped:
        key = _build_vase_key(
            source_csv,
            group['user_id'].iloc[0],
            group['generation'].iloc[0],
            group['vase'].iloc[0],
        )
        if key in already_processed_keys:
            skipped_count += 1
            continue
        pending_groups.append(group)

    if skipped_count > 0:
        print(f"   Resume mode: skipping {skipped_count} already processed vases.")

    if not pending_groups:
        print("   No pending vases left for this dataset.")
        return pd.DataFrame()

    print(f"4. Engaging parallel extraction across {n_jobs if n_jobs > 0 else 'ALL'} CPU cores...")
    batch_size = max(1, int(batch_size))
    checkpoint_interval_seconds = max(1, int(checkpoint_interval_seconds))
    last_checkpoint_time = time.time()
    buffered_rows = []
    collected_rows = []

    batch_starts = range(0, len(pending_groups), batch_size)
    if use_tqdm and tqdm is not None:
        batch_starts = tqdm(
            batch_starts,
            total=(len(pending_groups) + batch_size - 1) // batch_size,
            desc=f"Extracting batches ({source_csv if source_csv else 'single_csv'})",
            unit='batch',
        )

    for start_idx in batch_starts:
        batch_groups = pending_groups[start_idx:start_idx + batch_size]
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_single_vase)(group, targets_dict, passthrough_cols) for group in batch_groups
        )

        valid_results = [r for r in results if r is not None]
        if not valid_results:
            continue

        if source_csv is not None:
            for row in valid_results:
                row['source_csv'] = source_csv

        for row in valid_results:
            key = _build_vase_key(source_csv, row['user_id'], row['generation'], row['vase'])
            already_processed_keys.add(key)

        collected_rows.extend(valid_results)
        buffered_rows.extend(valid_results)

        now = time.time()
        if checkpoint_output_path and (now - last_checkpoint_time >= checkpoint_interval_seconds):
            checkpoint_df = pd.DataFrame(buffered_rows)
            _append_checkpoint(checkpoint_df, checkpoint_output_path)
            print(
                f"   Checkpoint saved: {len(checkpoint_df)} rows appended to "
                f"{checkpoint_output_path}."
            )
            if process_log_path:
                _append_process_log(
                    log_csv_path=process_log_path,
                    event_type='checkpoint',
                    details={
                        'source_csv': source_csv,
                        'rows_written': int(len(checkpoint_df)),
                        'rows_collected_total': int(len(collected_rows)),
                        'pending_groups_total': int(len(pending_groups)),
                    },
                )
            buffered_rows = []
            last_checkpoint_time = now

    print("5. Purging corrupted arrays and compiling master matrix...")
    if checkpoint_output_path and buffered_rows:
        checkpoint_df = pd.DataFrame(buffered_rows)
        _append_checkpoint(checkpoint_df, checkpoint_output_path)
        print(
            f"   Final checkpoint saved: {len(checkpoint_df)} rows appended to "
            f"{checkpoint_output_path}."
        )
        if process_log_path:
            _append_process_log(
                log_csv_path=process_log_path,
                event_type='checkpoint_final',
                details={
                    'source_csv': source_csv,
                    'rows_written': int(len(checkpoint_df)),
                    'rows_collected_total': int(len(collected_rows)),
                    'pending_groups_total': int(len(pending_groups)),
                },
            )

    return pd.DataFrame(collected_rows)

# -----------------------------------------
# MASTER ORCHESTRATOR
# -----------------------------------------
def execute_master_pipeline(
    raw_csv_path,
    target_dir,
    output_csv_path,
    metadata_dir=None,
    n_jobs=-1,
    prototyping_mode=True,
    resume=False,
    checkpoint_interval_seconds=120,
    batch_size=50,
    process_log_path=None,
    use_tqdm=False,
):
    '''
    Ingests the raw data and orchestrates the parallel extraction matrix.
    '''
    print(f"1. Loading raw dataset: {raw_csv_path}...")
    start_time = time.time()
    df_raw = pd.read_csv(raw_csv_path)
    metadata_df = _load_metadata_dataframe(metadata_dir)
    df_raw = _normalise_xy_schema(df_raw, metadata_df=metadata_df)
    passthrough_cols = list(metadata_df.columns) if metadata_df is not None else []
    if passthrough_cols:
        passthrough_cols = [
            ('user_id' if c == 'participant_id' else c)
            for c in passthrough_cols
        ]
        passthrough_cols = list(dict.fromkeys(passthrough_cols))

    already_processed_keys = set()
    if resume:
        already_processed_keys = _load_processed_keys(output_csv_path)
        print(f"   Resume mode: loaded {len(already_processed_keys)} processed vase keys.")

    if process_log_path:
        _append_process_log(
            log_csv_path=process_log_path,
            event_type='run_start_single',
            details={
                'raw_csv_path': raw_csv_path,
                'output_csv_path': output_csv_path,
                'resume': bool(resume),
                'checkpoint_interval_seconds': int(checkpoint_interval_seconds),
                'batch_size': int(batch_size),
            },
        )
    
    print("2. Loading Standardised Target S-Curves...")
    targets_dict = load_standardised_targets(target_dir)
    if not targets_dict:
        raise ValueError("CRITICAL ERROR: No target S-curves found. Pipeline halted.")

    df_final = _extract_features_from_dataframe(
        df_raw=df_raw,
        targets_dict=targets_dict,
        n_jobs=n_jobs,
        prototyping_mode=prototyping_mode,
        source_csv=os.path.basename(raw_csv_path),
        already_processed_keys=already_processed_keys,
        checkpoint_output_path=output_csv_path,
        checkpoint_interval_seconds=checkpoint_interval_seconds,
        batch_size=batch_size,
        process_log_path=process_log_path,
        use_tqdm=use_tqdm,
        passthrough_cols=passthrough_cols,
    )

    if not resume:
        print(f"6. Writing 164-dimensional master matrix to {output_csv_path}...")
        df_final.to_csv(output_csv_path, index=False)
    else:
        print(f"6. Resume mode complete. Output incrementally checkpointed to {output_csv_path}.")
    
    elapsed = (time.time() - start_time) / 60
    print(f"PIPELINE COMPLETE. Extracted {len(df_final)} vases in {elapsed:.2f} minutes.")

    if process_log_path:
        _append_process_log(
            log_csv_path=process_log_path,
            event_type='run_complete_single',
            details={
                'raw_csv_path': raw_csv_path,
                'vases_extracted_in_run': int(len(df_final)),
                'elapsed_minutes': round(float(elapsed), 4),
            },
        )


def execute_master_pipeline_for_directory(
    raw_csv_dir,
    target_dir,
    output_csv_path,
    metadata_dir=None,
    n_jobs=-1,
    prototyping_mode=False,
    resume=True,
    checkpoint_interval_seconds=120,
    batch_size=50,
    process_log_path=None,
    use_tqdm=False,
):
    '''
    Runs extraction across all CSV files in a directory and writes one combined master matrix.
    '''
    print(f"1. Scanning directory for raw datasets: {raw_csv_dir}...")
    start_time = time.time()

    if not os.path.isdir(raw_csv_dir):
        raise ValueError(f"CRITICAL ERROR: Raw CSV directory not found: {raw_csv_dir}")

    csv_files = sorted(
        [
            os.path.join(raw_csv_dir, f)
            for f in os.listdir(raw_csv_dir)
            if f.lower().endswith('.csv')
        ]
    )

    if not csv_files:
        raise ValueError(f"CRITICAL ERROR: No CSV files found in {raw_csv_dir}")

    if process_log_path is None:
        process_log_path = f"{output_csv_path}.process_log.csv"

    _append_process_log(
        log_csv_path=process_log_path,
        event_type='run_start_directory',
        details={
            'raw_csv_dir': raw_csv_dir,
            'output_csv_path': output_csv_path,
            'resume': bool(resume),
            'checkpoint_interval_seconds': int(checkpoint_interval_seconds),
            'batch_size': int(batch_size),
            'csv_file_count': int(len(csv_files)),
            'use_tqdm': bool(use_tqdm and tqdm is not None),
        },
    )

    print(f"   Found {len(csv_files)} CSV files to process.")
    print("2. Loading Standardised Target S-Curves...")
    targets_dict = load_standardised_targets(target_dir)
    if not targets_dict:
        raise ValueError("CRITICAL ERROR: No target S-curves found. Pipeline halted.")

    metadata_df = _load_metadata_dataframe(metadata_dir)
    passthrough_cols = list(metadata_df.columns) if metadata_df is not None else []
    if passthrough_cols:
        passthrough_cols = [
            ('user_id' if c == 'participant_id' else c)
            for c in passthrough_cols
        ]
        passthrough_cols = list(dict.fromkeys(passthrough_cols))

    already_processed_keys = set()
    if resume:
        if os.path.exists(output_csv_path):
            preview = pd.read_csv(output_csv_path, nrows=1)
            if 'source_csv' not in preview.columns:
                legacy_backup = f"{output_csv_path}.legacy_backup"
                os.replace(output_csv_path, legacy_backup)
                print(
                    "   Resume mode: existing output is legacy (no source_csv key). "
                    f"Moved to backup: {legacy_backup}"
                )

        already_processed_keys = _load_processed_keys(output_csv_path, require_source_csv=True)
        print(f"   Resume mode: loaded {len(already_processed_keys)} processed vase keys.")
    elif os.path.exists(output_csv_path):
        os.remove(output_csv_path)
        print(f"   Existing output removed: {output_csv_path}")

    combined_frames = []
    file_iter = enumerate(csv_files, start=1)
    if use_tqdm and tqdm is not None:
        file_iter = enumerate(tqdm(csv_files, desc='Processing CSV files', unit='file'), start=1)

    for idx, csv_path in file_iter:
        print(f"\n--- Processing file {idx}/{len(csv_files)}: {csv_path} ---")
        df_raw = pd.read_csv(csv_path)
        df_raw = _normalise_xy_schema(df_raw, metadata_df=metadata_df)

        _append_process_log(
            log_csv_path=process_log_path,
            event_type='file_start',
            details={
                'source_csv': os.path.basename(csv_path),
                'file_index': int(idx),
                'file_total': int(len(csv_files)),
            },
        )

        df_file = _extract_features_from_dataframe(
            df_raw=df_raw,
            targets_dict=targets_dict,
            n_jobs=n_jobs,
            prototyping_mode=prototyping_mode,
            source_csv=os.path.basename(csv_path),
            already_processed_keys=already_processed_keys,
            checkpoint_output_path=output_csv_path,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            batch_size=batch_size,
            process_log_path=process_log_path,
            use_tqdm=use_tqdm,
            passthrough_cols=passthrough_cols,
        )
        combined_frames.append(df_file)
        print(f"   File complete: extracted {len(df_file)} vases.")
        _append_process_log(
            log_csv_path=process_log_path,
            event_type='file_complete',
            details={
                'source_csv': os.path.basename(csv_path),
                'file_index': int(idx),
                'file_total': int(len(csv_files)),
                'vases_extracted_in_file_run': int(len(df_file)),
            },
        )

    if not os.path.exists(output_csv_path):
        # Ensures output file exists even if every vase was already processed.
        df_final = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
        print(f"\n6. Writing combined master matrix to {output_csv_path}...")
        df_final.to_csv(output_csv_path, index=False)

    final_row_count = 0
    try:
        final_row_count = len(pd.read_csv(output_csv_path))
    except Exception:
        pass

    elapsed = (time.time() - start_time) / 60
    print(
        f"PIPELINE COMPLETE. Processed {len(csv_files)} files and extracted "
        f"{final_row_count} cumulative vases in {elapsed:.2f} minutes."
    )
    _append_process_log(
        log_csv_path=process_log_path,
        event_type='run_complete_directory',
        details={
            'raw_csv_dir': raw_csv_dir,
            'csv_file_count': int(len(csv_files)),
            'cumulative_vases_in_output': int(final_row_count),
            'elapsed_minutes': round(float(elapsed), 4),
        },
    )


def execute_master_pipeline_from_dataframe(
    df,
    target_dir,
    output_csv_path,
    n_jobs=-1,
    prototyping_mode=False,
    resume=False,
    checkpoint_interval_seconds=120,
    batch_size=50,
    process_log_path=None,
    use_tqdm=False,
    passthrough_cols=None,
):
    '''
    Runs extraction directly from a pre-loaded DataFrame (e.g. from a master parquet file)
    instead of reading from CSV files on disk.

    Column mapping applied automatically before extraction:
      gen / generation_key  → generation
      session_id            → user_id  (when user_id / participant_id absent)
      is_selected_vase      → gen_selected_vase
    '''
    start_time = time.time()

    if process_log_path is None:
        process_log_path = f"{output_csv_path}.process_log.csv"

    _append_process_log(
        log_csv_path=process_log_path,
        event_type='run_start_dataframe',
        details={
            'output_csv_path': output_csv_path,
            'input_rows': int(len(df)),
            'resume': bool(resume),
            'checkpoint_interval_seconds': int(checkpoint_interval_seconds),
            'batch_size': int(batch_size),
        },
    )

    print(f"1. Normalising master dataframe schema ({len(df):,} rows)...")
    df_raw = _normalise_xy_schema(df)

    # Infer passthrough columns from the extra columns in the dataframe when not supplied.
    if passthrough_cols is None:
        core_cols = {'user_id', 'generation', 'vase', 'x', 'y', 'gen_selected_vase',
                     'gen', 'generation_key', 'session_id', 'participant_id', 'is_selected_vase'}
        passthrough_cols = [c for c in df_raw.columns if c not in core_cols]

    print("2. Loading Standardised Target S-Curves...")
    targets_dict = load_standardised_targets(target_dir)
    if not targets_dict:
        raise ValueError("CRITICAL ERROR: No target S-curves found. Pipeline halted.")

    already_processed_keys = set()
    if resume:
        already_processed_keys = _load_processed_keys(output_csv_path)
        print(f"   Resume mode: loaded {len(already_processed_keys)} processed vase keys.")
    elif os.path.exists(output_csv_path):
        os.remove(output_csv_path)
        print(f"   Existing output removed: {output_csv_path}")

    df_final = _extract_features_from_dataframe(
        df_raw=df_raw,
        targets_dict=targets_dict,
        n_jobs=n_jobs,
        prototyping_mode=prototyping_mode,
        source_csv=None,
        already_processed_keys=already_processed_keys,
        checkpoint_output_path=output_csv_path,
        checkpoint_interval_seconds=checkpoint_interval_seconds,
        batch_size=batch_size,
        process_log_path=process_log_path,
        use_tqdm=use_tqdm,
        passthrough_cols=passthrough_cols,
    )

    if not os.path.exists(output_csv_path):
        print(f"6. Writing master matrix to {output_csv_path}...")
        df_final.to_csv(output_csv_path, index=False)
    else:
        print(f"6. Output incrementally checkpointed to {output_csv_path}.")

    elapsed = (time.time() - start_time) / 60
    print(f"PIPELINE COMPLETE. Extracted {len(df_final)} vases in {elapsed:.2f} minutes.")

    _append_process_log(
        log_csv_path=process_log_path,
        event_type='run_complete_dataframe',
        details={
            'output_csv_path': output_csv_path,
            'vases_extracted': int(len(df_final)),
            'elapsed_minutes': round(float(elapsed), 4),
        },
    )
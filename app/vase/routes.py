from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash, current_app
from app.vase.potgen import generate_vase_shape, generate_similar_vases, vase_image, make_json_safe, get_standardised_xy, calculate_decaying_std
from app.vase.forms import ErrorReportForm
import numpy as np
import csv
from app.drive_api import upload_file
from io import StringIO
import os
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor
import json

# Create a Blueprint for the vase routes to be used in the init file
vaseBP = Blueprint('vase', __name__)

MaxGens = 5  # Set the number of generations for the experiment
gen_list = list(range(1, MaxGens + 1))  # Create a list with the number of generations (e.g. 1 - 10)
PCs = [f"PC{i}" for i in range(1, 7)]  # Create a list with PC1 - PC6
vases_per_gen = 2  # Set the number of vases to be selected from per generation
num_vase_points = 250  # Number of contour points in the full vase silhouette

num_reps = 5 # Number of repetitions for the experiment
include_previous_vase = True # Flag to include the previous vase in the next generation

# Bounded thread pool for background Drive uploads (shared across all workers via --preload)
_upload_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='drive-upload')

def _upload_async(file_content, filename, folder_id):
    """Submit a Drive upload to the thread pool with one retry on failure."""
    def _do_upload():
        for attempt in range(2):
            try:
                upload_file(file_content, filename, folder_id)
                return
            except Exception as e:
                if attempt == 0:
                    logging.warning(f"Upload attempt 1 failed for {filename}: {e} — retrying")
                else:
                    logging.error(f"Upload failed permanently for {filename}: {e}")
    _upload_pool.submit(_do_upload)

def _csv_scalar(value):
    """Serialize session values (especially lists) into stable CSV scalars."""
    if isinstance(value, (list, tuple, set)):
        return '|'.join(map(str, value))
    return value

def _get_cache():
    """Get the Flask-Caching instance from the current app."""
    return current_app.cache

@vaseBP.route('/prefetch_vases', methods=['POST'])
def prefetch_vases():
    """Pre-generate next-generation vases for all possible selections."""
    gen_track = session.get('gen_track', 0)

    if gen_track >= MaxGens:
        return jsonify(prefetched=False)

    pc_vases_dict = session.get('pc_vases_dict', {})
    user_id = session.get('user_id', '')
    cache = _get_cache()
    prefetch_data = {}

    for vase_key, vase_data in pc_vases_dict.items():
        selected_vase_PCs = vase_data['PCs']
        try:
            range_prop = calculate_decaying_std(gen_track - 1, total_gens=MaxGens)
            vases, new_pc_dict, new_gen_track = generate_similar_vases(
                vases_per_gen, num_vase_points, selected_vase_PCs,
                range_prop, PCs, gen_track, include_previous_vase
            )
            new_vase_id_dict = vase_image(vases, user_id)
            prefetch_data[vase_key] = {
                'vase_id_dict': make_json_safe(new_vase_id_dict),
                'pc_vases_dict': make_json_safe(new_pc_dict),
                'gen_track': new_gen_track
            }
        except Exception as e:
            logging.warning(f"Prefetch failed for {vase_key}: {e}")

    if prefetch_data:
        cache_key = f'prefetch:{user_id}:{gen_track}'
        cache.set(cache_key, json.dumps(prefetch_data), timeout=300)

    return jsonify(prefetched=bool(prefetch_data))

@vaseBP.route('/vase_selection')
def vase_selection():
    print("Starting vase selection...")

    if 'user_id' not in session:
        user_id = f"user-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
        session['user_id'] = user_id

    pc_global_dict = session.get('pc_global_dict', {})
    gen_track = session.get('gen_track', 0)
    play_counter = session.get('play_counter', 0)

    if gen_track == 0 and play_counter == 0:
        for i in range(1, num_reps + 1):
            session.pop(f'round_{i}_uploaded', None)

    # Detect page refresh: page was already rendered but no selection has been made since.
    # gen_track > 0 guards against treating a fresh round start as a refresh.
    if session.get('vase_page_rendered') and gen_track > 0 and session.get('vase_id_dict'):
        flash('Refreshing the page does not count as a selection — your progress has not changed. Please select a vase to continue.', 'warning')
        return render_template(
            'vase_selection.html',
            vase_id_dict=session['vase_id_dict'],
            current_gen=gen_track - 1,
            max_gens=MaxGens,
            include_previous_vase=include_previous_vase,
            play_counter=play_counter + 1
        )

    max_generation_attempts = 3
    vases = None
    pc_vases_dict = None

    for attempt in range(max_generation_attempts):
        try:
            vases, pc_vases_dict, gen_track = generate_vase_shape(
                PCs, vases_per_gen, num_vase_points, gen_track
            )
            if len(vases) == vases_per_gen:
                break
            else:
                logging.warning(f"Expected {vases_per_gen} vases, got {len(vases)} on attempt {attempt + 1}")
        except Exception as e:
            logging.error(f"Vase generation attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_generation_attempts - 1:
                raise RuntimeError(f"Failed to generate vases after {max_generation_attempts} attempts")

    user_id = session.get('user_id', 'anonymous')
    vase_id_dict = vase_image(vases, user_id)

    if len(vase_id_dict) != vases_per_gen:
        raise RuntimeError(f"Critical error: Expected {vases_per_gen} vase images, but only got {len(vase_id_dict)}")

    print(f"Successfully generated and validated {len(vase_id_dict)} vase images")

    session['vase_id_dict'] = make_json_safe(vase_id_dict)
    session['pc_vases_dict'] = make_json_safe(pc_vases_dict)
    session['gen_track'] = gen_track
    session['play_counter'] = play_counter
    session['start_time'] = time.time()

    if 'elapsed_times' not in session:
        session['elapsed_times'] = []

    session['pc_global_dict'] = make_json_safe(pc_global_dict)
    session['vase_page_rendered'] = True

    return render_template(
        'vase_selection.html',
        vase_id_dict=vase_id_dict,
        current_gen=gen_track-1,
        max_gens=MaxGens,
        include_previous_vase=include_previous_vase,
        play_counter=play_counter+1
    )

@vaseBP.route('/select_vase', methods=['POST'])
def select_vase():
    user_processing = session.get('processing_vase', False)
    if user_processing:
        # Clear the flag if it has been stuck for more than 15 seconds (e.g. from a failed session save)
        if time.time() - session.get('processing_since', 0) > 15:
            session['processing_vase'] = False
        else:
            return jsonify(error="A vase is already being processed. Please wait.")

    session['processing_vase'] = True
    session['processing_since'] = time.time()
    pc_global_dict = session.get('pc_global_dict', {})

    data = request.get_json()
    selected_vase = data.get('vaseName')
    decision_time_ms = data.get('decisionTime')
    adjusted_decision_time_ms = None

    gen_track = session.get('gen_track', 0)
    start_time = session.get('start_time', None)
    elapsed_time = None

    if decision_time_ms is not None:
        try:
            decision_time_ms = int(decision_time_ms)
            elapsed_time = decision_time_ms / 1000.0
        except (TypeError, ValueError):
            decision_time_ms = None

    if elapsed_time is None and start_time:
        elapsed_time = time.time() - start_time
        logging.info(f"Time taken for generation {gen_track}: {elapsed_time:.2f} seconds")
        elapsed_times = session.get('elapsed_times', [])
        elapsed_times.append(elapsed_time)
        session['elapsed_times'] = elapsed_times

    calibration = session.get('rt_calibration', {}) or {}
    frame_interval_ms = calibration.get('frame_interval_ms')
    if decision_time_ms is not None and frame_interval_ms is not None:
        try:
            onset_delay_ms = float(frame_interval_ms) / 2.0
            adjusted_decision_time_ms = max(0, int(round(decision_time_ms - onset_delay_ms)))
        except (TypeError, ValueError):
            adjusted_decision_time_ms = None

    vase_id_dict = session.get('vase_id_dict', {})
    pc_vases_dict = session.get('pc_vases_dict', {})

    user_id = session.get('user_id', '')
    if selected_vase.startswith(user_id + '_'):
        vase_key = selected_vase[len(user_id) + 1:]
    else:
        vase_key = selected_vase

    pc_gen_list = [pc_vases_dict, vase_key, elapsed_time, decision_time_ms, adjusted_decision_time_ms]

    play_counter = session.get('play_counter', 0)
    pc_global_dict[f"round_{play_counter}_gen_{gen_track}"] = pc_gen_list
    session['pc_global_dict'] = make_json_safe(pc_global_dict)

    if gen_track >= MaxGens:
        play_counter += 1
        session['play_counter'] = play_counter
        print(f"Final vase selected for play counter {play_counter}: {selected_vase}")

        if 'final_vases' not in session:
            session['final_vases'] = {}

        vase_data_uri = vase_id_dict.get(selected_vase, '')
        if vase_data_uri.startswith('data:image/png;base64,'):
            session['final_vases'][f'Round {play_counter}'] = vase_data_uri
            session['final_selected_vase_path'] = vase_data_uri
        else:
            logging.error(f"No valid base64 data URI for {selected_vase}")

        session['processing_vase'] = False
        return jsonify(finished=True, vase_id_dict=vase_id_dict, current_gen=gen_track-1, max_gens=MaxGens)

    cache = _get_cache()
    cache_key = f'prefetch:{user_id}:{gen_track}'
    cached_raw = cache.get(cache_key)
    prefetch_hit = False

    if cached_raw:
        try:
            prefetch_data = json.loads(cached_raw)
            if vase_key in prefetch_data:
                cached = prefetch_data[vase_key]
                vase_id_dict = cached['vase_id_dict']
                pc_vases_dict = cached['pc_vases_dict']
                gen_track = cached['gen_track']
                prefetch_hit = True
                logging.info(f"Prefetch cache hit for {vase_key} at gen {gen_track - 1}")
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Prefetch cache unusable: {e}")
        cache.delete(cache_key)

    if not prefetch_hit:
        selected_vase_PCs = pc_vases_dict.get(vase_key).get("PCs")
        max_generation_attempts = 3

        for attempt in range(max_generation_attempts):
            try:
                range_prop = calculate_decaying_std(gen_track - 1, total_gens=MaxGens)
                vases, pc_vases_dict, gen_track = generate_similar_vases(
                    vases_per_gen, num_vase_points, selected_vase_PCs, range_prop, PCs, gen_track, include_previous_vase
                )
                if len(vases) == vases_per_gen:
                    break
                else:
                    logging.warning(f"Expected {vases_per_gen} vases, got {len(vases)} on attempt {attempt + 1}")
            except Exception as e:
                logging.error(f"Similar vase generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_generation_attempts - 1:
                    session['processing_vase'] = False
                    return jsonify(error=f"Failed to generate new vases after {max_generation_attempts} attempts")

        user_id = session.get('user_id', 'anonymous')
        vase_id_dict = vase_image(vases, user_id)

        if len(vase_id_dict) != vases_per_gen:
            session['processing_vase'] = False
            return jsonify(error=f"Critical error: Expected {vases_per_gen} vase images, but only got {len(vase_id_dict)}")

    print(f"Successfully {'used prefetched' if prefetch_hit else 'generated'} {len(vase_id_dict)} vase images")

    session['vase_id_dict'] = make_json_safe(vase_id_dict)
    session['pc_vases_dict'] = make_json_safe(pc_vases_dict)
    session['gen_track'] = gen_track
    session['start_time'] = time.time()
    session['processing_vase'] = False
    session['vase_page_rendered'] = False

    return jsonify(finished=False, vase_id_dict=vase_id_dict, current_gen=gen_track-1, max_gens=MaxGens)

@vaseBP.route('/play_again')
def play_again():
    session['gen_track'] = 0
    play_counter = session.get('play_counter', 0)
    play_counter += 1
    session['play_counter'] = play_counter

    if play_counter > num_reps:
        return redirect(url_for('vase.experiment_finished'))

    return redirect(url_for('vase.vase_selection'))

@vaseBP.route('/experiment_finished')
def experiment_finished():
    session['gen_track'] = 0
    pc_global_dict = session.get('pc_global_dict', {})

    user_id = session.get('user_id', '')
    session_id = session.get('session_id', '')
    metadata_schema_version = session.get('metadata_schema_version', 1)
    session_started_at = session.get('session_started_at', '')
    session_duration_s = time.time() - session_started_at if session_started_at else None
    completion_status = 'completed'
    play_counter = session.get('play_counter', 0)

    session['completion_status'] = completion_status
    print(f"Experiment finished. Generation tracker reset to 0. Play counter: {play_counter}")

    final_vases = session.get('final_vases', {})
    if not isinstance(final_vases, dict):
        logging.error("final_vases is not a dictionary. Resetting to an empty dictionary.")
        final_vases = {}
        session['final_vases'] = final_vases

    upload_key = f'round_{play_counter}_uploaded'
    if session.get(upload_key, False):
        print(f"Round {play_counter} data already uploaded, skipping duplicate upload.")
        if play_counter < num_reps:
            return redirect(url_for('vase.vase_selection'))
        else:
            play_counter = 0
            session['play_counter'] = play_counter
            return render_template('experiment_finished.html', final_vases=final_vases)

    completed_round_index = max(play_counter - 1, 0)
    round_prefix = f"round_{completed_round_index}_"
    round_pc_data = {
        gen_key: gen_data
        for gen_key, gen_data in pc_global_dict.items()
        if gen_key.startswith(round_prefix)
    }

    if not round_pc_data:
        if 'user_id' not in session:
            return redirect(url_for('metadata.start_experiment'))
        logging.error(
            f"No generation rows found for {round_prefix}. "
            f"Available keys: {list(pc_global_dict.keys())}. "
            f"CSV export aborted for round {play_counter}."
        )
        session[upload_key] = True
        return redirect(url_for('vase.goodbye'))

    csv_buffer = StringIO()
    fieldnames = ['generation_key', 'vase', 'pc_name', 'pc_value', 'is_selected_vase',
                  'is_parent_vase', 'selection_elapsed_time_s', 'decision_time_ms',
                  'adjusted_selection_elapsed_time_s', 'adjusted_decision_time_ms',
                  'rt_onset_correction_ms',
                  'rt_calibration_status', 'rt_calibration_median_ms',
                  'rt_calibration_trials_n', 'participant_id', 'session_id', 'play_counter',
                  'metadata_schema_version', 'session_started_at', 'session_duration_s', 'completion_status']

    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    calibration = session.get('rt_calibration', {}) or {}
    rt_calibration_status = calibration.get('status', 'missing')
    rt_calibration_median_ms = calibration.get('median_rt_ms')
    rt_calibration_trials_n = calibration.get('trials_n')
    _frame_ms = calibration.get('frame_interval_ms')
    rt_onset_correction_ms = round(float(_frame_ms) / 2.0, 3) if _frame_ms is not None else None

    for gen, vases in round_pc_data.items():
        elapsed_time = vases[2]
        decision_time_ms = vases[3] if len(vases) > 3 else None
        adjusted_decision_time_ms = vases[4] if len(vases) > 4 else None
        adjusted_elapsed_time_s = adjusted_decision_time_ms / 1000.0 if adjusted_decision_time_ms is not None else None
        gen_num = int(gen.split('_gen_')[-1])
        is_first_gen = (gen_num == 1)
        for vase_name, vase_data in vases[0].items():
            for pc, value in vase_data['PCs'].items():
                writer.writerow({
                    'generation_key': gen,
                    'vase': vase_name,
                    'pc_name': pc,
                    'pc_value': value,
                    'is_selected_vase': vase_name == vases[1],
                    'is_parent_vase': (not is_first_gen) and vase_name == 'vase_1',
                    'selection_elapsed_time_s': elapsed_time,
                    'decision_time_ms': decision_time_ms,
                    'adjusted_selection_elapsed_time_s': adjusted_elapsed_time_s,
                    'adjusted_decision_time_ms': adjusted_decision_time_ms,
                    'rt_onset_correction_ms': rt_onset_correction_ms,
                    'rt_calibration_status': rt_calibration_status,
                    'rt_calibration_median_ms': rt_calibration_median_ms,
                    'rt_calibration_trials_n': rt_calibration_trials_n,
                    'participant_id': user_id,
                    'session_id': session_id,
                    'play_counter': play_counter,
                    'metadata_schema_version': metadata_schema_version,
                    'session_started_at': session_started_at,
                    'session_duration_s': session_duration_s,
                    'completion_status': completion_status
                })

    pc_folder_id = os.getenv('PC_DRIVE_FOLDER_ID')
    if pc_folder_id:
        _upload_async(csv_buffer.getvalue(), f'round_{play_counter}_{user_id}_pc.csv', pc_folder_id)
        print(f"PC CSV upload submitted for round {play_counter}")
    else:
        logging.error("PC_DRIVE_FOLDER_ID environment variable is not set.")

    # Reconstruct XY using the deterministic standardisation methodology
    csv_buffer_xy = StringIO()
    fieldnames_xy = ['generation_key', 'vase', 'x', 'y', 'is_selected_vase',
                     'is_parent_vase', 'selection_elapsed_time_s', 'decision_time_ms',
                     'adjusted_selection_elapsed_time_s', 'adjusted_decision_time_ms',
                     'rt_onset_correction_ms',
                     'rt_calibration_status', 'rt_calibration_median_ms',
                     'rt_calibration_trials_n', 'participant_id', 'session_id', 'play_counter',
                     'metadata_schema_version', 'session_started_at', 'session_duration_s', 'completion_status']

    writer_xy = csv.DictWriter(csv_buffer_xy, fieldnames=fieldnames_xy)
    writer_xy.writeheader()
    xy_rows_written = 0
    xy_vases_skipped = 0
    for gen, vases in round_pc_data.items():
        elapsed_time = vases[2]
        decision_time_ms = vases[3] if len(vases) > 3 else None
        adjusted_decision_time_ms = vases[4] if len(vases) > 4 else None
        adjusted_elapsed_time_s = adjusted_decision_time_ms / 1000.0 if adjusted_decision_time_ms is not None else None
        gen_num = int(gen.split('_gen_')[-1])
        is_first_gen = (gen_num == 1)
        for vase_name, vase_data in vases[0].items():
            try:
                x_vals, y_vals = get_standardised_xy(vase_data['PCs'])
            except Exception as e:
                logging.error(f"get_standardised_xy failed for {vase_name} in {gen} (user {user_id}): {e}")
                xy_vases_skipped += 1
                continue

            for x, y in zip(x_vals, y_vals):
                writer_xy.writerow({
                    'generation_key': gen,
                    'vase': vase_name,
                    'x': x,
                    'y': y,
                    'is_selected_vase': vase_name == vases[1],
                    'is_parent_vase': (not is_first_gen) and vase_name == 'vase_1',
                    'selection_elapsed_time_s': elapsed_time,
                    'decision_time_ms': decision_time_ms,
                    'adjusted_selection_elapsed_time_s': adjusted_elapsed_time_s,
                    'adjusted_decision_time_ms': adjusted_decision_time_ms,
                    'rt_onset_correction_ms': rt_onset_correction_ms,
                    'rt_calibration_status': rt_calibration_status,
                    'rt_calibration_median_ms': rt_calibration_median_ms,
                    'rt_calibration_trials_n': rt_calibration_trials_n,
                    'participant_id': user_id,
                    'session_id': session_id,
                    'play_counter': play_counter,
                    'metadata_schema_version': metadata_schema_version,
                    'session_started_at': session_started_at,
                    'session_duration_s': session_duration_s,
                    'completion_status': completion_status
                })
                xy_rows_written += 1

    if xy_vases_skipped:
        logging.warning(f"XY export for user {user_id} round {play_counter}: {xy_vases_skipped} vase(s) skipped due to errors, {xy_rows_written} rows written.")

    xy_folder_id = os.getenv('XY_DRIVE_FOLDER_ID')
    if xy_folder_id:
        if xy_rows_written > 0:
            _upload_async(csv_buffer_xy.getvalue(), f'round_{play_counter}_{user_id}_xy.csv', xy_folder_id)
            print(f"XY CSV upload submitted for round {play_counter} ({xy_rows_written} rows)")
        else:
            logging.error(f"XY CSV has no rows for user {user_id} round {play_counter} — upload skipped.")
    else:
        logging.error("XY_DRIVE_FOLDER_ID environment variable is not set.")

    session[upload_key] = True

    if play_counter < num_reps:
        return redirect(url_for('vase.vase_selection'))
    else:
        play_counter = 0
        session['play_counter'] = play_counter
        return render_template('experiment_finished.html', final_vases=final_vases)

@vaseBP.route('/final_choice')
def final_choice():
    final_vases = session.get('final_vases', {})
    print(f"Final vases available for selection: {final_vases}")
    session['start_time'] = time.time()
    return render_template('experiment_finished.html', final_vases=final_vases)

@vaseBP.route('/final_vase_selection', methods=['POST'])
def final_vase_selection():
    data = request.get_json()
    selected_round = data.get('vaseName')
    if selected_round:
        final_vases = session.get('final_vases', {})
        selected_vase_url = final_vases.get(selected_round)

        if not selected_vase_url:
            return jsonify(success=False, error="Selected vase not found")

        logging.info(f"Final vase selected from {selected_round}: {selected_vase_url}")
        session['final_selected_vase'] = selected_round
        session['final_selected_vase_url'] = selected_vase_url
        user_id = session.get('user_id', '')
        session_id = session.get('session_id', '')
        metadata_schema_version = session.get('metadata_schema_version', 1)
        session_started_at = session.get('session_started_at', '')
        session_duration_s = time.time() - session_started_at if session_started_at else None
        completion_status = session.get('completion_status', 'in_progress')

        start_time = session.get('start_time', None)
        elapsed_time = None
        if start_time:
            elapsed_time = time.time() - start_time
            logging.info(f"Elapsed time for final vase selection: {elapsed_time:.2f} seconds")

        csv_buffer_final_choice = StringIO()
        fieldnames_final = ['selected_round', 'participant_id', 'session_id', 'selection_elapsed_time_s',
                    'metadata_schema_version', 'completion_status', 'session_duration_s']
        writer = csv.DictWriter(csv_buffer_final_choice, fieldnames=fieldnames_final)
        writer.writeheader()
        writer.writerow({
            'selected_round': selected_round,
            'participant_id': user_id,
            'session_id': session_id,
            'selection_elapsed_time_s': elapsed_time,
            'metadata_schema_version': metadata_schema_version,
            'completion_status': completion_status,
            'session_duration_s': session_duration_s
        })

        final_choice_folder_id = os.getenv('FINAL_DRIVE_FOLDER_ID')
        if final_choice_folder_id:
            _upload_async(csv_buffer_final_choice.getvalue(), f'final_choice_{user_id}.csv', final_choice_folder_id)
            print(f"Final choice CSV upload submitted for {user_id}")
        else:
            logging.error("FINAL_DRIVE_FOLDER_ID environment variable is not set.")

        return jsonify(success=True)
    return jsonify(success=False, error="No vase selected.")

@vaseBP.route('/goodbye', methods=['GET', 'POST'])
def goodbye():
    form = ErrorReportForm()

    if session.get('completion_status') != 'completed':
        session['completion_status'] = 'dropped'

    final_selected_vase = session.get('final_selected_vase', None)
    final_vase_image = session.get('final_selected_vase_url', None)

    if form.validate_on_submit():
        user_id = session.get('user_id', '')

        # Always save the goodbye form response to CSV
        csv_buffer_error = StringIO()
        fieldnames_error = ['user_id', 'hypothesis_response', 'hypothesis_text', 'reported_issue', 'issue_type', 'other_issue_description', 'timestamp']
        writer = csv.DictWriter(csv_buffer_error, fieldnames=fieldnames_error)
        writer.writeheader()
        writer.writerow({
            'user_id': user_id,
            'hypothesis_response': form.hypothesis.data,
            'hypothesis_text': form.hyp_text.data,
            'reported_issue': form.report.data,
            'issue_type': form.issue.data,
            'other_issue_description': form.other_issue.data,
            'timestamp': time.time()
        })

        error_folder_id = os.getenv('ERROR_DRIVE_FOLDER_ID')
        if error_folder_id:
            _upload_async(csv_buffer_error.getvalue(), f'error_report_{user_id}_{int(time.time())}.csv', error_folder_id)
            logging.info(f"Goodbye form submitted by user {user_id}: hypothesis={form.hypothesis.data}, issue={form.issue.data}")
        else:
            logging.error("ERROR_DRIVE_FOLDER_ID environment variable is not set.")

        flash('Thank you for your feedback!', 'info')
        session['goodbye_submitted'] = True

        return redirect(url_for('vase.goodbye'))

    goodbye_submitted = session.get('goodbye_submitted', False)
    return render_template('goodbye.html', final_vase_image=final_vase_image, form=form, goodbye_submitted=goodbye_submitted)

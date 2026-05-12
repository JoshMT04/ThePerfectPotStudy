from flask import Blueprint, render_template, session, redirect, url_for, request, jsonify
from app.metadata.forms import UserInfoForm
from app.drive_api import upload_file
import uuid
import time
import csv
import os
import logging
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

_metadata_upload_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='metadata-upload')

metadata = Blueprint('metadata', __name__) # Create a Blueprint object for the main routes to be accessed in the __init__.py file when building the web page


def _csv_scalar(value):
    if isinstance(value, (list, tuple, set)):
        return '|'.join(map(str, value))
    return value


def _upload_participants_row(session_obj):
    """Upload one participants row so static metadata is not duplicated in large event files."""
    participants_folder_id = os.getenv('PARTICIPANTS_DRIVE_FOLDER_ID')
    if not participants_folder_id:
        logging.warning("PARTICIPANTS_DRIVE_FOLDER_ID environment variable is not set. Skipping participants upload.")
        return

    csv_buffer = StringIO()
    fieldnames = [
        'participant_id', 'session_id', 'metadata_schema_version',
        'age_years', 'gender_identity', 'sexual_orientation',
        'current_residence_region_code', 'upbringing_region_code',
        'disability_identity_codes', 'pottery_experience_level',
        'browser_language', 'timezone', 'screen_width_px', 'screen_height_px',
        'device_type', 'user_agent', 'consent_version', 'consent_timestamp',
        'metadata_submitted_at', 'session_started_at',
        'rt_calibration_status', 'rt_calibration_trials_n',
        'rt_calibration_median_ms', 'rt_calibration_mean_ms',
        'rt_calibration_mad_ms', 'rt_calibration_frame_interval_ms',
        'rt_calibration_timer_resolution_ms', 'rt_calibration_completed_at'
    ]

    calibration = session_obj.get('rt_calibration', {}) or {}

    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        'participant_id': session_obj.get('user_id', ''),
        'session_id': session_obj.get('session_id', ''),
        'metadata_schema_version': session_obj.get('metadata_schema_version', 1),
        'age_years': session_obj.get('age_years', ''),
        'gender_identity': session_obj.get('gender_identity', ''),
        'sexual_orientation': session_obj.get('sexual_orientation', ''),
        'current_residence_region_code': _csv_scalar(session_obj.get('current_residence_region_code', '')),
        'upbringing_region_code': _csv_scalar(session_obj.get('upbringing_region_code', '')),
        'disability_identity_codes': _csv_scalar(session_obj.get('disability_identity_codes', '')),
        'pottery_experience_level': session_obj.get('pottery_experience_level', ''),
        'browser_language': session_obj.get('browser_language', ''),
        'timezone': session_obj.get('timezone', ''),
        'screen_width_px': session_obj.get('screen_width', ''),
        'screen_height_px': session_obj.get('screen_height', ''),
        'device_type': session_obj.get('device_type', ''),
        'user_agent': session_obj.get('user_agent', ''),
        'consent_version': session_obj.get('consent_version', 1),
        'consent_timestamp': session_obj.get('consent_timestamp', ''),
        'metadata_submitted_at': session_obj.get('metadata_submitted_at', ''),
        'session_started_at': session_obj.get('session_started_at', ''),
        'rt_calibration_status': calibration.get('status', 'pending'),
        'rt_calibration_trials_n': calibration.get('trials_n', ''),
        'rt_calibration_median_ms': calibration.get('median_rt_ms', ''),
        'rt_calibration_mean_ms': calibration.get('mean_rt_ms', ''),
        'rt_calibration_mad_ms': calibration.get('mad_rt_ms', ''),
        'rt_calibration_frame_interval_ms': calibration.get('frame_interval_ms', ''),
        'rt_calibration_timer_resolution_ms': calibration.get('timer_resolution_ms', ''),
        'rt_calibration_completed_at': calibration.get('completed_at', '')
    })

    participant_id = session_obj.get('user_id', 'unknown')
    session_id = session_obj.get('session_id', 'unknown')
    filename = f'participant_{participant_id}_{session_id}.csv'
    upload_file(csv_buffer.getvalue(), filename, participants_folder_id)


def _detect_device_type(user_agent_string):
    ua = (user_agent_string or '').lower()
    if 'ipad' in ua or 'tablet' in ua:
        return 'tablet'
    if 'mobi' in ua or 'iphone' in ua or 'android' in ua:
        return 'mobile'
    return 'desktop'


def _finalize_participant_metadata_upload():
    """Upload participants metadata once per participant after calibration state is known."""
    if session.get('participants_uploaded', False):
        return

    session_snapshot = dict(session)

    def _do_upload():
        try:
            _upload_participants_row(session_snapshot)
        except Exception as e:
            logging.error(f"Participants CSV upload failed: {e}")

    _metadata_upload_pool.submit(_do_upload)
    session['participants_uploaded'] = True

@metadata.route("/start_experiment", methods=['GET', 'POST'])
def start_experiment():
    form = UserInfoForm() # Retrieve the form from the forms.py file to be displayed on the home page
    # Ensure form fields are empty initially
    if request.method == 'GET': # Checks if the from is being requested, if true the form fields are empty
        form.age.data = ''
        form.gender.data = ''
        form.current_residence_region.data = ''
        form.upbringing_region.data = ''
        form.sexuality.data = ''
        form.disability.data = []
        form.consent.data = ''
        form.potter.data = ''
        
    if form.validate_on_submit(): # If the form is submitted by pressing the submit button, store the user's input in the session object
        
        session['user_id'] = str(uuid.uuid4())  # Generate and store a unique user ID to be retrieved in the vase\routes.py file
        session['session_id'] = str(uuid.uuid4())
        session['metadata_schema_version'] = 1
        session['metadata_submitted_at'] = time.time()
        session['consent_version'] = 1
        session['consent_timestamp'] = time.time()
        session['session_started_at'] = time.time()
        session['completion_status'] = 'in_progress'
        session['participants_uploaded'] = False
        session['rt_calibration'] = {'status': 'pending'}

        # Store the user's age inputted into the form so it can be retrieved in the vase\routes.py file
        session['age'] = form.age.data
        session['gender'] = form.gender.data
        session['sexuality'] = form.sexuality.data
        session['disability'] = form.disability.data
        session['potter'] = form.potter.data

        # Preferred normalized keys for analysis/export.
        session['age_years'] = form.age.data
        session['gender_identity'] = form.gender.data
        session['sexual_orientation'] = form.sexuality.data
        session['current_residence_region_code'] = form.current_residence_region.data
        session['upbringing_region_code'] = form.upbringing_region.data
        session['disability_identity_codes'] = form.disability.data
        session['pottery_experience_level'] = form.potter.data

        # Store client/session context for downstream analysis.
        session['browser_language'] = form.browser_language.data or request.accept_languages.best or ''
        session['timezone'] = form.timezone.data or ''
        session['screen_width'] = form.screen_width.data or ''
        session['screen_height'] = form.screen_height.data or ''
        session['device_type'] = form.device_type.data or _detect_device_type(request.headers.get('User-Agent', ''))
        session['user_agent'] = request.headers.get('User-Agent', '')
        
        print('Metadata submitted.')
        
        return redirect(url_for('metadata.reaction_calibration'))
    
    return render_template('meta_data.html', form=form) # Render the home page with the form using an html template


@metadata.route('/reaction_calibration', methods=['GET'])
def reaction_calibration():
    if 'user_id' not in session:
        return redirect(url_for('metadata.start_experiment'))
    return render_template('reaction_calibration.html')


@metadata.route('/save_reaction_calibration', methods=['POST'])
def save_reaction_calibration():
    if 'user_id' not in session:
        return jsonify(success=False, error='Session expired. Please restart the experiment.'), 400

    payload = request.get_json(silent=True) or {}
    trials = payload.get('trials_ms', [])

    status = 'completed'
    if not isinstance(trials, list):
        trials = []
        status = 'invalid_payload'

    cleaned_trials = []
    for value in trials:
        try:
            trial_ms = int(value)
            if trial_ms >= 0:
                cleaned_trials.append(trial_ms)
        except (TypeError, ValueError):
            continue

    if len(cleaned_trials) < 3:
        status = 'insufficient_trials'

    session['rt_calibration'] = {
        'status': status,
        'trials_n': len(cleaned_trials),
        'median_rt_ms': payload.get('median_rt_ms'),
        'mean_rt_ms': payload.get('mean_rt_ms'),
        'mad_rt_ms': payload.get('mad_rt_ms'),
        'frame_interval_ms': payload.get('frame_interval_ms'),
        'timer_resolution_ms': payload.get('timer_resolution_ms'),
        'hardware_concurrency': payload.get('hardware_concurrency'),
        'device_memory_gb': payload.get('device_memory_gb'),
        'platform': payload.get('platform'),
        'calibration_version': payload.get('calibration_version', 'rt-cal-v1'),
        'completed_at': time.time(),
        'trials_ms': cleaned_trials
    }

    _finalize_participant_metadata_upload()
    return jsonify(success=True, next_url=url_for('vase.vase_selection'))


@metadata.route('/skip_reaction_calibration', methods=['POST'])
def skip_reaction_calibration():
    if 'user_id' not in session:
        return jsonify(success=False, error='Session expired. Please restart the experiment.'), 400

    session['rt_calibration'] = {
        'status': 'skipped',
        'trials_n': 0,
        'completed_at': time.time(),
        'calibration_version': 'rt-cal-v1'
    }
    _finalize_participant_metadata_upload()
    return jsonify(success=True, next_url=url_for('vase.vase_selection'))
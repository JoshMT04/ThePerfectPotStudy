
from google.oauth2.service_account import Credentials
from google.auth.transport.httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from io import BytesIO
import httplib2
import logging
import os
import json
import threading

# Define the scope for Google Drive API allowing to create files
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Timeout in seconds for Drive API requests (default httplib2 is 60s — too short for large XY files)
_DRIVE_TIMEOUT_S = 300

# Cached Drive service per thread (googleapiclient is not thread-safe)
_thread_local = threading.local()

def _get_service():
    """Return a cached Drive API service for the current thread."""
    if not hasattr(_thread_local, 'service'):
        service_account_info = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        if service_account_info:
            creds = Credentials.from_service_account_info(
                json.loads(service_account_info), scopes=SCOPES)
        else:
            creds = Credentials.from_service_account_file(
                r'app/service_account.json', scopes=SCOPES)
        authorized_http = AuthorizedHttp(creds, http=httplib2.Http(timeout=_DRIVE_TIMEOUT_S))
        _thread_local.service = build('drive', 'v3', http=authorized_http)
    return _thread_local.service

def upload_file(file_content, filename, folder_id):
    """Uploads a file to Google Drive using a cached service client."""
    try:
        service = _get_service()
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        media = MediaIoBaseUpload(BytesIO(file_content.encode('utf-8')), mimetype='text/csv')
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"File ID: {file.get('id')}")
        logging.info(f"File {filename} uploaded to folder {folder_id} with ID: {file.get('id')}")
    except Exception as e:
        logging.error(f"Failed to upload file {filename} to Google Drive: {e}")
        raise
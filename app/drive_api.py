

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from io import BytesIO
import logging
import os
import json
import threading

# Define the scope for Google Drive API allowing to create files
SCOPES = ['https://www.googleapis.com/auth/drive.file']

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
        _thread_local.service = build('drive', 'v3', credentials=creds)
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
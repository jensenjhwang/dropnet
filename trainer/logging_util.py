from pprint import pprint
from googleapiclient import discovery
import os
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import numpy as np
from tensorflow.python.lib.io import file_io

# TODO: Change placeholder below to generate authentication credentials. See
# https://developers.google.com/sheets/quickstart/python#step_3_set_up_the_sample
#
# Authorize using one of the following scopes:
#     'https://www.googleapis.com/auth/drive'
#     'https://www.googleapis.com/auth/drive.file'
#     'https://www.googleapis.com/auth/spreadsheets'

def append_spread_sheet(spreadsheet_id, data_list, dir, cloud_mode):

    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    creds = None
    token_path = os.path.join(dir, 'token.pickle')
    cred_path = os.path.join(dir, 'credentials.json')

    if cloud_mode:
        with file_io.FileIO(token_path, mode='rb') as token:
            creds = pickle.load(token)
    else:
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    cred_path, SCOPES)
                creds = flow.run_local_server()
            # Save the credentials for the next run
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)

    service = discovery.build('sheets', 'v4', credentials=creds)

    # The A1 notation of a range to search for a logical table of data.
    # Values will be appended after the last row of the table.
    range_ = 'A:XX'

    # How the input data should be interpreted.
    value_input_option = 'RAW'

    # How the input data should be inserted.
    insert_data_option = 'INSERT_ROWS'

    data_list = [np.asarray(data) if isinstance(data, (list,)) else data for data in data_list]
    data_list = [np.array2string(data, separator=',') if isinstance(data, np.ndarray) else data for data in data_list]
    data_list = [np.float64(data) if isinstance(data, np.float32) else data for data in data_list]

    body = {'values': [data_list]}

    request = service.spreadsheets().values().append(spreadsheetId=spreadsheet_id, range=range_, valueInputOption=value_input_option, insertDataOption=insert_data_option, body=body)

    # request = service.spreadsheets().values().append(spreadsheetId=spreadsheet_id, range=range_, valueInputOption=value_input_option, insertDataOption=insert_data_option, body=value_range_body)
    response = request.execute()

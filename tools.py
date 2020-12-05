import mne
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import requests
from os import path
import json


from mne import Epochs, find_events, create_info, concatenate_epochs
from mne.io import RawArray


def _json_to_df(jsonData):
	index = []
	values = []
	for sample in jsonData["samples"]:
			index.append(sample["timestamp"])
			values.append(sample["data"])

	df = pd.DataFrame(values, index=index, columns=channels_names)

	return df

def get_session_df(session_data):
	global samplingRate, channels_names
	jsonData = session_data
	try:
		channels_names = jsonData['channelNames']
		samplingRate = jsonData['samplingRate']
	except:
		print("No channels_names in the data")
        
	return _json_to_df(jsonData)

def get_markers(session_data):
	jsonData = session_data
	return jsonData["markers"]

def get_session_split_and_grouped_by_markers(sessionData, markers_labels):
	rawdataobject = sessionData
	df = _json_to_df(rawdataobject)
	markers = rawdataobject["markers"] # TODO only choose from markers_labels


	# Split dataframe and collect splitted pieces
	dframes = []
	i = 0
	for marker in markers:
		if i > 0:
			small_df = df.loc[prevmarker["timestamp"]:marker["timestamp"]]
			small_df.label = prevmarker["label"]
			dframes.append(small_df)
		prevmarker = marker
		i+=1

	# Initialize result data format
	result = {}
	for electrode in df.columns:
		result[electrode] = {}
		for group in markers_labels:
			result[electrode][group] = []

	# Add small dataframes to their good place in results
	for experience_dframe in dframes:
		label = experience_dframe.label
		for (columnName, columnData) in experience_dframe.iteritems():
			result[columnName][label].append(columnData)

	return result

def get_session_erp_epochs(session_datas, markers_const, tmin=-0.1, tmax=0.8):
	event_id = {}
	counter = 1
	for marker in markers_const:
		event_id[marker] = counter
		counter += 1
	epochs_list = []  
	for session_data in session_datas :
		df = get_session_df(session_data)
		channels = df.columns.tolist()
		n_channel = len(channels)

		info = create_info(ch_names=channels + ["stim"], ch_types=["eeg"] * n_channel + ["stim"], sfreq=samplingRate)

		markers = get_markers(session_data)

		df["stim"] = [0] * len(df)
		for marker in markers:
			marker_timestamp = marker["timestamp"]
			pandas_timestamp = df.index[df.index.get_loc(marker_timestamp, method='nearest')]
			if marker["label"] in markers_const:
				df.at[pandas_timestamp, "stim"] = event_id[marker["label"]]

		nparr = df.to_numpy().T
		raw = RawArray(data=nparr, info=info, verbose=False)

		events = find_events(raw)


        # Create an MNE Epochs object representing all the epochs around stimulus presentation
		epochs = Epochs(raw, events=events, event_id=event_id,
                                        tmin=tmin, tmax=tmax, baseline=None,
                                        preload=True,
                                        verbose=False)
		print('sample drop %: ', (1 - len(epochs.events)/len(events)) * 100)
		epochs_list.append(epochs)
	concat_epochs = concatenate_epochs(epochs_list)

	return concat_epochs
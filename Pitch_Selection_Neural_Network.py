#MODEL WORKS, CHANGE WHAT YOU ARE PREDICTING IN THE 'GET_PITCH_CODE' FUNCTION
#THE PREDICTION FUCNTION DOESN'T WORK YET, NOT SURE WHAT IS CAUSING IT TO NOT WORK BUT IT ALWAYS PREDICTS 1/MAX

from __future__ import absolute_import, division, print_function

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#Preproccesing modules
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Helper libraries
import numpy as np
import pandas as pd
import os
import sys

#Data library
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup

#The possible pitch types from statcast
#https://www.daktronics.com/support/kb/Pages/DD3312647.aspx
pitcher_pitches = ['CH', 'CU', 'FC', 'FF', 'FO', 'FS', 'FT','KC', 'KN', 'SC', 'SI', 'SL']

def get_pitcher_model(first_name, last_name):
	
	#Returns an int corresponding to the pitch type and location
	def get_pitch_code(row, pitcher_pitches):
		pitch_num = pitcher_pitches.index(row['pitch_type'])
		#since there are 14 possible zones
		#return (pitch_num*14)+row['zone']
		return row['zone']

	#This gets the previous three pitches and adds them to the row
	#If it's the first pitch of a PA, the previous pitches will all be -1
	#If it's the second or third, null pitches will be 0
	def get_prev_pitch(dataDF):
		data = dataDF.values

		#Three, two, and one pitch ago
		prev_pitch_three = []
		prev_pitch_two = []
		prev_pitch_one = [] 

		batter_id = 7
		prev_pitch_code = 90
		game_date = 2

		for i in range(0, np.size(data, 0)):
			
			#If this is the first pitch of the PA
			if(data[i-1][batter_id] != data[i][batter_id] and data[i-1][game_date] != data[i][game_date]):
				prev_pitch_one.append(-1)
				prev_pitch_two.append(-1)
				prev_pitch_three.append(-1)
			else:
				prev_pitch_one.append(data[i-1][prev_pitch_code])

				#If this is the second pitch of the PA
				if(data[i-2][batter_id] != data[i][batter_id] and data[i-2][game_date] != data[i][game_date]):
					prev_pitch_two.append(0)
					prev_pitch_three.append(0)
				else:
					prev_pitch_two.append(data[i-2][prev_pitch_code])

					#If this is the third pitch of the PA
					if(data[i-3][batter_id] != data[i][batter_id] and data[i-3][game_date] != data[i][game_date]):
						prev_pitch_three.append(0)
					else:
						prev_pitch_three.append(data[i-3][prev_pitch_code])

		dataDF['prev_pitch_1'] = prev_pitch_one
		dataDF['prev_pitch_2'] = prev_pitch_two
		dataDF['prev_pitch_3'] = prev_pitch_three
		return dataDF		



	#Read in and return the data we need
	def get_data(first_name, last_name):

		train_filename = 'Data/'+str(last_name)+"_"+str(first_name)+"_train.csv"
		test_filename = 'Data/'+str(last_name)+"_"+str(first_name)+"_test.csv"

		if os.path.isfile(train_filename) and os.path.isfile(test_filename): #If we've already gotten the data, read it in
			train_data = pd.read_csv(train_filename)
			test_data = pd.read_csv(test_filename)
		else: #If we haven't, get it off the web and store it for future runs
			#training is done on data from 2015 through 2017
			train_data = statcast_pitcher(start_dt='2015-01-01', end_dt='2019-12-31', player_id=int(playerid_lookup('sale', 'chris')['key_mlbam']))
			train_data.to_csv(train_filename)
			#testing is done on data from the beginning of 2018 to present
			test_data = statcast_pitcher(start_dt='2018-01-01', end_dt='2019-12-31', player_id=int(playerid_lookup('sale', 'chris')['key_mlbam']))
			test_data.to_csv(test_filename)

		#Get all of the pitch types that a pitcher throws, then encode them using our system
		train_data = train_data[train_data['pitch_type'].isin(pitcher_pitches)]
		train_data = train_data.dropna(subset=['pitch_type'])
		train_data['pitch_code'] = train_data.apply (lambda row: get_pitch_code(row, pitcher_pitches), axis=1)


		#Do the same as above but for the testing data in case they added a new pitch
		test_data = test_data[test_data['pitch_type'].isin(pitcher_pitches)]
		test_data = test_data.dropna(subset=['pitch_type'])
		#Encode all the pitch type/location info to a unique int
		test_data['pitch_code'] = test_data.apply (lambda row: get_pitch_code(row, pitcher_pitches), axis=1)

		train_data = get_prev_pitch(train_data)
		test_data = get_prev_pitch(test_data)

		#Fills the Na values, turns the batter ID for the player on base into a bool value
		train_data['on_3b'] = train_data['on_3b'].fillna(value=0).astype(bool).astype(int)
		train_data['on_2b'] = train_data['on_2b'].fillna(value=0).astype(bool).astype(int)
		train_data['on_1b'] = train_data['on_1b'].fillna(value=0).astype(bool).astype(int)

		test_data['on_3b'] = test_data['on_3b'].fillna(value=0).astype(bool).astype(int)
		test_data['on_2b'] = test_data['on_2b'].fillna(value=0).astype(bool).astype(int)
		test_data['on_1b'] = test_data['on_1b'].fillna(value=0).astype(bool).astype(int)

		#Get the data we need and drop any null values (which is why it double selects)
		train_data_input = train_data[['prev_pitch_3', 'prev_pitch_2', 'prev_pitch_1', 'balls', 'strikes', 'stand', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'pitch_number', 'pitch_code']].dropna()
		train_data_result = train_data_input[['pitch_code']]
		train_data_input = train_data_input[['prev_pitch_3', 'prev_pitch_2', 'prev_pitch_1', 'balls', 'strikes', 'stand', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'pitch_number']]

		test_data = test_data[['prev_pitch_3', 'prev_pitch_2', 'prev_pitch_1', 'balls', 'strikes', 'stand', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'pitch_number', 'pitch_code']].dropna()

		return train_data_input, train_data_result, test_data


	#Turns string columns into categorical ints
	def pre_process_data(train_data, train_data_result, test_data):

		label_encoder_handedness = LabelEncoder()
		train_data['stand'] = label_encoder_handedness.fit_transform(train_data['stand'])
		test_data['stand'] = label_encoder_handedness.fit_transform(test_data['stand'])
		

		train_data_result = np_utils.to_categorical(train_data_result)
		print("TRAIN DATA RESULT", train_data_result)

		sc = StandardScaler()
		train_data = sc.fit_transform(train_data)
		test_data[['prev_pitch_3', 'prev_pitch_2', 'prev_pitch_1', 'balls', 'strikes', 'stand', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'pitch_number']] = sc.fit_transform(test_data[['prev_pitch_3', 'prev_pitch_2', 'prev_pitch_1', 'balls', 'strikes', 'stand', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'pitch_number']])

		return train_data, train_data_result

	def create_model(train_data_input, train_data_result, saved_model_name):
		
		#Initialize nerual network
		model = Sequential()
		model.add(Dense(64, activation = 'relu', input_dim = 11))
		#model.add(Dense(64, activation = 'relu', input_dim = np.size(train_data_input,1)))
		#model.add(Dense(6, init='uniform', activation = 'relu'))
		model.add(Dense(15, activation = 'softmax'))

		# Compiling Neural Network
		model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

		# Fitting our model 
		model.fit(train_data_input, train_data_result, batch_size = 5, nb_epoch = 50, verbose=1)

		model_json = model.to_json()
		with open(saved_model_name+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(saved_model_name+".h5")
		print("Saved model to disk")

		return model

	def fetch_model(saved_model_name):
		# load json and create model
		json_file = open(saved_model_name+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(saved_model_name+".h5")
		print("Loaded model from disk")
		return loaded_model

	def test_model(train_data, train_data_result, model):
		model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
		predict = model.predict(train_data)
		print(predict)
		returnData = pd.DataFrame()

		actual_pitch = []
		predicted_pitch = []

		for i in range(0,len(train_data_result)):
			actual_pitch.append(np.argmax(train_data_result[i]))
			predicted_pitch.append(np.argmax(predict[i]))

		returnData['pitch'] = actual_pitch
		returnData['predicted'] = predicted_pitch
		print(returnData)
		return returnData

	def score_data(returnData):

		def is_next_to(actual, projected):
			if predicted == 1:
				if actual in([11, 2, 4, 5]):
					return True
			if predicted == 2:
				if actual in([11, 12, ]):
					return True

	#Get the pitch data and process it so that it is ready for the machine learning model
	train_data_input, train_data_result, test_data = get_data(first_name, last_name)
	train_data, train_data_result = pre_process_data(train_data_input, train_data_result, test_data)
	saved_model_name = 'Data/'+str(last_name)+"_"+str(first_name)+"_model"
	print(train_data)
	print(train_data_result)
	#If we already have the model, load it, else make it
	if os.path.isfile(saved_model_name+'.h5'):
		model = fetch_model(saved_model_name)
	else:
		model = create_model(train_data_input, train_data_result, saved_model_name)

	returnData = test_model(train_data, train_data_result, model)
	print(test_data)
	return test_data

get_pitcher_model('chris', 'sale')
import os
from flask import Flask, request, jsonify, send_from_directory, after_this_request
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import numpy as np

from src import create_workspace_object, predict_model

UPLOAD_FOLDER = os.path.join('src', 'models')
ALLOWED_EXTENSIONS = {'csv'}

CONNECTION_PASSWORD = 'blZEH6f6efEBr2jF'
CONNECTION_STRING = f'mongodb+srv://dylantarlowe:{CONNECTION_PASSWORD}@easyaicluster.9u2xuhc.mongodb.net/?retryWrites=true&w=majority'

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def connect_mongo(collection):
	mongo_client = MongoClient(CONNECTION_STRING)
	db = mongo_client['easyAI'][collection]
	return db

def get_user(db, userID):
	res = db.find_one({'_id' : ObjectId(userID)})
	if not res:
		return False
	return res

def serialize_predictions(preds):
	return jsonify({ i:pred for (i,pred) in zip([i for i in range(preds.size)], list(preds))})

@app.route('/train/<userID>', methods=['POST'])
def train(userID):
	#check if valid file
	if 'file' not in request.files:
		return 'no file in request'
	f = request.files['file']
	if f.filename == '':
		return 'no selected file'
	#load data into dataframe to be passed into the training algo
	x_train = pd.read_csv(f.stream)

	#obtain user input from request
	task = request.form.get('task')
	labels = request.form.get('labels')
	projectID = request.form.get('projectID')
	#train model
	workspace_object = create_workspace_object(x_train, task, labels)
	
	#extract relavent information from model
	datapoints = len(workspace_object.preprocessor.targets)
	accuracy = workspace_object.performance_metric
	features = len(workspace_object.preprocessor.samples.T)
	layers = len(workspace_object.model.layers)
	pickled_model = pickle.dumps(workspace_object.model)

	db = connect_mongo('users')
	query = { '_id' : ObjectId(userID)}
	db.update_one(query, {
		'$set' : { 
			f'models.{projectID}' : { 
				'title' : projectID,
				'task' : task,
				'labels' : labels,
				'model' : pickled_model,
				'metrics' : {
					'datapoints' :	datapoints,
					'accuracy' : accuracy,
					'features' : features,
					'layers' : layers
				}
		}}
	})

	return 'successful'

@app.route('/predict/<userID>', methods=['POST'])
def predict(userID):
	#check if valid file
	if 'file' not in request.files:
		return 'no file in request'
	f = request.files['file']
	if f.filename == '':
		return 'no selected file'
	#create dataframe from file
	x_predict = pd.read_csv(f.stream).to_numpy()
	#pull data from form
	projectID = request.form.get('projectID')
	#pull model to load it for predictions
	db = connect_mongo('users')
	query = { '_id' : ObjectId(userID) }
	user = db.find_one(query)
	model = pickle.loads(user['models'][projectID]['model'])

	predictions = model.predict(x_predict)

	# TODO : Return index with heighest probability so that user gets deterministic answers
	# Discuss with Richie on how to handle these predictions, whether the user might want the
	# model certainty or not
	
	return jsonify({'predictions:' : predictions.tolist()})

@app.route('/download/<userID>/<projectID>')
def download(userID, projectID):
	db = connect_mongo('users')
	query = {'_id' : ObjectId(userID) }
	user = db.find_one(query)
	model = pickle.loads(user['models'][projectID]['model'])
	model.save(f'{app.config["UPLOAD_FOLDER"]}/{projectID}.h5')
	@after_this_request
	def remove_file(res):
		try:
			os.remove(f'{app.config["UPLOAD_FOLDER"]}/{projectID}.h5')
		except:
			return 'error removing file'
		return res
	return send_from_directory(app.config['UPLOAD_FOLDER'], f'{projectID}.h5', as_attachment=True)

		

@app.route('/version', methods=['GET'])
def version():
	return 'beta 4'

# @app.route('/test/<userID>/<projectID>', methods=['GET'])
# def test(userID, projectID):

if __name__ == "__main__":	
	app.run()
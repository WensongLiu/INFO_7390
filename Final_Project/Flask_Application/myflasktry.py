from flask import Flask
from flask import request, url_for, redirect
from flask import render_template
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
#import argparse
import imutils
import pickle
import cv2
import ssl
import urllib.request as ur
import os

app = Flask(__name__)

modelPath = './src/final.h5'
labelPath = './src/fruits.pickle'
picPath = './static/test.jpg'


ssl._create_default_https_context = ssl._create_unverified_context

def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	keras.backend.get_session().run(tf.local_variables.initializer())
	return auc

global graph
graph = tf.get_default_graph()
model = load_model(modelPath, custom_objects={'auc':auc})

@app.route('/')
@app.route('/index')
def index():
	return render_template('homepage.html')

@app.route('/test', methods=['POST'])
def test():
	u = request.form['imageurl']
	name = request.form['fruit']
	download_img_from_url(u)
	with graph.as_default():
		pred_label= classify (labelPath, picPath, name)
	return redirect(url_for('result',label=pred_label, imgurl = u))

@app.route('/refresh')
def refresh_model():
	ur.urlretrieve('http://7390final.oss-us-east-1.aliyuncs.com/final.h5', './src/final.h5')
	ur.urlretrieve('http://7390final.oss-us-east-1.aliyuncs.com/fruits.pickle', './src/fruits.pickle')
	return redirect(url_for('index'))

@app.route('/result')
def result():
	label = request.args.get('label')
	url = request.args.get('imgurl')
	return render_template('result.html',label=label, url=url)

# def loadings(modelPath, labelbinPath):
# 	model = load_model(modelPath)
# 	lb = pickle.loads(open(labelbinPath, "rb").read())

def download_img_from_url(imageurl):
	filepath = './static/test.jpg'
	if os.path.exists(filepath):
		os.remove(filepath)
	ur.urlretrieve(imageurl, filepath)



def classify (labelbinPath, imagePath, name):
	image = cv2.imread(imagePath)

	image = cv2.resize(image, (100, 100))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	lb = pickle.loads(open(labelbinPath, "rb").read())
	
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]

	# filename = imagePath[imagePath.rfind(os.path.sep) + 1:]
	correct = "correct" if name.rfind(label) != -1 else "incorrect"

	label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)

	return label

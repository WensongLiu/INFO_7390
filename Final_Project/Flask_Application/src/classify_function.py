# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
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

def download_img_from_url(imageurl):
	ssl._create_default_https_context = ssl._create_unverified_context
	ur.urlretrieve(imageurl, './files/test.jpg')

def classify (model, lb, imagePath):
	image = cv2.imread(imagePath)

	image = cv2.resize(image, (100, 100))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]

	return label, proba[idx] * 100










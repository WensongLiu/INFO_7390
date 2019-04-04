import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.models import load_model
import imutils




def train_model(datasetPath, modelPath, labelPath):
	EPOCHS = 150
	INIT_LR = 1e-3
	BS = 64
	IMAGE_DIMS = (100, 100, 3)

	data = []
	labels = []

	print("[INFO] loading images...")
	imagePaths = sorted(list(paths.list_images(datasetPath)))
	random.seed(42)
	random.shuffle(imagePaths)

	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
		data.append(image)
	 
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	print("[INFO] data matrix: {:.2f}MB".format(
		data.nbytes / (1024 * 1000.0)))

	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)

	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.2, random_state=42)

	aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")

	print("[INFO] compiling model...")
	model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
		depth=IMAGE_DIMS[2], classes=len(lb.classes_))
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	print("[INFO] training network...")
	H = model.fit_generator(
		aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS, verbose=1)

	print("[INFO] serializing network...")
	model.save(modelPath)

	print("[INFO] serializing label binarizer...")
	f = open(labelPath, "wb")
	f.write(pickle.dumps(lb))
	f.close()

	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="upper left")
	plt.savefig(./plot.jpg)


def test_model(modelPath, labelPath, testImagePath)：
	image = cv2.imread(testImagePath)
	output = image.copy()
	 
	image = cv2.resize(image, (100, 100))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	print("[INFO] loading network...")
	model = load_model(modelPath)
	lb = pickle.loads(open(labelPath, "rb").read())

	print("[INFO] classifying image...")
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]

	filename = (testImagePath)[testImagePath.rfind(os.path.sep) + 1:]
	correct = "correct" if filename.rfind(label) != -1 else "incorrect"

	label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
	output = imutils.resize(output, width=400)
	cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)

	print("[INFO] {}".format(label))
	cv2.imshow("Output", output)
	cv2.waitKey(0)



if __name__ == '__main__':
	datasetPath = "./dataset"
	modelPath = "./Final_Result/final_model.h5"
	labelPath = "./Final_Result/classes_labels.pickle"
	testImagePath = "./Images_For_Test/apple_01.jpeg"
	train_model(datasetPath, modelPath, labelPath)
	test_model(modelPath, labelPath, testImagePath)

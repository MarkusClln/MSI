from imutils import paths
import face_recognition
import pickle
import cv2
import os

# This script is used for creating encodings of the given dataset
# Build the dataset as follows:
# dataset-name Folder
# > Name-of-person Folder
# >> Images of the person, which should be used for learning
dataset = "dataset-office"

# use hog for cpu usage
# use cnn for gpu usage (only worth if dlib was installed with gpu usage)
detectionMethod = "hog"

# get all images from the dataset
print("[INFO] Messe Gesichter...")
imagePaths = list(paths.list_images(dataset))

# initialize lists for encodings and names
knownEncodings = []
knownNames = []

# iterate over all images
for (i, imagePath) in enumerate(imagePaths):
	# get the persons name by extracting the foldername
	print("[INFO] Verarbeite Bild {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect faces
	boxes = face_recognition.face_locations(rgb, model=detectionMethod)

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# save encodings in pickle file
print("[INFO] Serialisiere Kodierungen...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
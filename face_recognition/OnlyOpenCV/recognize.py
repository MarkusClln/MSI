import numpy as np
import imutils
import pickle
import cv2

from os.path import dirname, abspath

faceRecBasePath = str(dirname(dirname(dirname(abspath(__file__)))))+"/example_images/face_recognition/"

demoNumber = 1
imageNumber = 3

if(demoNumber == 1):
	inputImg = faceRecBasePath + "theoffice.jpg"
	recognizerOut = "output/recognizer_office.pickle"
	labelEnc = "output/le_office.pickle"
else:
	if(imageNumber == 1):
		inputImg = faceRecBasePath + "jurassic_park_01.png"
	elif(imageNumber == 2):
		inputImg = faceRecBasePath + "jurassic_park_02.png"
	else:
		inputImg = faceRecBasePath + "jurassic_park_03.png"
	recognizerOut = "output/recognizer_jurassicpark.pickle"
	labelEnc = "output/le_jurassicpark.pickle"

detectorDeploy = "face_detection_model/deploy.prototxt"
detectorModel = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
embeddingModel ="openface_nn4.small2.v1.t7"
confidenceDef = 0.5

# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = cv2.dnn.readNetFromCaffe(detectorDeploy, detectorModel)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(recognizerOut, "rb").read())
le = pickle.loads(open(labelEnc, "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(inputImg)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections
	if confidence > confidenceDef:
		# compute the (x, y)-coordinates of the bounding box for the
		# face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# draw the bounding box of the face along with the associated
		# probability
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(image, name, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
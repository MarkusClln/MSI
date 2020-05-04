from imutils import face_utils
import imutils
import dlib
import cv2
from os.path import dirname, abspath

exampleImagesDir = str(dirname(dirname(abspath(__file__))))+"/example_images/facial_landmarks/"

demoNumber = 1

# chris pratt frontal
if(demoNumber == 1):
	imageName = "facial_landmarks_1.jpeg"
	thickness = 3
# chris pratt angle
elif(demoNumber == 2):
	imageName = "facial_landmarks_2.jpg"
	thickness = 2
# himym group
else:
	imageName = "facial_landmarks_3.png"
	thickness = 1

# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(exampleImagesDir+imageName)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# face_detection in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine facial landmarks for the face region
	shape = predictor(gray, rect)

	# convert landmarks to array
	shape = face_utils.shape_to_np(shape)

	# dlib rectangle need to be converted to an opencv bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)

	# loop over the facial landmarks and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), thickness)

# show the output image with the face detections and facial landmarks
cv2.imshow("Facial landmarks", image)
cv2.waitKey(0)
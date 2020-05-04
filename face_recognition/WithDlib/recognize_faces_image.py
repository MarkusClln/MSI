from os.path import abspath, dirname

import face_recognition
import pickle
import cv2

faceRecBasePath = str(dirname(dirname(dirname(abspath(__file__)))))+"/example_images/face_recognition/"

demoNumber = 2
imageNumber = 1

if(demoNumber ==1):
    encodingsFile = "encodings_office.pickle"
    inputImg = faceRecBasePath + "theoffice.jpg"
else:
    if (imageNumber == 1):
        inputImg = faceRecBasePath + "jurassic_park_01.png"
    elif (imageNumber == 2):
        inputImg = faceRecBasePath + "jurassic_park_02.png"
    else:
        inputImg = faceRecBasePath + "jurassic_park_03.png"

    encodingsFile = "encodings_jurassicpark.pickle"

unknownFaceTitle = "???"

# use hog for cpu usage
# use cnn for gpu usage (only worth if dlib was installed with gpu usage)
detectionMethod = "hog"

# load generated encodings (you need to run "encode_faces.py" before you can use this script)
print("[INFO] Lade Kodierungen...")
data = pickle.loads(open(encodingsFile, "rb").read())

# select an input image and convert it from BGR to RGB
image = cv2.imread(inputImg)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# face_detection
boxes = face_recognition.face_locations(rgb, model=detectionMethod)

# calculate encodings
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# loop over the facial embeddings
for encoding in encodings:
    # attempt to match each face in the input image to our known
    # encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = unknownFaceTitle

    # check to see if we have found a match
    if True in matches:
        # find the indexes of all matched faces then initialize a
        # dictionary to count the total number of times each face
        # was matched
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # determine the recognized face with the largest number of
        # votes (note: in the event of an unlikely tie Python will
        # select first entry in the dictionary)
        name = max(counts, key=counts.get)

    # update the list of names
    names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    green = (0, 255, 0)
    red = (0, 0, 255)

    # draw the predicted face name on the image
    if name == unknownFaceTitle:
        cv2.rectangle(image, (left, top), (right, bottom), red, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
    else:
        cv2.rectangle(image, (left, top), (right, bottom), green, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
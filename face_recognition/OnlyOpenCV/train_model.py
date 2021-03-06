# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--embeddings", required=True,
# 	help="path to serialized db of facial embeddings")
# ap.add_argument("-r", "--recognizer", required=True,
# 	help="path to output model trained to recognize faces")
# ap.add_argument("-l", "--le", required=True,
# 	help="path to output label encoder")
# args = vars(ap.parse_args())

demoNumber = 2

if(demoNumber == 1):
    embeddings = "output/embeddings_office.pickle"
    recognizerOut = "output/recognizer_office.pickle"
    labelEnc = "output/le_office.pickle"
else:
    embeddings = "output/embeddings_jurassicpark.pickle"
    recognizerOut = "output/recognizer_jurassicpark.pickle"
    labelEnc = "output/le_jurassicpark.pickle"

#embeddings = "output/embeddings.pickle"
#recognizerOut = "output/recognizer.pickle"
#labelEnc = "output/le.pickle"


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(embeddings, "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(recognizerOut, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(labelEnc, "wb")
f.write(pickle.dumps(le))
f.close()
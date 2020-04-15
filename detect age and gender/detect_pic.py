import cv2
import gad
from os.path import dirname, abspath

exampleImagesDir = str(dirname(dirname(abspath(__file__))))+"/example_images/"
imageName = "trump.jpg"
image = cv2.imread(exampleImagesDir+imageName)
image = gad.run(image)
cv2.imshow("Output", image)
cv2.waitKey(0)
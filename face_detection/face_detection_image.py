import cv2
from os.path import dirname, abspath

exampleImagesDir = str(dirname(dirname(abspath(__file__))))+"/example_images/face_detection/"

demoNumber = 2
useEyeDetection = False

# massive group
if(demoNumber == 1):
    imageName = "face_detection_2.png"
# himym
else:
    imageName = "face_detection_1.jpg"

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

img = cv2.imread(exampleImagesDir+imageName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    if(useEyeDetection):
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
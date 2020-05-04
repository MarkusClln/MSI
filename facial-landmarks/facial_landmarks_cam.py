from imutils import face_utils
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
def get_landmarks(image):
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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # loop over the facial landmarks and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    return image



cam = cv2.VideoCapture(0)
cv2.namedWindow("Facial landmark CAM")

img_counter = 0

while True:
    ret, frame = cam.read()
    print(frame)
    cv2.imshow("Facial landmark CAM", get_landmarks(frame))
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()
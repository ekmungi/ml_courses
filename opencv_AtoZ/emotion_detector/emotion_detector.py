import cv2 

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_detector = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect(gray, frame):
    face = face_detector.detectMultiScale(gray, scaleFactor=2, minNeighbors=2)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        face_roi = gray[x:x+w, y:y+h]
        
        # Change the last two parameters to play around with the number of ensembles to use to get the prediction
        smile = smile_detector.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22) 
        for (xs, ys, ws, hs) in smile:
            cv2.rectangle(frame, (x+xs,y+ys), (x+xs+ws,y+ys+hs), (0,255,255), 2)

        eyes = eyes_detector.detectMultiScale(face_roi, scaleFactor=1.2, minNeighbors=22)
        for (xe, ye, we, he) in eyes:
            cv2.rectangle(frame, (x+xe,y+ye), (x+xe+we,y+ye+he), (255,0,255), 2)
            


    return frame






# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
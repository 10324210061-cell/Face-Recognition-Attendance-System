import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Path where face images are stored
path = "faces"
images = []
classNames = []

# Load all images and names
for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = face_recognition.load_image_file(f"{path}/{file}")
        images.append(img)
        classNames.append(os.path.splitext(file)[0])

print("Loaded Students:", classNames)

# Encode all known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Completed")

# Mark attendance in CSV (one entry per person per day)
def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        attendanceList = []

        today = datetime.now().strftime('%Y-%m-%d')

        for line in myDataList:
            entry = line.strip().split(',')
            if len(entry) >= 2:
                attendanceList.append((entry[0], entry[1]))

        if (name, today) not in attendanceList:
            now = datetime.now()
            dateString = now.strftime('%Y-%m-%d')
            timeString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString},{timeString}')
            print(f"Attendance marked for {name}")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    # Recognition block (corrected)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        if len(encodeListKnown) == 0:
            name = "UNKNOWN"
        else:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDist)
            confidence = (1 - faceDist[matchIndex]) * 100

            if matches[matchIndex] and confidence > 50:   # 50% confidence threshold
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else:
                name = "UNKNOWN"

        # Scale back face location
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Attendance System", img)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

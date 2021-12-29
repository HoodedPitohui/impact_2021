import cv2
import imutils

#initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('humans.mp4')
#video from: https://www.pexels.com/search/videos/pedestrians/

while cap.isOpened():
    #reading the video stream
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width = min(400, image.shape[1]))
        #Detect all regions with peds. inside of them
        (regions, _) = hog.detectMultiScale(image,
                                            winStride=(4, 4),
                                            padding = (4, 4),
                                            scale = 1.05)
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y),
                          (x + w, y + h),
                          (0, 0, 255), 2)
            #showing output image
            cv2.imshow("Image", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    else:
        break
cap.release()
cv2.destroyAllWindows()
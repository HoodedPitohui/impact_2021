import numpy as np
import cv2
#https://www.geeksforgeeks.org/pedestrian-detection-using-opencv-python/
#For human detection, opencv has Histogram of Ordered Descents (HOG)
import imutils

#initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#future note: try daimler people detector

image = cv2.imread('people.jpg')

#resize image
image = imutils.resize(image, width = min(400, image.shape[1]))

#Detect all the regions in the image that could have people
(regions, _) = hog.detectMultiScale(image,
                                    winStride = (4, 4),
                                    padding = (4, 4),
                                    scale = 1.05)

#Drawing the regions in the image
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h),
                  (0, 0, 255), 2)

#Showing the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
import cv2
import imutils

image_raw = cv2.imread('data/00.jpg')
image = cv2.resize(image_raw, (800, 600))
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# creating masks for matching
banana_mask = cv2.inRange(hsv, (23, 85, 100), (40, 255, 255))
orange_mask = cv2.inRange(hsv, (0, 216, 120), (17, 255, 255))
apple_mask = cv2.inRange(hsv, (0, 28, 0), (7, 255, 255))

masks = [apple_mask, orange_mask, banana_mask]
apple_res = cv2.bitwise_and(image, image, mask=apple_mask)
orange_res = cv2.bitwise_and(image, image, mask=orange_mask)
banana_res = cv2.bitwise_and(image,image, mask=banana_mask)

# converting colorspaces for further thresholding
apple_res = cv2.cvtColor(apple_res, cv2.COLOR_HSV2BGR)
apple_res = cv2.cvtColor(apple_res, cv2.COLOR_BGR2GRAY)
orange_res = cv2.cvtColor(orange_res, cv2.COLOR_HSV2BGR)
orange_res = cv2.cvtColor(orange_res, cv2.COLOR_BGR2GRAY)
banana_res = cv2.cvtColor(banana_res, cv2.COLOR_HSV2BGR)
banana_res = cv2.cvtColor(banana_res, cv2.COLOR_BGR2GRAY)

# thresholding result image to find contours
blur_apple = cv2.GaussianBlur(apple_res, (7, 7), 0)
ret2, apple_res = cv2.threshold(blur_apple, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur_orange = cv2.GaussianBlur(orange_res, (7, 7), 0)
ret3, orange_res = cv2.threshold(blur_orange, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur_banana = cv2.GaussianBlur(banana_res, (7, 7), 0)
ret4, banana_res = cv2.threshold(blur_banana, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# contours searching
apple_contours = cv2.findContours(apple_res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
apple_contours = imutils.grab_contours(apple_contours)
orange_contours = cv2.findContours(orange_res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
orange_contours = imutils.grab_contours(orange_contours)
banana_contours = cv2.findContours(banana_res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
banana_contours = imutils.grab_contours(banana_contours)

# declaration of variables
apple_area = []
banana_area = []
orange_area = []

apple = 0
banana = 0
orange = 0

for a_c in apple_contours:
    apple_area = cv2.contourArea(a_c)

    if apple_area> 2000:
        apple = apple+1
        cv2.drawContours(image, [a_c], -1, (255, 0, 0), 2)

for o_c in orange_contours:
    orange_area = cv2.contourArea(o_c)

    if orange_area> 2000:
        orange = orange+1
        cv2.drawContours(image, [o_c], -1, (0, 255, 0), 2)

for b_c in banana_contours:
    banana_area = cv2.contourArea(b_c)

    if banana_area> 2000:
        banana = banana+1
        cv2.drawContours(image, [b_c], -1, (0, 0, 255), 2)

cv2.imshow('Result', image)
print('Banana: ' + str(banana) + '\n' 'Orange: ' + str(orange) + '\n' 'Apple: ' + str(apple) + '\n')

cv2.waitKey(0)


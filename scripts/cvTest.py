# import cv2

# im = cv2.imread('../pics/14.png')
# gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# idx =0 
# for cnt in contours:
#     idx += 1
#     x,y,w,h = cv2.boundingRect(cnt)
#     roi=im[y:y+h,x:x+w]
#     cv2.imwrite(str(idx) + '.png', rpytonoi)
#     #cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
# cv2.imshow('img',im)
# cv2.waitKey(0)    
# cv2.destroyWindow()

import numpy as np
import cv2
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

cv2.imshow('Rec',img)
cv2.imwrite('../pics/t.png', img)
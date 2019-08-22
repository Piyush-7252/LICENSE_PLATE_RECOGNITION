import imutils
import numpy as np
import cv2
import pytesseract

image=cv2.imread('number2.jpg') #provide image here

#image=imutils.resize(image,width=500)
image = cv2.resize(image,(620,480))

cv2.imshow("orignal image",image) # showing the orignal image
cv2.waitKey(0)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convertthe image into gray image
cv2.imshow("1-Grayscale image",gray)
cv2.waitKey(0)

gray=cv2.bilateralFilter(gray,11,17,17)  # bilateralFilter is user for smoothness
cv2.imshow("2-BILATERAL filter",gray)
cv2.waitKey(0)

edged=cv2.Canny(gray,190,210) #Canny is used for edge detection you can change the value 190 to 170 or else
cv2.imshow("3-Canny_edge",edged)
cv2.waitKey(0)

# now find the contours on image
#this gives 2 things one is contours and 2nd is harraricy
cnts,new=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#copy the image so orignal image will same if we need any other operation on it
img1=image.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3) # This draws the contours on image
cv2.imshow("4-ALL contours",img1)
cv2.waitKey(0)

#Consider only TOP 30 contours
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:30]

#we assuming NumberplateCnt is string  and none
NumberplateCnt=None

img2=image.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3) #this is for only top 30 contours here cnts is changed for top 30
cv2.imshow("5-TOP 30 contours",img2)
cv2.waitKey(0)

for c in cnts:
    peri=cv2.arcLength(c,True)  #gives peramiters
    approx=cv2.approxPolyDP(c,0.05*peri,True)  #shape of curves like 4 for rect , 1  for circle

    # Now take only rectangle part if apporx is 4
    if len(approx)==4:
        NumberplateCnt=approx

        x,y,w,h=cv2.boundingRect(c) #provides the cordinate of plate
        new_img=gray[y:y+h,x:x+w]
        #cv2.imshow('nn',new_img)
        #cv2.waitKey(0)
        cv2.imwrite('1.png',new_img)   #save the number plate image
        break

# now draw contours on only number plate
cv2.drawContours(image,[NumberplateCnt],-1,(0,255,0),3)
cv2.imshow("Final image",image)
cv2.waitKey(0)
cropped_image=cv2.imread('1.png')
cv2.imshow("cropped image",cropped_image)
cv2.waitKey(0)

# pytesseract is for reading character in string form
text=pytesseract.image_to_string(cropped_image,lang='eng')
print("number is:",text)
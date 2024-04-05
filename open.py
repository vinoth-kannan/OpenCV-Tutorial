import cv2 as cv
import numpy as np

'''for rescaling a frame'''
# can be used for image, video, live videos
def rescaleframe(frame,scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimen=(width,height)
    return cv.resize(frame,dimen)

''' for playing a video'''

vid=cv.VideoCapture("D:\\Domi_final.mp4")
while True:
    isTrue,frame=vid.read()
    resizedframe=rescaleframe(frame)
    cv.imshow("video",resizedframe)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

vid.release()
cv.destroyAllWindows()


'''for reading a image'''

img=cv.imread(r"C:\Users\vinot\OneDrive\Pictures\pexels.jpeg")
resizedimage=rescaleframe(img,2.0)
cv.imshow("resized image",resizedimage)
cv.imshow("original image",img)


'''for drawing shapes'''

blank=np.zeros((500,500,3),dtype='uint8')
cv.imshow("blank",blank)

# paint the entire image with certain color
blank[:,:,:]=0,255,0
cv.imshow("color",blank)

# paint particular area of the image with certain color
blank[100:200,200:300]=13,1,130
cv.imshow("specific Area",blank)

# draw a rectange 
cv.rectangle(blank,(100,100),(200,200),(20,150,250),thickness=1)
cv.imshow("rectangle",blank)

# draw a circle
cv.circle(blank,(250,250),color=(100,50,255),thickness=-1,radius=100)
cv.imshow("circle",blank)

# draw a line
cv.line(blank,(250,0),(250,500),color=(120,240,120),thickness=2)
cv.line(blank,(0,250),(5000,250),color=(120,240,120),thickness=2)
cv.imshow("line",blank)

# displaying text in a image
cv.putText(blank,"hello",(125,130),fontFace=cv.FONT_HERSHEY_TRIPLEX,thickness=5,fontScale=3.0,color=(255,255,255))
cv.imshow("text",blank)

'''color conversion of images'''

hsvimg=cv.cvtColor(img,cv.COLOR_BGR2HSV)
greyimg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
hlsimg=cv.cvtColor(img,cv.COLOR_BGR2HLS)
labimg=cv.cvtColor(img,cv.COLOR_BGR2LAB)
rgbimg=cv.cvtColor(img,cv.COLOR_BGRA2RGB)
cv.imshow("original image",img)
cv.imshow("HSV image",hsvimg)
cv.imshow("Grey image",greyimg)
cv.imshow("HLS image",hlsimg)
cv.imshow("LAB image",labimg)
cv.imshow("RGB image",rgbimg)
print(greyimg[:])


'''dilating the image'''
dilated=cv.dilate(canny,(7,7),iterations=3)
cv.imshow("dilated",dilated)

'''eroding the image'''
# convert dilated into canny
eroded=cv.erode(dilated,(7,7),iterations=3)
cv.imshow("eroded",eroded)

'''resize an image'''
resized=cv.resize(img,(700,500),interpolation=cv.INTER_CUBIC)
cv.imshow("resized",resized)

'''cropping an image'''
cropped=img[0:100,10:200]
cv.imshow("cropped",cropped)

'''translating an image'''
#  moving the image
def translate(img,x,y):
    transmat=np.float32([[1,0,x],[0,1,y]])
    dimens=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transmat,dimens)

translated=translate(img,-20,-30)
cv.imshow("transated",translated)


'''rotating an image'''
# rotates an image from a rotation point
def rotate(img,deg,rotpoint=None):
    height,width=img.shape[0:2]
    if rotpoint is None:
        rotpoint=(width//2,height//2)
    rotmat=cv.getRotationMatrix2D(rotpoint,deg,1.0)
    dimens=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,rotmat,dimens)

rotated=rotate(img,75)
cv.imshow("rotated",rotated)

'''flipping an image'''
flip0=cv.flip(img,0)
cv.imshow("vertical flip",flip0)
flip1=cv.flip(img,1)
cv.imshow("horizontal flip",flip1)
flipneg1=cv.flip(img,-1)
cv.imshow("both horizontal and vertical flip",flipneg1)

'''color channels'''
# allows the image to be splitted in their respective colors (b,g,r)
blankimg=np.zeros(img.shape[:2],dtype='uint8')
b,g,r=cv.split(img)
blue=cv.merge([b,blankimg,blankimg])
green=cv.merge([blankimg,g,blankimg])
red=cv.merge([blankimg,blankimg,r])
cv.imshow("original",img)
cv.imshow("blue",blue)
cv.imshow("green",green)
cv.imshow("red",red)
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)


# '''blurring an image'''
# reduces and noises and smoothens the image
# cv.imshow("original",img)
''' type1 - averaging blur ''' 
# smoothens a middle pixel based on the average of its neighbour pixels 
avgblur=cv.blur(img,(5,5))
cv.imshow("average blur",avgblur)

# ''' type2 - Gaussian blur ''' 
# to reduce noise for eg in low light photographs
# similar to average blur, but each neighbouring pixel is given a weight and
# average of product of these pixels gives middle
# gives less blur when compared to average blur but will be more natural (capture more edges)
# 0 - standard deviation in direction of X (sigmaX)
gaussblur=cv.GaussianBlur(img,(5,5),0)
cv.imshow("Guassian Blur",gaussblur)

# '''type3 - Median blur'''
# # Similar to average blur, but it finds the median of neighbouring pixels
# for finding the middle 
# more efficient for reducing noise
medblur=cv.medianBlur(img,5)
cv.imshow("Median Blur",medblur)

# '''type4 - Bilateral Blur'''
#  most effective blurring method used
#  reason: other methods simply blur the images without considering edges
#  It applies blur and also retains edges
#  sigma color - 10 - larger values indicates that more colors in neighbour will be used for computing blur
# sigma space - 15 - larger value indicates that middle pixel will be influenced also by faraway pixels
bilat=cv.bilateralFilter(img,5,10,15)
cv.imshow("bilateral Filter(blur)",bilat)


# canny(cascade edges) for differentiating different blur
avgcanny=cv.Canny(avgblur,50,100)
cv.imshow("average blurred canny",avgcanny)
gausscanny=cv.Canny(gaussblur,50,100)
cv.imshow("gaussian blurred canny",gausscanny)
medcanny=cv.Canny(medblur,50,100)
cv.imshow("Medium blurred canny",medcanny)
bilatcanny=cv.Canny(bilat,50,100)
cv.imshow("Bilateral blurred canny",bilatcanny)

'''saving a image'''
# to store the image to any storage device
# this will save the image according to given format

cv.imwrite(r"E:\newimg.jpeg",hsvimg)

'''thresholding value'''
# converting an image to binary 
# There will be a range upto 255
# Only greyscale image will be used
# If the pixel value is greater than thresholding value, then it wil be white
# else it will be set to black
ret,thresh=cv.threshold(greyimg,50,255,cv.THRESH_BINARY)
ret, thresh1 = cv.threshold(greyimg, 50, 255, cv.THRESH_BINARY_INV)
ret,thresh2=cv.threshold(greyimg,50,255,cv.THRESH_TRUNC)
ret,thresh3=cv.threshold(greyimg,50,255,cv.THRESH_TOZERO)
ret,thresh4=cv.threshold(greyimg,50,255,cv.THRESH_TOZERO_INV)
adaptive_threshold_mean=cv.adaptiveThreshold(greyimg,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,13,7)
adaptive_threshold_gauss=cv.adaptiveThreshold(greyimg,255,cv.ADAPTIVE_THRESH_GASS_C,cv.THRESH_BINARY,13,7)
thresh,otsuthresh=cv.threshold(greyimg,50,255,cv.THRESH_OTSU)


cv.imshow("original image",img)
cv.imshow("Binary Threshold",thresh)
cv.imshow("Binary Inverse Threshold", thresh1)
cv.imshow("Truncated Threshold",thresh2)
cv.imshow("ToZero Threshold",thresh3)
cv.imshow("To Zero In threshold",thresh4)
cv.imshow("Adaptive Thresholding Mean",adaptive_threshold_mean)
cv.imshow("Adaptive Thresholding Gaussian",adaptive_threshold_gauss)
cv.imshow("OTSU Thresholding",otsuthresh)


'''edge cascade (or) edge detection'''
#finds the edges present in a image
# canny - advanced edge detection algorithm
canny=cv.Canny(img,50,255)
cv.imshow("canny edges",canny)

# simple methods - laplacian and sobel method
# laplacian - calculates gradients of greyscale image
lap=cv.Laplacian(greyimg,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow("Laplacian",lap)
# sobel - calculates gradients in x and y directions
sobelx=cv.Sobel(greyimg,cv.CV_64F,1,0)
sobely=cv.Sobel(greyimg,cv.CV_64F,0,1)
combined=cv.bitwise_or(sobelx,sobely)
cv.imshow("sobelX",sobelx)
cv.imshow("sobelY",sobely)
cv.imshow("Combined sobel",combined)

cv.rectangle(greyimg,(0,0),(100,100),(0,255,0),thickness=1)
cv.circle(greyimg,(100,100),10,(0,255,0),thickness=1)
cv.line(greyimg,(0,0),(100,100),(0,255,0),thickness=1)
cv.putText(greyimg,"hello",(120,100),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2.0,color=(0,255,0),)
cv.imshow("rect",greyimg)


'''Contours'''
# contours - used in object detection and recognition
# boundaries of objects connected with a continuous line
blur=cv.GaussianBlur(greyimg,(5,5),cv.BORDER_DEFAULT)
cv.imshow("blur",blur)
canny=cv.Canny(greyimg,50,100)
cv.imshow("canny",canny)
contours,hierarchies=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(len(contours))
blank=np.zeros(img.shape[:],dtype='uint8')
cv.drawContours(blank,contours,-1,(0,255,0),thickness=1)
cv.imshow("blank",blank)


'''Morphological Operations'''
# Process images based on shapes and edges
# kernel - tells how to change value of any pixel
# erosion
# diltuion

kernela=np.ones((5,5),dtype='uint8')
eroded=cv.erode(img,kernel=kernela,iterations=1)
dilated=cv.dilate(img,kernel=kernela,iterations=1)
opening=cv.morphologyEx(img,cv.MORPH_OPEN,kernela)
closing=cv.morphologyEx(img,cv.MORPH_CLOSE,kernela)

cv.imshow("original image",img)
cv.imshow("erosion",eroded)
cv.imshow("dilated",dilated)
cv.imshow("opening",opening)
cv.imshow("closing",closing)


'''bitwise operators'''
# and  - return 1 if both pixels are 1
# or - returns 1 if even if any one of the pixels is 1
# not - returns the opposite of a pixel

blank=np.zeros((500,500,3),'uint8')
rect=cv.rectangle(blank.copy(),(100,100),(400,400),(0,255,0),-1)
circ=cv.circle(blank.copy(),(250,250),200,(0,255,0),-1)
bit_and=cv.bitwise_and(rect,circ)
bit_or=cv.bitwise_or(rect,circ)
bit_not=cv.bitwise_not(rect,circ)
bit_xor=cv.bitwise_xor(rect,circ)
cv.imshow("rect",rect)
cv.imshow("circle",circ)
cv.imshow("and",bit_and)
cv.imshow("or",bit_or)
cv.imshow("not",bit_not)
cv.imshow("xor",bit_xor)



'''masking'''
# allows us to focus on certain parts of a image


blank=np.zeros(img.shape[:2],dtype='uint8')
mask=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
masked=cv.bitwise_and(img,img,mask=mask)


cv.imshow("original image",img)
cv.imshow("circ",mask)
cv.imshow("masked",masked)

cv.waitKey(0)
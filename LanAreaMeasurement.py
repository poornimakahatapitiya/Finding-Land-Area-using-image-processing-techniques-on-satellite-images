from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import statistics
from PIL import Image


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
def get_image(image_path):
    image = cv2.imread(image_path)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(imageRGB)
    plt.show()
    return imageRGB
def remove_noise(image):
    imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    medianBlur = cv2.medianBlur(imageGray, 3)
    cv2.imshow("blured image",medianBlur)
    return medianBlur
def findEdges(image):

    v = np.median(image)
    print(v)
    sigma = 0.33
    lower_thresh = int(max(0, (1 - sigma) * v))
    upper_thresh = int(min(255, (1+ sigma) * v))
    edges=cv2.Canny(image,lower_thresh,upper_thresh)
    cv2.imshow("canny",edges)
    cv2.imwrite("canny.jpg",edges)
    dialte = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cv2.imshow("Image with edges",dialte)
    cv2.imwrite("dialted.jpg",dialte)
    return dialte
def findContours(image):
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("no of contours found= " + str(len(contours)))
    return contours
def drawContour(contours,image):
    contoured=cv2.drawContours(image,contours, -1, (0,255,0), 3)
    cv2.imshow("contoured image",contoured)

def findArea(contours):
    number=0
    areas=[]

    for contour in contours:
      area=cv2.contourArea(contour)
      areas.append(area)

    median=statistics.median(areas)
    mean=statistics.mean(areas)
    for area in areas:
        if (area > mean):
            number = number + 1
            print(area)

    print("The no of land plots: "+str(number))
def drawRectangle(contours,image):
    number = 0
    for c in contours:
     #x, y, w, h = cv2.boundingRect(c)
     (x,y),(w,h),a = cv2.minAreaRect(c)
     rect=cv2.minAreaRect(c)
     print(x,y,w,h,a)
     if(w>30 or h>30):
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
     #if(w>25 or h>10):
      #number = number + 1
      area=cv2.contourArea(c)
      #print(area)


      #cv2.putText(image, str(area), (box[0] , box[1]), 0, 0.3, (255, 255, 255))
    #print("The no of land plots: " + str(number))
    cv2.imshow("rectangle",image)


def get_Colors(image,no_of_colors):
 modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
 modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)
 clf = KMeans(n_clusters = no_of_colors)
 labels = clf.fit_predict(modified_image)
 counts = Counter(labels)
 center_colors = clf.cluster_centers_
 ordered_colors = [center_colors[i] for i in counts.keys()]
 hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
 rgb_colors = [ordered_colors[i] for i in counts.keys()]
 plt.figure(figsize = (8, 6))
 plt.pie(counts.values(),  colors = hex_colors,autopct='%.1f%%')
 plt.show()
 plt.savefig("pie.png")
 return rgb_colors

def findCropHealth(image):

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_res = imageRGB[ 733:839,196:1085]
    img_resBgr=cv2.cvtColor(img_res,cv2.COLOR_RGB2BGR)
    cv2.imshow("scale",img_resBgr)
    return img_resBgr
#colors=get_Colors(get_image('C:\\Users\\poornima\\Documents\\cs\\cs314\\project\\images\\Satellite-Map-NDVI-Agriculture-med.jpg'),8)


cv2.waitKey(0)
cv2.destroyAllWindows()

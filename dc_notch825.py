import numpy as np
import cv2
from matplotlib import pyplot as plt
H = 18 
W = 18
K = 1.1 
H1 = 1.8 
H2 = 0.6

ori = cv2.imread('img.jpeg')
f = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
f_bar = cv2.blur(f,(H,W))
f_sub = abs(np.mat(f) - np.mat(f_bar))

avg = int(round(f.mean() * K))
g = f_sub + avg 

hold1 = int(round(f.mean()*H1))
hold2 = int(round(g.mean()*H2))

ret,thresh1 = cv2.threshold(f,hold1,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(g,hold2,255,cv2.THRESH_BINARY_INV)

#width histogram create
img_width = thresh1[0].size
img_height = thresh1.size/thresh1[0].size

def gethistw(img_width, img_height, thresh):
	w_hist = []
	for i in range(0, img_width-1):
		sum = 0
		for j in range(0, img_height-1):
			if thresh[j][i] == 255:
				sum+=1
		w_hist.append(sum)
	return w_hist

def gethisth(img_width, img_height, thresh):
	h_hist = []
	for i in range(0, img_height-1):
		sum = 0
		for j in range(0, img_width-1):
			if thresh[i][j] == 255:
				sum+=1
		h_hist.append(sum)
	return h_hist

def getDiffer(hist, high, low):
	diff = []
	for i in range(0, len(hist)-2):
		if i < low or i > len(hist)-2 - high:
			diff.append(0)
		else:
			diff.append(abs(hist[i] - hist[i+1]))
	return diff

def getEdge(diff, K):
	result = []
	#max value trim
#	for i in range(0, len(diff)):
#		if diff[i] == max(diff):
#			diff[i] = 0
	thresh = max(diff) * K
	print 'thresh = '
	print thresh
	for i in range(0, len(diff)):
		if diff[i] < thresh or diff[i] == max(diff):
			result.append(0)
		else:
			result.append(diff[i])
	return result
def multiplyArr(a, b):
	result = []
	for i in range(0,len(a)):
		result.append( a[i] * b[i])
	return result

def drawXaxis(point, img):
 for i in range(0, point.size):
  if point.item(i) != 0:
   cv2.line(img, (i,0), (i, img.size / img[0].size), (255,0,0))

def drawYaxis(point, img):
	for i in range(0, point.size):
		if point.item(i) != 0:
			cv2.line(img, (0,i), (img[0].size, i),(255,0,0))
def drawRect(Arr, img):
 for i in range(0, len(Arr)):
  cv2.rectangle(img,Arr[i][0],Arr[i][1],(255,0,0), 3)

def periodFilter(w_diff1, K):
	result = []
	for i in range(0, len(w_diff1)/K):
		flag = 0
		for j in range(i*K, i*K+K):
			if w_diff1[j] != max(w_diff1[i*K:i*K+K]) or flag:
				w_diff1[j] = 0
			else:
				flag = 1
	for i in range(len(w_diff1)/K * K,len(w_diff1)):
		w_diff1[i] = 0


w_hist1 = gethistw(img_width, img_height, thresh1)
w_hist2 = gethistw(img_width, img_height, thresh2)
h_hist1 = gethisth(img_width, img_height, thresh1)
h_hist2 = gethisth(img_width, img_height, thresh2)

w_diff1 = getDiffer(w_hist1, 100, 100)
w_diff2 = getDiffer(w_hist2, 100, 100)
h_diff1 = getDiffer(h_hist1, 50, 150)
h_diff2 = getDiffer(h_hist2, 50, 150)

w_diff1 = getEdge(w_diff1, 0.1)
w_diff2 = getEdge(w_diff2, 0.1)
h_diff1 = getEdge(h_diff1, 0.1)
h_diff2 = getEdge(h_diff2, 0.1)
#80 50 180 150
periodFilter(w_diff1, 60)
periodFilter(w_diff2, 60)
periodFilter(h_diff1, 100)
periodFilter(h_diff2, 100)

def getPointArr(pointa, pointb):
 point1 = np.squeeze(np.asarray(pointa))
 point2 = np.squeeze(np.asarray(pointb))
 arr = []
 for i in range(0, len(point1)):
  subarr = []
  for j in range(0, len(point2)):
   if point1[i] !=0 and point2[j] != 0:
    subarr.append( (i,j) )
  if len(subarr) >0:
   arr.append(subarr)
 return arr

w_point = np.mat(w_diff1) + np.mat(w_diff2)
h_point = np.mat(h_diff1) + np.mat(h_diff2)

Arr = getPointArr(h_point,w_point)
print 'Arr = ' 
print Arr
#
#def getValidRect(Arr):
# result = []
# for i in range(0, len(Arr)-1):
#  for j in range(0, len(Arr[0])-1):
#   width = Arr[i+1][j+1][1] - Arr[i][j][1]
#   height = Arr[i+1][j+1][0] - Arr[i][j][0]
#   if width > 50 and height > width*1.5:
#    result.append( ( (Arr[i][j][1], Arr[i][j][0] ), (Arr[i+1][j+1][1], Arr[i+1][j+1][0] ) ))
# return result
#
def getValidRect(Arr, img):
 W = 30
 T = 0.4
 H = 1.2
 result = []
 for i in range(0, len(Arr)-1):
  for j in range(0, len(Arr[0])-1):
   width = Arr[i+1][j+1][1] - Arr[i][j][1]
   height = Arr[i+1][j+1][0] - Arr[i][j][0]
   threshhold = width * height * T
   rectVal = 0
   if width > W and height > width * H and height < width * 5* H:
    for k in range(Arr[i][j][0], Arr[i+1][j+1][0]):
     for q in range(Arr[i][j][1], Arr[i+1][j+1][1]):
      if img[k][q] == 255:
       rectVal = rectVal +1
    print 'squer = ' + str(width * height) + ' thr = ' + str(width*height*0.15)
    print 'rectVal = ' + str(rectVal) + ', threshhold = ' + str(threshhold)
    if rectVal > threshhold: 
     result.append( ( (Arr[i][j][1], Arr[i][j][0] ), (Arr[i+1][j+1][1], Arr[i+1][j+1][0] ) ))
     print result
 return result


rectList = getValidRect(Arr, thresh1)
print rectList
drawRect(rectList, ori)

#drawXaxis(w_point, f)
#drawYaxis(h_point, f)
##
#drawXaxis(np.mat(w_diff1), thresh1)
#drawYaxis(np.mat(h_diff1), thresh1)
#drawXaxis(np.mat(w_diff2), thresh2)
#drawYaxis(np.mat(h_diff2), thresh2)
#plt.plot(range(0, len(w_diff1)),w_diff1)
#plt.plot(range(0, len(w_diff2)),w_diff2)
#plt.plot(range(0, len(h_diff1)),h_diff1)
#plt.plot(range(0, len(h_diff2)),h_diff2)
##
cv2.imshow('ori',ori)
cv2.imshow('f',f)
cv2.imshow('f_bar',f_bar)
cv2.imshow('f_sub',f_sub)
cv2.imshow('g',g)
cv2.imshow('thresh1',thresh1)
cv2.imshow('thresh2',thresh2)
#
cv2.waitKey(0)


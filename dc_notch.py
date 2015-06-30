import numpy as np
import cv2
#from matplotlib import pyplot as plt
H = 18 
W = 18
K = 1.1 
H1 = 2.5 
H2 = 0.6

f = cv2.imread('img1.jpeg',0)
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

def getDiffer(hist, size, high, low):
	diff = []
	for i in range(0, size-2):
		if i < low or i > size-2 - high:
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

def periodFilter(w_diff1, K):
	result = []
	for i in range(0, len(w_diff1)/K):
		for j in range(i*K, i*K+K):
			if w_diff1[j] != max(w_diff1[i*K:i*K+K]):
				w_diff1[j] = 0


w_hist1 = gethistw(img_width, img_height, thresh1)
w_hist2 = gethistw(img_width, img_height, thresh2)
h_hist1 = gethisth(img_width, img_height, thresh1)
h_hist2 = gethisth(img_width, img_height, thresh2)

w_diff1 = getDiffer(w_hist1, img_width, 100, 100)
w_diff2 = getDiffer(w_hist2, img_width, 100, 100)
h_diff1 = getDiffer(h_hist1, img_height, 50, 150)
h_diff2 = getDiffer(h_hist2, img_height, 50, 150)

w_diff1 = getEdge(w_diff1, 0.1)
w_diff2 = getEdge(w_diff2, 0.1)
h_diff1 = getEdge(h_diff1, 0.1)
h_diff2 = getEdge(h_diff2, 0.1)

periodFilter(w_diff1, 80)
periodFilter(w_diff2, 50)
periodFilter(h_diff1, 180)
periodFilter(h_diff2, 150)



#w_point = np.mat(w_diff1) + np.mat(w_diff2)
#h_point = np.mat(h_diff1) + np.mat(h_diff2)
#drawXaxis(w_point, f)
#drawYaxis(h_point, f)
#
drawXaxis(np.mat(w_diff1), thresh1)
drawYaxis(np.mat(h_diff1), thresh1)
drawXaxis(np.mat(w_diff2), thresh2)
drawYaxis(np.mat(h_diff2), thresh2)
#plt.plot(range(0, len(w_diff1)),w_diff1)
#plt.plot(range(0, len(w_diff2)),w_diff2)
#plt.plot(range(0, len(h_diff1)),h_diff1)
#plt.plot(range(0, len(h_diff2)),h_diff2)
##
cv2.imshow('f',f)
cv2.imshow('f_bar',f_bar)
cv2.imshow('f_sub',f_sub)
cv2.imshow('g',g)
cv2.imshow('thresh1',thresh1)
cv2.imshow('thresh2',thresh2)
#
cv2.waitKey(0)


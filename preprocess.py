import cv2
import numpy as np
def maxContrast(img):
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	imgTopHat = cv2.morphologyEx(imgGray, cv2.MORPH_TOPHAT, kernel, iterations = 8)
	imgBlackHat = cv2.morphologyEx(imgGray, cv2.MORPH_BLACKHAT, kernel, iterations = 8)
	result = cv2.add(imgGray, imgTopHat)
	result = cv2.subtract(result, imgBlackHat)
	return result

def getData(imgMaxContrast):
	#areaCharacter
	imgBlur = cv2.GaussianBlur(imgMaxContrast, (7,7), 0)
	imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	imgCanny = cv2.Canny(imgThresh, 100, 200)
	imgCanny = cv2.morphologyEx(imgCanny, cv2.MORPH_OPEN,  kernel)
	imgCannyDilate = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kernel, iterations = 2)
	contours, hierachy = cv2.findContours(imgCannyDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#scanAreaCharacter
	char_x = []
	char_ind_x = {}
	for ind,c in enumerate(contours):
		x,y,w,h = cv2.boundingRect(c)
		if (5*w > h and w > 10):
			cut_char = imgThresh[y:(y+h), x:(x+w)]
			cut_char_resize = cv2.resize(cut_char, (24,24))
			char_standard = cv2.copyMakeBorder(cut_char_resize, 2,2,2,2, cv2.BORDER_CONSTANT)
			if x is char_x:
				x = x + 1
			char_x.append(x)
			char_ind_x[x] = (x, ind, char_standard)
	char_x = sorted(char_x)
	data = []
	for char in char_x:
		char_data = char_ind_x[char][2]
		char_data = np.reshape(char_data, (28,28,1))
		data.append(char_data)
	data = np.array(data)
	return data
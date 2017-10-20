import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy import misc
from matplotlib.colors import LinearSegmentedColormap


#This is  a function written for discrete 2D convolution. 
#It has been developed using the general formula for 2D convolution
#Parameters X,H - H is the filter, while X is the image to be filtered
def convol2d1(X, H):
	s1, s2 = X.shape
	t1, t2 = H.shape
	
	for i in range(s1+1):
		H = np.concatenate((H,np.zeros((1,t2))),0)

	for i in range(t1+1):
		X = np.concatenate((X,np.zeros((1,s2))),0)

	Y = np.zeros(X.shape)
	for n1 in range (s1-1):
		for n2 in range (s2-1):
			l1 = 1
			while (t1-l1 >= 0):
				l2 = 1
				while (t2-l2 >= 0):
					Y[n1][n2] = Y[n1][n2] + X [n1 - l1-1][n2 - l2-1] * H [t1 - l1][t2 - l2]
					l2 = l2+1
				l1 = l1+1

	return Y


#This is  a function written for discrete 2D convolution. 
#It has been developed using a slightly different method of applying the filter
#This function doesn't work for all types of inputs
#Parameters Y,H - H is the filter, while Y is the image to be filtered
def convol2d2(Y,H):
	print(Y.shape,H.shape)
	print(np.average(H))
	if(np.sum(H)!=0):
		H = H/np.sum(H)
	new = Y
	offset = (H.shape[1]-1)/2
	x = offset
	y = offset
	while(y<Y.shape[1]-offset):
		x = offset
		while(x<Y.shape[0]-offset):
			#print(y,x)
			sum = np.sum(Y[x-offset:x+offset+1,y-offset:y+offset+1]*H)
			if(sum<0):
				new[x,y] = 0
			elif(sum>255):
				new[x,y] = 255
			else:
				new[x,y] = sum
			x = x+1
		y = y+1

	return new


if __name__ == "__main__":

	#readinng and displaying the required image
	img = misc.imread('img1.jpg')
	plt.title('Original image')
	plt.imshow(img)
	plt.show()

	cmap = LinearSegmentedColormap.from_list('mycmap', ['blue', 'black'])

	#plot of RGB components of the image
	plt.subplot(2,2,1)
	plt.title('The original image in RGB')
	plt.imshow(img,cmap = 'gray')

	plt.subplot(2,2,2)
	plt.title('Red component')
	plt.imshow(img[:,:,0],cmap = 'Reds')

	plt.subplot(2,2,3)
	plt.title('Green component')
	plt.imshow(img[:,:,1],cmap = 'Greens')

	plt.subplot(2,2,4)
	plt.title('Blue component')
	plt.imshow(img[:,:,2],cmap = cmap)
	plt.show()
	H = np.ones((5,5))
	H1= np.eye(5)

	#Thresholding the image to two binary values (black and white)
	img1 = img[:,:,0]
	threshH = img1[:,:]>180
	img1[threshH] = 255
	img1[-threshH] = 0
	plt.title('Binary colour')
	plt.imshow(img1,cmap = 'gray')
	plt.show()

	#This is a sobel edge detection filter kernel
	H2 = np.array([[ 1,  2,  0,  -2,  -1], [ 4,  8,  0,  -8,  -4], [ 6,  12,  0,  -12,  -6], [ 4,  8,  0,  -8,  -4], [ 1,  2,  0,  -2,  -1]])

	#this is an edge detection filter
	H3 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

	#7x7 gaussian blur kernel
	H4 = np.array([[0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036],
[0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363],
[ 0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446],
[ 0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291],
[ 0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446],
[ 0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363],
[ 0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036]])

	#prewitt's edge detection kernel
	H5 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

	#uniform blur kernel
	H6 = np.ones((7,7))/49

	#sharpening kernel
	H7 = np.array([[0,-1,0],[-1,8,-1],[0,-1,0]])/8
	
	#adding noise to an image 
	#The noise can be effectively removed by applying a 3x3 blurring filter multiple times
	noise_mask = np.random.poisson(img[:,:,0])
	noisy_img = img[:,:,0] + 2*noise_mask
	plt.imshow(noise_mask,cmap = 'gray')
	plt.imshow(noisy_img,cmap = 'gray')
	plt.show()

	a = convol2d1(misc.imresize(img1,0.99),H6)

	plt.subplot(1,3,2)
	plt.title("5x5 smoothening filter (convol2d1)")
	plt.imshow(a,cmap = 'gray')

	plt.subplot(1,3,3)

	#the inbuilt 2d convolution function in scipy
	b = sp.convolve(misc.imresize(img1,0.99),H6)
	
	plt.title("5x5 smoothening filter (inbuilt)")
	plt.imshow(b,cmap = 'gray')
	plt.subplot(1,3,1)
	plt.title("original image")
	plt.imshow(img1,cmap = 'gray')
	
	plt.show()

	'''
	c = convol2d2(misc.imresize(img[:,:,0],0.99),H2)
	#print(a.shape)
	plt.subplot(1,3,2)
	plt.title('5x5 Sobel filter (convol2d2)')
	plt.imshow(c,cmap = 'gray')
	'''
	

	
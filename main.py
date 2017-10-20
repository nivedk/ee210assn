import numpy as np
import math
import random
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile

#two frequencies taken from user as input
v1 = float(input ("enter frequency of signal 1::"))
v2 = float(input ("enter frequency of signal 2::"))

#Angular frequencies calculated
w1 = 2*math.pi*v1
w2 = 2*math.pi*v2

#We've increased the sampling frequency (compared to what was given)
ts = 0.01/max(v1,v2);	

#Amplitudes of the two signals
a1 = 1;
a2 = 1;

#The maximum magnitude of noise to be introduced stochastically into the system
sigma = 0.2

#Filter size
M = 50

#Parameter for delta filter
p = 0.5

#This function performs 2D convolution on given inputs
#Arguements: Y and H (both are 1D arrays that are to be convolved)
#This function returns an array that is the convolution of Y with H
def convol(Y,H):
	s1 = Y.shape[0]
	s2 = H.shape[0]
	FY = []
	
	for i in range(s1+1):
		H = np.append(H,[0])
	for i in range(s2+1):
		Y = np.append(Y,[0])

	i = 0
	k = 0
	for i in range(s1 + s2 + 1) :
		FY = np.append (FY, [0])
		k =0
		while (((i - k) >= 0)) :
			FY[i] = FY[i]+ H[k]*Y[i-k]
			k = k+1
		i = i+1
	return(FY)


#This was an initial version of the convolution code we'd written using a slightly different code, however, it doesn't work
def convol1(Y,H):
	s1= Y.shape[0];
	s2 = H.shape[0];
	FSY = [];
	FY = np.zeros(s1+s2+1);
	s3 = FY.shape[0];

	for i in range(s3):
		for k in range(i-2):
			FY[i] = FY[i]+ H[k]*Y[i-k]

	return(FY)


if __name__ == "__main__":

	#array of numbers (sampling frequency)
	T = np.arange(0,10,ts);

	#the sinusoidal signals that have been worked with and analysed
	S1 = np.array(a1*[math.cos(i*w1) for i in T]).reshape(len(T))
	S2 = np.array(a2*[math.cos(i*w2) for i in T])
	Y1 = np.array(S1 + S2)

	#an array of random number within the range -sigma and sigma
	rand = np.random.uniform(-1*sigma,sigma,Y1.shape[0])
	noisy = Y1+rand

	#plots of initial inputs and superpositions
	plt.subplot(3, 1, 1)
	plt.title('Signal 1 (S1)')
	plt.plot(S1)

	plt.subplot(3, 1, 2)
	plt.title('Signal 2 (S2)')
	plt.plot(S2)

	plt.subplot(3, 1, 3)
	plt.title('S1 + S2')
	plt.plot(Y1)

	plt.show()

	#plot for signal, noise and final input
	plt.subplot(3, 1, 1)
	plt.title('S1 + S2')
	plt.plot(Y1)

	plt.subplot(3, 1, 2)
	plt.title('Noise')
	plt.plot(rand)

	plt.subplot(3, 1, 3)
	plt.title('Signal + noise')
	plt.plot(noisy)

	plt.show()
	
	#definition for box filter
	H1 = 1/M*np.ones(M)
	plt.plot(H1)
	plt.title('Box Filter')
	plt.show()
	
	#definition for triangular filter
	d = 4.0/(M*(M-2))
	H2 = np.zeros(M)
	for i in range(int (M/2)):
		H2 [i] = i * d
		H2 [M-i-1] = H2 [i]
	H2 = np.array(H2)
	plt.plot(H2)
	plt.title('Triangular Filter')
	plt.show()

	#definition for delta filter
	H3 = [ 1 + p, -1*p ]
	markerline, stemlines, baseline = plt.stem(H3, '-.')
	testConH3 = np.array(H3)
	plt.setp(baseline, 'color', 'r', 'linewidth', 2)
	plt.title('Delta filter')
	plt.show ()

	#convolution statements
	FY1 = np.convolve(noisy,H1)
	FY2 = np.convolve(noisy,H2)
	FY3 = np.convolve(noisy,H3)

	#plot statements
	plt.subplot(4,1,1)
	plt.title('noisy input')
	plt.plot(noisy)
	plt.subplot(4,1,2)
	plt.title('filtering with box filter')
	plt.plot(FY1)
	plt.subplot(4,1,3)
	plt.title('filtering with triangular filter')
	plt.plot(FY2)
	plt.subplot(4,1,4)
	plt.title('filtering with delta(differential) filter')
	plt.plot(FY3)

	plt.show()

	#audio file convolutionn hereafter
	#reading the input audio file
	sampFreq, snd = wavfile.read('speech.wav')
	data = snd[:,0]
	#sampling at an interval of 10
	data1 = data[::10]


	#plot for input audio and its sampled part
	plt.subplot (2,1,1)
	plt.plot (data)
	plt.title('original file')
	plt.subplot (2,1,2)
	plt.plot (data1)
	plt.title('sampled')

	plt.show ()
	
	#convolution statements
	testConH1 = convol(data1,np.array(H1))
	testConH2 = convol(data1,np.array(H2))
	testConH3 = convol(data1,np.array(H3))

	#plots for audio clip convolution
	plt.subplot(4,1,1)
	plt.title('Input voice data')
	plt.plot(data1)
	plt.subplot(4,1,2)
	plt.title('Input voice data through box filter')
	plt.plot(testConH1)
	plt.subplot(4,1,3)
	plt.title('Input voice data through triangular filter')
	plt.plot(testConH2)
	plt.subplot(4,1,4)
	plt.title('Input voice data through delta filter')
	plt.plot(testConH3)

	plt.show()

	#writing out the audio files
	wavfile.write("in.wav", 4410, data1)
	wavfile.write("out1.wav", 4410, testConH1)
	wavfile.write("out2.wav", 4410, testConH2)
	wavfile.write("out3.wav", 4410, testConH3)
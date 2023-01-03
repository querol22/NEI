# Welcome! Jump this initial comment section in case you have installed the common libraries in python already.
# pip install numpy
# pip install pandas
# pip install -q ipywidgets
# The process is the same for any missing library I may have forgotten :)

# import (aka. use) the following library 
# "as" is used as a shortcut
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #the pandas module allows us to load DataFrames from external files and work on them.
import math

from scipy.optimize import leastsq
import os
from ipywidgets import IntSlider,interact, FloatSlider, IntRangeSlider, FloatLogSlider, FloatText

# Specify the location and name of the text file
path_raw = 'PraktikumFiles' # directory
raw_data = 'DummyTest.txt' # name of your file
# Define the columns from our txt table
columns = ["Pt.","Frequency","Z'","Z''","Frequency (Hz)","|Z|","theta"]
# Read the txt file
data = pd.read_csv(raw_data,names=columns, sep="\t" or " ", skiprows =6) #skiprows is used to remove the header as parameters

# Parameters to extract individually from the .txt file using the list function
frequency = list(data["Frequency"])
Z_real = list(data["Z'"])
Z_imaginary = list(data["Z''"])
second_frequency = list(data["Frequency (Hz)"])
Z_betrag = list(data["|Z|"])
theta = list(data["theta"])

#theta = float(theta)
#i=0
#for i in theta:
#theta = math.degrees(theta)

#[float(i) for i in theta]
#theta = theta*180/math.pi

#print("length data: ", len(data))
#print("length data Freq: ", len(data["Frequency"]))

#lastValue = data["Frequency"].iat[-1]
#print("lastValue "+lastValue)


# Diagram Show-Down!
# 1. Bode Plot

# Font size, bold, etc to be choosen
font = {'size'   : 5}

# Amplitude
plt.subplot(211)
plt.grid(True)
#plt.axes().yaxis.grid()
plt.title('Bode Diagram')
#plt.plot(frequency,Z_betrag,'o')
plt.semilogx(second_frequency,Z_betrag,'o') # plot with log scale on the x-axis
#plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

# Phase
plt.subplot(212)
plt.grid(True)
#plt.axes().yaxis.grid()
#plt.plot(second_frequency,theta,'o')
plt.semilogx(second_frequency,theta,'o') # plot with log scale on the x-axis
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase')
#plt.plot(frequency,Z_imaginary,'o')
plt.rc('font', **font)
plt.show()


# 2. Nyquist Plot
#increase Font size
font = {'size'   : 6}

plt.grid(True)
plt.title('Nyquist Diagram')
#plt.semilogx(real,imag,'o') # plot with log scale on the x-axis
plt.plot(Z_real,Z_imaginary,'o')
plt.xlabel('Re(s)')
plt.ylabel('Im(s)')
plt.rc('font', **font)
plt.show()

#plt.savefig('output_20_def/'+raw_data[file_index]+'_fit.png',bbox_inches='tight',pad_inches=0)
#plt.savefig('output_20_def/'+raw_data[file_index]+'_bode_M_check.png',bbox_inches='tight',pad_inches=0)
#plt.savefig('output_20_def/'+raw_data[file_index]+'_bode_P_check.png',bbox_inches='tight',pad_inches=0)
#fitted_params

# Welcome! Jump this initial comment section in case you have installed the common libraries in python already.
# pip install numpy
# pip install pandas
# pip install -q ipywidgets
# The process is the same for any missing library I may have forgotten :)

# import (aka. use) the following library 
# "as" is used as a shortcut
import numpy as np
import matplotlib
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
data = pd.read_csv(raw_data,names=columns, sep="\t" or " ", skiprows =7, decimal=",") #skiprows is used to remove the header as parameters

# Repeat
raw_data2 = 'EISTest16_10CV_02.txt' # name of your file
raw_data3 = 'EISTest16_10CV_03.txt' # name of your file
raw_data4 = 'EISTest16_10CV_04.txt' # name of your file
raw_data5 = 'EISTest16_10CV_05.txt' # name of your file
raw_data6 = 'EISTest16_10CV_06.txt' # name of your file
#raw_data7 = 'EISTest16_10CV_07.txt' # name of your file

data2 = pd.read_csv(raw_data2,names=columns, sep="\t" or " ", skiprows =7, decimal=",") #skiprows is used to remove the header as parameters
data3 = pd.read_csv(raw_data3,names=columns, sep="\t" or " ", skiprows =7, decimal=",") #skiprows is used to remove the header as parameters
data4 = pd.read_csv(raw_data4,names=columns, sep="\t" or " ", skiprows =7, decimal=",") #skiprows is used to remove the header as parameters
data5 = pd.read_csv(raw_data5,names=columns, sep="\t" or " ", skiprows =7, decimal=",") #skiprows is used to remove the header as parameters
data6 = pd.read_csv(raw_data6,names=columns, sep="\t" or " ", skiprows =7, decimal=",") #skiprows is used to remove the header as parameters
#data7 = pd.read_csv('EISTest16_10CV_07.txt',names=columns, sep="\t" or " ", skiprows =6) #skiprows is used to remove the header as parameters

print("first type: ")
print(type(data["Frequency"]))

# Parameters to extract individually from the .txt file using the list function
frequency = np.asarray((data["Frequency"]))
Z_real = np.asarray(data["Z'"])
Z_imaginary = np.asarray(data["Z''"])
second_frequency = np.asarray(data["Frequency (Hz)"])
Z_betrag = np.asarray((data["|Z|"]))
theta = np.asarray(data["theta"])

print("first 3 frequencies: ")
test = '{0:,.2f}'.format(float(frequency[0]))
#test = type((frequency[0]).format(float()))
print(test)
print(type(test))
#print(round(test, 2))
#floatfrequency = float(frequency[1])
#print(floatfrequency)
#print(np.asarray(frequency[2]))
print("first 3 Z: ")
print(Z_betrag[0])
print(Z_betrag[1])
print(Z_betrag[2])

# Repeat
frequency2 = np.asarray(data2["Frequency (Hz)"])
frequency3 = np.asarray(data3["Frequency (Hz)"])
frequency4 = np.asarray(data4["Frequency (Hz)"])
frequency5 = np.asarray(data5["Frequency (Hz)"])
frequency6 = np.asarray(data6["Frequency (Hz)"])

Z_betrag2 = np.asarray(data2["|Z|"])
Z_betrag3 = np.asarray(data3["|Z|"])
Z_betrag4 = np.asarray(data4["|Z|"])
Z_betrag5 = np.asarray(data5["|Z|"])
Z_betrag6 = np.asarray(data6["|Z|"])

# HOW TO CHANGE theta FROM rad to deg ???
#theta = float(theta)
#i=0
#for i in theta:
#theta = math.degrees(theta)
#[float(i) for i in theta]
#theta = theta*180/math.pi


# HOW TO
# Obtain value for R1, R2 & C. R1 is in series with the parallel circuit bt R2 and C ???


###
# Diagram Show-Down!
###
# 1. Bode Plot
# esto es correcto? creo que no. la magnitud no está en db ni la phase en deg.
# Amplitude
plt.subplot(211)
plt.grid(True)
plt.title('Bode Diagram')
plt.loglog(frequency, Z_betrag,'o') # plot with log scale on the x-axis
### ADD log also to the yachse
plt.xlabel('Frequency [Hz]')
plt.xscale("log")
plt.ylabel('Magnitude [dB]')
plt.margins(0.1, 0.1)
matplotlib.ticker.MultipleLocator() #??
#plt.gca().invert_xaxis() # invert axis as it is in undescended order
#plt.xticks([1, 4, 9, 30, len(frequency)-2])
#plt.yticks([1, 4, 9, 30, len(Z_betrag)-2])

# Phase
plt.subplot(212)
plt.grid(True)
plt.semilogx(frequency, theta,'o') # plot with log scale on the x-axis
plt.xlabel('Frequency [Hz]')
plt.xscale("log")
plt.ylabel('Phase')
plt.margins(0.1, 0.1)
matplotlib.ticker.MultipleLocator()
#plt.gca().invert_xaxis()
plt.autoscale(enable=True, axis="x", tight=None)
#plt.xticks([1, 4, 9, 30, len(frequency)-2])
#plt.yticks([1, 4, 9, 30, len(theta)-2])
plt.show()


# 2. Nyquist Plot
#increase Font size
#font = {'size'   : 6}

valueY = Z_imaginary[-1]

fig2, ax2 = plt.subplots()
plt.grid(True)
plt.title('Nyquist Diagram')
#ax2.hlines(y=valueY, xmin=-100000, xmax=1000000, linewidth=2, color='r')
#plt.axhline(y=0, xmin=0, xmax=1, color='r')
ax2.plot(Z_real, Z_imaginary,'o')
plt.xlabel('Re(s)')
plt.ylabel('Im(s)')
plt.margins(0.1, 0.1)
#plt.gca().invert_yaxis()
#plt.autoscale(enable=True, axis="x", tight=None)
#plt.xticks([1, 9, 30, 60, len(Z_real)-2])
#plt.yticks([1, 9, 30, 60, len(Z_imaginary)-2])
plt.show()


# 3. Electrochemical impedance spectroscopy characterization
#axis_font = {'fontname':'Arial', 'size':'14'}

fig3, ax3 = plt.subplots(3,2)
#ax3.title('Electrochemical Impedance Spectroscopy (EIS) characterization')

# Top left plot
ax3[0,0].semilogx(second_frequency, Z_betrag,'o', c='b') # plot with log scale on the x-axis
ax3[0,0].set_title(raw_data)

# Top right plot
ax3[0,1].semilogx(frequency2, Z_betrag2,'o', c='g') # plot with log scale on the x-axis
ax3[0,1].set_title(raw_data2)

# Midle left plot
ax3[1,0].semilogx(frequency3, Z_betrag3,'o', c='r') # plot with log scale on the x-axis
ax3[1,0].set_title(raw_data3)

# Middle right plot
ax3[1,1].semilogx(frequency4, Z_betrag4,'o', c='m') # plot with log scale on the x-axis
ax3[1,1].set_title(raw_data4)

# Bot left plot
ax3[2,0].semilogx(frequency5, Z_betrag5,'o', c='y') # plot with log scale on the x-axis
ax3[2,0].set_title(raw_data5)

# Bot right plot
ax3[2,1].semilogx(frequency6, Z_betrag6,'o', c='c') # plot with log scale on the x-axis
ax3[2,1].set_title(raw_data6)

for ab in ax3.flat:
    ab.set(xlabel='Frequency [Hz]', ylabel='Impedance  $[\Omega]$')
    ab.set_xscale('log')
    ab.xaxis.grid(True, which='minor')
    #ab.invert_xaxis()
    ab.set_yscale('linear')
    ab.yaxis.grid(True, which='minor')
    ab.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
    ab.locator_params(tight=True)

plt.show()
plt.savefig('Electrochemical Impedance Spectroscopy (EIS) characterization'+'_final',bbox_inches='tight',pad_inches=0)

#fitted_params
### ???
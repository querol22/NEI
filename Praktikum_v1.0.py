### Welcome! Jump this initial comment section in case this isnt your first time using python (installing libraries)
# pip install numpy
# pip install pandas
# pip install scipy
# pip install -q ipywidgets
# The process is the same for any missing library I may have forgotten :)
### ### ###

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

#########
### Part 0 ###
### Data ###
#########
# Specify the location and name of the text file
path_raw = 'PraktikumFiles' # directory
raw_data = 'EISTest16_10CV_03.txt' # name of your file
# Define the columns from our txt table
columns = ["Pt.","Frequency","Z'","Z''","Frequency (Hz)","|Z|","theta"]
# Read the txt file
data = pd.read_csv(raw_data,names=columns, sep="\t" or " ", skiprows =7, decimal=",") #skiprows is used to remove the header as parameters

#raw_data = ['EISTest16_10CV_03.txt', 'EISTest16_10CV_02.txt', 'EISTest16_10CV_01.txt', 'EISTest16_10CV_04.txt', 'EISTest16_10CV_05.txt', 'EISTest16_10CV_06.txt']

raw_data_total = [el for el in os.listdir() if el.endswith('.txt')]
print("Raw data total: ")
print(raw_data_total)
# my_files = [el for el in os.listdir() if el.endswith('.txt') ]


# Repeat
raw_data2 = 'EISTest16_10CV_02.txt' # name of your file
raw_data3 = 'EISTest16_10CV_01.txt' # name of your file
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

# Parameters to extract individually from the .txt file using the list function
frequency = np.asarray((data["Frequency"]))
Z_real = np.asarray(data["Z'"])
Z_imaginary = np.asarray(data["Z''"])
second_frequency = np.asarray(data["Frequency (Hz)"])
Z_betrag = np.asarray((data["|Z|"])) # this will be change to Z_modulus
theta = np.asarray(data["theta"])

# Repeat
frequency2 = np.asarray(data2["Frequency (Hz)"])
frequency3 = np.asarray(data3["Frequency (Hz)"])
frequency4 = np.asarray(data4["Frequency (Hz)"])
frequency5 = np.asarray(data5["Frequency (Hz)"])
frequency6 = np.asarray(data6["Frequency (Hz)"])

Z_betrag2 = np.asarray(data2["|Z|"]) # this will be change to Z_modulus
Z_betrag3 = np.asarray(data3["|Z|"]) # this will be change to Z_modulus
Z_betrag4 = np.asarray(data4["|Z|"]) # this will be change to Z_modulus
Z_betrag5 = np.asarray(data5["|Z|"]) # this will be change to Z_modulus
Z_betrag6 = np.asarray(data6["|Z|"]) # this will be change to Z_modulus

#########
### Part 1 - Diagram Show-Down ###
### Visualization of our Data ###
#########

###
# 1. Bode Plot
###
# IMPORTANT: At the time of visualization, it is important to discard the high frequency values. That means to zoom in at the
# Spectrum range below ~10^5.
# Once this is done. We can visualize the constant value from 10^1 to 10^4 (approx.), and having an increase of Magnitude
# Amplitude
plt.subplot(211)
plt.grid(True)
plt.title('Bode Diagram')
plt.semilogx(frequency, Z_betrag,'o') # plot with log scale on the x-axis
plt.xlabel('Frequency [Hz]')
plt.xscale("log")
plt.ylabel('|Z| $[\Omega]$')
plt.yscale("linear")
plt.margins(0.1, 0.1)
matplotlib.ticker.MultipleLocator() 

# Phase - in rad
plt.subplot(212)
plt.grid(True)
plt.semilogx(frequency, theta,'o') # plot with log scale on the x-axis
plt.xlabel('Frequency [Hz]')
plt.xscale("log")
plt.ylabel('Phase [rad]')
plt.margins(0.1, 0.1)
matplotlib.ticker.MultipleLocator()
plt.autoscale(enable=True, axis="x", tight=None)
plt.savefig('output_20_def/'+raw_data+'_bode_normal.png',bbox_inches='tight',pad_inches=0)
plt.show()

###
# Part 1.2 Nyquist Plot
###
fig2, ax2 = plt.subplots()

plt.grid(True)
plt.title('Nyquist Diagram')
ax2.plot(Z_real, -Z_imaginary,'o')
plt.xlabel('Re(Z) $[\Omega]$')
plt.ylabel('Im(Z) $[\Omega]$')
plt.margins(0.1, 0.1)
plt.gca().invert_yaxis()
plt.savefig('output_20_def/'+raw_data+'_Nyquist.png',bbox_inches='tight',pad_inches=0)
plt.show()

###
# Part 1.3 Electrochemical impedance spectroscopy characterization
###
fig3, ax3 = plt.subplots(1,1, figsize=(8,4))

freqs=[second_frequency,frequency2,frequency3,frequency4,frequency5,frequency6]
betrag=[Z_betrag,Z_betrag2,Z_betrag3,Z_betrag4,Z_betrag5,Z_betrag6]

i = 0

for f,b in zip(freqs,betrag):
    i+=1
    ax3.plot(f,b,'o', label=i)
    ax3.legend()
    
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_ylim(10,10**5)
ax3.set_title("Comparison EIS")
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('Impedance  $[\Omega]$')
ax3.grid()

plt.savefig('output_20_def/'+raw_data+'_EIS_Comparison.png',bbox_inches='tight',pad_inches=0)
plt.show()

# ALTERNATIVELY: One can plot the different graphs on each column&row
# # Top left plot
# ax3[0,0].semilogx(second_frequency, Z_betrag,'o', c='b') # plot with log scale on the x-axis
# ax3[0,0].set_title(raw_data)
# # Top right plot
# ax3[0,1].semilogx(frequency2, Z_betrag2,'o', c='g') # plot with log scale on the x-axis
# ax3[0,1].set_title(raw_data2)
# # Midle left plot
# ax3[1,0].semilogx(frequency3, Z_betrag3,'o', c='r') # plot with log scale on the x-axis
# ax3[1,0].set_title(raw_data3)
# # Middle right plot
# ax3[1,1].semilogx(frequency4, Z_betrag4,'o', c='m') # plot with log scale on the x-axis
# ax3[1,1].set_title(raw_data4)
# # Bot left plot
# ax3[2,0].semilogx(frequency5, Z_betrag5,'o', c='y') # plot with log scale on the x-axis
# ax3[2,0].set_title(raw_data5)
# # Bot right plot
# ax3[2,1].semilogx(frequency6, Z_betrag6,'o', c='c') # plot with log scale on the x-axis
# ax3[2,1].set_title(raw_data6)
# for ab in ax3.flat:
#     ab.set(xlabel='Frequency [Hz]', ylabel='Impedance  $[\Omega]$')
#     ab.set_xscale('log')
#     ab.xaxis.grid(True, which='minor')
#     ab.set_yscale('log')
#     ab.yaxis.grid(True, which='minor')
#     ab.set_xlim((0.05,100000))
#     #ab.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots. # Comment this line to Add the inner labels (repetition of x or y axes)
#     ab.locator_params(tight=True)
# plt.autoscale(enable=True, axis='y', tight=True)
#plt.subplots_adjust(hspace=0.5, wspace=0.5)

#########
### Part 2 ###
### Fitting our Data ###
#########
# The value of the impedance = real part - imaginary part
Z = Z_real -1j*Z_imaginary

# start and end from our .txt files (d.h. 71 values)
start = 0
end = 70

# Define the bot and top limits, if neccesary
fe = second_frequency[start:end]
Z_cut=Z[start:end]

# Define the parallel circuit bt R || Rc
def r_rc_imp(inputs, fe):
    Rs,Rc,Q,n=inputs
    w = 2*np.pi*fe # angular frequency, w = 2 pi f
    
    a = Rc
    b = 1./(Q*(w*1j)**n)
    par = 1./(1/a + 1/b)

    return Rs + par

# Define the residuals
def residuals(p,y,x):
    w=2*np.pi*fe
    Rs,Rc,Q,n=p
    a = Rc
    b = 1./(Q*(w*1j)**n)
    par = 1./(1/a + 1/b) 
    
    Zi = Rs + par
    
    c=(y-Zi)
    return c.real**2+c.imag**2

# These are the tweaking parameters, depending on your values, you may want to
# adapt these values.
guess=np.array([21000, 6000,  10e-7,  0.9]) # Rs = 11000 , Rc = 100000, C = 10e-5, n = 0.8 CP value
prova = r_rc_imp(guess, fe)

# r_rc fit (aka fitting)
fitted_params, flag = leastsq(residuals, guess, args=(Z_cut,fe))
Zfit = r_rc_imp(fitted_params, fe)

# Compute Modulus and phase
re = np.real(Zfit)
im = np.imag(Zfit)
Zfit_mod = np.sqrt(((re*re) + (im*im)))
Zfit_pha = np.arctan2(-im,re) *(180/np.pi)

#########
### Part 3 - Diagram Show-Down ###
### Visualization of Fitting our data ###
#########
# Some data results may have been outdated, giving a wrong value for the imaginary impedance (it should be close to 0 for lower frequencies). So in the Nyquist plots, the fitting does not work.
# This may occur on your graphs as well, if the experiment settup failed, which is pretty common.
plt.title('Nyquist Diagram')
plt.plot(np.real(Z)[start:end], -np.imag(Z)[start:end], 'o-', label='Previous result Z')
plt.plot(np.real(prova), -np.imag(prova), 'o',label='First Approach')
plt.xlabel(r'Re(Z) $[\Omega]$', fontsize=18)
plt.ylabel(r'-Im(Z) $[\Omega]$', fontsize=18)
plt.legend(fontsize=15)
plt.savefig('output_20_def/'+raw_data+'_firstImage.png',bbox_inches='tight',pad_inches=0)
plt.show()

plt.title('Nyquist Diagram')
plt.plot(np.real(Z_cut), -np.imag(Z_cut),'o-',label='Previous result Z')
plt.plot(np.real(Zfit), -np.imag(Zfit),'o',label='Fit Z')
plt.xlabel(r'Re(Z) $[\Omega]$',fontsize=18)
plt.ylabel(r'-Im(Z) $[\Omega]$',fontsize=18)
plt.legend(fontsize=15)
plt.savefig('output_20_def/'+raw_data+'_fit.png',bbox_inches='tight',pad_inches=0)
plt.show()

plt.title('Bode Magnitude Diagram')
plt.semilogx(second_frequency[start:end], Z[start:end], label='Previous result Z modulus') # modulus = 
plt.semilogx(second_frequency[start:end], Zfit_mod,'o',label= 'Fit Z modulus')
plt.xlabel('Frequency [Hz]',fontsize=18)
plt.ylabel(r'|Z| $[\Omega]$',fontsize=18)
plt.legend(fontsize=15)
plt.savefig('output_20_def/'+raw_data+'_bode_Magnitude_check.png',bbox_inches='tight',pad_inches=0)
plt.show()

plt.title('Bode Phase Diagram')
plt.semilogx(second_frequency[start:end], theta[start:end], label='Previous result Z phase')
plt.semilogx(second_frequency[start:end], Zfit_pha,'o', label='Fit Z phase')
plt.xlabel('Frequency [Hz]',fontsize=18)
plt.ylabel(r'Phase(Z) $[º]$',fontsize=18)
plt.legend(fontsize=15)
plt.savefig('output_20_def/'+raw_data+'_bode_Phase_check.png',bbox_inches='tight',pad_inches=0)
plt.show()

# Read and compare in the output terminal the fitting values:
print("Output values for the fitting are: ")
print(fitted_params)

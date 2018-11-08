import numpy as np
import pandas as pd
import glob 
import os
from matplotlib import pyplot as plt


def getdata(P, A, base, initfile, finfile, makefig=False, XrayRaw=False):
	filestr=str(P + ' ' + str('%.3f'%float(A)) + ' '+ "*.txt")
#Making array of strings to help pandas find all data files
# for a single measured amplitude and location (point)
	all_files=glob.glob(os.path.join(base,P, filestr))
	data=pd.concat((pd.read_csv(f,delimiter='\t') for f in all_files[initfile:finfile]),axis=1)
	
	nr=int(np.shape(data)[0]*np.shape(data)[1]/5) #desired total number of rows
	data.columns=np.tile(('Time','Io','If','J','V'),int(np.shape(data)[1]/5))
	
#Making time array
	dt=data.iloc[1,0]
	time=np.arange(0,nr)[:,]*dt
	time.resize(nr,1)

#Grab each signal into distinct arrays
	Io=np.array(data['Io'])
	If=np.array(data['If'])
	J=np.array(data['J'])
	V=np.array(data['V'])

#Reshape each array into single column arrays
	Io.resize(nr,1)
	If.resize(nr,1)
	J.resize(nr,1)
	V.resize(nr,1)
	
	if makefig==True:
		fig=plt.figure(figsize=(6.5,5))
		ax1 = fig.add_subplot(211)
		ax1.plot(time, V,time,J)

		ax2=fig.add_subplot(212)
		ax2.plot(time,Io,time,If)

		plt.show()
		
#Keeping option to report raw X-ray data, but it's usually uneccesary
	if XrayRaw==True:
		return time,Io, If, J, V
	else:
		Ir= If/Io
		return time,Ir, J, V
		
def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def fftbin(freqin,freqlist, Ns,dt, FFTtype, harmonics):
    tmeas=np.ceil(Ns*dt)
        
    if FFTtype=="Real" or FFTtype=="Imag":
        mid=np.size(freqlist)/2
        bins=np.tile(mid,(10))
        for i in range(0,harmonics):
            bins[2*i:2*i+2]=np.array(harm_switch(i+1,mid,tmeas,freqin))
        return bins
    else:
        mid=np.size(freqlist)/4
        bins=np.tile(mid,(10,2))
        for i in range(0,harmonics):
            bins[2*i:2*i+2,0]=np.array(harm_switch(i+1,mid,tmeas,freqin))
            bins[2*i:2*i+2,1]=np.array(harm_switch(i+1,mid*2,tmeas,freqin))
        return bins	
		
#making switch case for finding harmonics fft bins

def first(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-freqin*tmeas)
    ind2=int(mid+freqin*tmeas)
    return ind1, ind2

def second(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-2*freqin*tmeas)
    ind2=int(mid+2*freqin*tmeas)
    return ind1, ind2

def third(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-3*freqin*tmeas)
    ind2=int(mid+3*freqin*tmeas)
    return ind1, ind2

def fourth(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-4*freqin*tmeas)
    ind2=int(mid+4*freqin*tmeas)
    return ind1, ind2

def fifth(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-5*freqin*tmeas)
    ind2=int(mid+5*freqin*tmeas)
    return ind1, ind2
    
switcher={
    1:first,
    2:second,
    3:third,
    4:fourth,
    5:fifth
}

def harm_switch(arg,mid,tmeas,freqin):
    func=switcher.get(arg, "Wrong")
    
    if func=="Wrong":
        return "Invalid Harmonic"
    else:
        return func(mid,tmeas,freqin)
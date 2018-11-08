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
		return Io, If, J, V, time
	else:
		Ir= If/Io
		return Ir, J, V, time
	
	
	
	
	
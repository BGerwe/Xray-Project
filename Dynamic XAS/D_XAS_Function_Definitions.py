import numpy as np
import pandas as pd
import glob 
import os
from matplotlib import pyplot as plt
import matplotlib as mpl

#Small, basic functions

def CalcEta(data,Rohm=63.97,Zparams=[63.97,47.43,0.0066108,.932]):
    R1=Zparams[0]
    R2=Zparams[1]
    Q=Zparams[2]
    a=Zparams[3]
    
    freq=1/(data[0,0,-1]+data[0,0,1])
    
    V=data[0,3,:,None].T
    I=data[0,2,:,None].T
#     eta=np.real(V)*np.abs(R2/(R1*(1+(2*np.pi*freq*1j*Q)**a*R2)+R2))
    eta=V-I*Rohm
    data=np.concatenate((data,np.tile(eta,(data.shape[0],1,1))),axis=1)
    
    return data

def colorfun(V,Max=1):
#     print(V, Max)
    if V>0:
        R=V/Max
        G=0
        B=0
    else:
        R=0
        G=0
        B=-V/Max
    return (R,G,B)

def Dawsonapp(b,fa,data):    
    dataD=np.array(data[0,:,:],ndmin=3)
    Dfunc=np.array(np.exp(-(fa*data[0,:,:]/b)**2),ndmin=3)
    dataD=data[1:,:,:]*Dfunc
    dataD=np.append(np.array(data[0,:,:],ndmin=3),dataD,axis=0)
    #print(np.shape(dataD),dataD[0,:,0],Dfunc)
    return dataD, Dfunc

def decimate(data,tarr=np.array([]), decfac=1):
    reshaped=data.reshape(int(data.size/decfac),decfac)
    decimated=np.mean(reshaped,axis=1)
#     print('dec',np.isnan(decimated))
#     print('resh',np.isnan(reshaped))

    if tarr.size>0:
        tarre=tarr[::decfac]
        del data, tarr
        return decimated, tarre
    else:
        del data
        return decimated

#function for getting the bin index of harmonic peaks in the fft
#Will return array of these indices depending on format of fft passed
#For "single" ffts (i.e. [-freq Re/Im -> +freq Re/Im]) will give
#[-first +first -second +second -third +third -fourth +fourth -fifth +fifth]
#For "combined ffts (i.e. [-freq Re +freq Re -freq Im +freq Im]) will give
#two columns of indices in form of "single" fft
def fftbin(freqin,freqlist, Ns,dt, FFTtype, harmonics):
#     tmeas=np.ceil(Ns*dt)
    tmeas=Ns*dt
        
    if FFTtype=="Real" or FFTtype=="Imag":
        mid=np.size(freqlist)/2
        bins=np.tile(mid,(2*harmonics))
        for i in range(0,harmonics):
            bins[2*i:2*i+2]=np.array(harm_switch(i+1,mid,tmeas,freqin))
        return bins
    elif FFTtype=="Combo":
        mid=np.size(freqlist)/4
        bins=np.tile(mid,(2*harmonics,2))
        for i in range(0,harmonics):
            bins[2*i:2*i+2,0]=np.array(harm_switch(i+1,mid,tmeas,freqin))
            bins[2*i:2*i+2,1]=np.array(harm_switch(i+1,mid*2,tmeas,freqin))
        return bins
    else:
        print('Invalid FFT type selection')
        return

def getfft(data, tarr=np.array([])):
    datafft=np.zeros(np.shape(data))+1j*0
    Ns=np.shape(data)[1]
    
    if tarr.size>0:
        dt=tarr[1]
        freq=np.fft.fftshift(np.fft.fftfreq(Ns,dt))
        datafft=np.fft.fftshift(np.fft.fft(data,axis=1)/(Ns/2),axes=1)#Ns)
        del data, tarr
        return datafft, freq
    else:
        datafft=np.fft.fftshift(np.fft.fft(data)/(Ns/2))#Ns)
        del data
        return datafft

#Function to evaluate switch for finding fft bins
def harm_switch(arg,mid,tmeas,freqin):
    func=switcher.get(arg, "Wrong")
    
    if func=="Wrong":
        return "Invalid Harmonic"
    else:
        return func(mid,tmeas,freqin)

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

#Converting numbers from polar representation to complex representation
def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def sigavg(data, freq,tarr=np.array([]), sampf=50000, cycle=1, decfac=1):
    tf=data.size/sampf
    wfm=tf*freq
    wfmck=np.remainder(wfm,cycle)
    if wfmck>0.0:
        return 'Data not integer cycles'
    else:
        datare=data.reshape(int(wfm),int(sampf/freq))
        datavg=np.mean(datare,axis=0)
    if tarr.size>0:
        tarre=tarr[:int(sampf/freq)]
        del data, tarr
        return datavg, tarre
    else:
        del data
        return datavg
   
    
def submean(data):
    datare=data-np.mean(data)
    del data
    return datare
	
def fullanalyze(data, fa, dec, PhsCor="VPhs",Rohm=63.97,plotFFT=False,IRR=100):
    #Reads data, subtracts mean from current (J) and voltage (V) signals but not from X-ray ratio (Ir)
    #Adjusts phase of all signals to mimic a sine wave
    #Decimates by a factor "dec" to reduce sampling rate (e.g. 5000 samples becomes 50 with dec=100)
    #Signal averages down to one period of frequency = fa after decimating signal

    Ir=np.array(data['If']/data['Io'])
    
    #Finds any infinite number from calculating Ir and makes selection array 
#     delarr=np.isfinite(Ir)
#     Ir=Ir[delarr]
    
    Imean=np.mean(Ir)
    Iomean=np.mean(data['Io'])
    Ifmean=np.mean(data['If'])
    datan=np.array([data['Time'],submean(Ir),submean(data['J']/IRR),submean(data['V']),
                    submean(data['Io']),submean(data['If'])])

    if PhsCor=="EtaPhs":
        eta=datan[3,:,None]-datan[2,:,None]*Rohm
        datan=np.concatenate((datan,eta.T),axis=0)
        
        
    Ns=datan[0,:].size
    dt=datan[0,1]

    datafft, freq=getfft(datan[1:,:],datan[0,:])
    fftang=np.angle(datafft)
    fftmag=np.abs(datafft)
    bin1=fftbin(fa,freq,Ns,dt,"Real",1)
    bina=int(bin1[1])
    
    
        
    angadj=fftang[2,int(bin1[1])]+np.pi/2 #Voltage phase correction
    angadj2=fftang[0,int(bin1[1])]+np.pi/2 #Ir phase correction
    angadj3=fftang[1,int(bin1[1])]+np.pi/2 #J phase correction
    
    if PhsCor=="EtaPhs":
        angadj4=fftang[5,int(bin1[1])]+np.pi/2 #Eta phase correction

    fftang2=fftang.copy()

    
    if PhsCor=="VPhs":
        fftang2[:,int(Ns/2):]=fftang[:,int(Ns/2):]-angadj #adjusting positive frequencies
        fftang2[:,:int(Ns/2)]=fftang[:,:int(Ns/2)]+angadj #adjusting negative frequencies
    elif PhsCor=="IndivPhs":
        fftang2[0,int(Ns/2):]=fftang[0,int(Ns/2):]-angadj2 #adjusting positive frequencies for Ir
        fftang2[0,:int(Ns/2)]=fftang[0,:int(Ns/2)]+angadj2 #adjusting negative frequencies for Ir
        fftang2[3:5,int(Ns/2):]=fftang[3:5,int(Ns/2):]-angadj2 #adjusting positive frequencies for Io and If
        fftang2[3:5,:int(Ns/2)]=fftang[3:5,:int(Ns/2)]+angadj2 #adjusting negative frequencies for Io and If
        fftang2[2,int(Ns/2):]=fftang[2,int(Ns/2):]-angadj #adjusting positive frequencies for V
        fftang2[2,:int(Ns/2)]=fftang[2,:int(Ns/2)]+angadj #adjusting negative frequencies for V
        fftang2[1,int(Ns/2):]=fftang[1,int(Ns/2):]-angadj3 #adjusting positive frequencies for J
        fftang2[1,:int(Ns/2)]=fftang[1,:int(Ns/2)]+angadj3 #adjusting negative frequencies for J
    elif PhsCor=="EtaPhs":
        fftang2[:,int(Ns/2):]=fftang[:,int(Ns/2):]-angadj4 #adjusting positive frequencies
        fftang2[:,:int(Ns/2)]=fftang[:,:int(Ns/2)]+angadj4 #adjusting negative frequencies
    else:
        print(PhsCor+ ' is an invalid selection for PhsCor variable.')
        return
        
    
    datafft2=P2R(fftmag,fftang2)

    print('Z before Phs',datafft[2,int(bin1[1])]/datafft[1,int(bin1[1])])
    print('Z after Phs',datafft2[2,int(bin1[1])]/datafft2[1,int(bin1[1])])
    
    if plotFFT==True:
        fig=plt.figure(figsize=(6,4))
        ax1=fig.add_subplot(121)

        ax1.plot(freq,np.real(datafft[2,:]),freq,np.imag(datafft[2,:]))
        ax1.set_xlim(-1.1*fa,1.1*fa)
        ax1.set_title('V Before Phs adjust')
        ax1.legend(['Re','Im'])
        ax1=fig.add_subplot(122)

        ax1.plot(freq,np.real(datafft2[2,:]),freq,np.imag(datafft2[2,:]))
        ax1.set_xlim(-1.1*fa,1.1*fa)
        ax1.set_title('V After Phs adjust')
        ax1.legend(['Re','Im'])
        # ax1.set_ylim(0,.001)
        plt.show()
        
        fig=plt.figure(figsize=(6,4))
        ax1=fig.add_subplot(121)

        ax1.plot(freq,np.real(datafft[1,:]),freq,np.imag(datafft[1,:]))
        ax1.set_xlim(-1.1*fa,1.1*fa)
        ax1.set_title('I Before Phs adjust')
        ax1.legend(['Re','Im'])
        ax1=fig.add_subplot(122)

        ax1.plot(freq,np.real(datafft2[1,:]),freq,np.imag(datafft2[1,:]))
        ax1.set_xlim(-1.1*fa,1.1*fa)
        ax1.set_title('I After Phs adjust')
        ax1.legend(['Re','Im'])
        # ax1.set_ylim(0,.001)
        plt.show()
        print(bin1)
        print('Before Phs Adjust',datafft[:,int(bin1[0])],datafft[:,int(bin1[1])],
              '\n After PHs Adjust',datafft2[:,int(bin1[0])],datafft2[:,int(bin1[1])])

    dataffti=np.fft.ifft(np.fft.ifftshift(datafft2,axes=1)*Ns/2)

    t=datan[0,:]
    Ir=dataffti[0,:]
    V=dataffti[1,:]
    J=dataffti[2,:]
    Io=dataffti[3,:]
    If=dataffti[4,:]
    if PhsCor=="EtaPhs":
        Et=dataffti[5,:]

    # fa=0.5
    # dec=2000
    fs=50000

    Irdec,tdec=decimate(Ir,tarr=t,decfac=dec)
    Iav,tav=sigavg(Irdec, freq=fa,tarr=tdec,sampf=fs/dec,decfac=dec)
    Vav=sigavg(decimate(V,decfac=dec),freq=fa,sampf=fs/dec,decfac=dec)
    Jav=sigavg(decimate(J,decfac=dec),freq=fa,sampf=fs/dec,decfac=dec)
    Ioav=sigavg(decimate(Io,decfac=dec),freq=fa,sampf=fs/dec,decfac=dec)
    Ifav=sigavg(decimate(If,decfac=dec),freq=fa,sampf=fs/dec,decfac=dec)
    if PhsCor=="EtaPhs":
        Etav=sigavg(decimate(Et,decfac=dec),freq=fa,sampf=fs/dec,decfac=dec)
    
#     print(angadj)#,angadj2,angadj3)
#     print(fftang2[0,(bina-2):(bina+2)],fftang2[1,bina-2:bina+2],fftang2[2,bina-2:bina+2])
#     print(fftang2[0,int(bin1[1])],fftang2[1,int(bin1[1])],fftang2[2,int(bin1[1])])
    Iav=Iav+Imean
    Ioav=Ioav+Iomean
    Ifav=Ifav+Ifmean
    
    
    if PhsCor=="EtaPhs":
        datav=np.array([[tav,Iav,Vav,Jav,Ioav,Ifav,Etav]])
    else:
        datav=np.array([[tav,Iav,Vav,Jav,Ioav,Ifav]])
    del data
    return datav


def partanalyze(data, fa, dec):
    #Reads data, subtracts mean from current (J) and voltage (V) signals but not from X-ray ratio (Ir)
    #Decimates by a factor "dec" to reduce sampling rate (e.g. 5000 samples becomes 50 with dec=100)
    #Signal averages down to one period of frequency = fa after decimating signal
    
    Ir=np.array(data['If']/data['Io'])
    t=np.array(data['Time'])
    J=submean(np.array(data['J']))
    V=submean(np.array(data['V']))
    
    fs=50000
    
    Irdec,tdec=decimate(Ir,tarr=t,decfac=dec)
    Iav,tav=sigavg(Irdec, freq=fa,tarr=tdec,sampf=fs/dec,decfac=dec)
    Vav=sigavg(decimate(V,decfac=dec),freq=fa,sampf=fs/dec,decfac=dec)
    Jav=sigavg(decimate(J,decfac=dec),freq=fa,sampf=fs/dec,decfac=dec)
    
    datav=np.array([[tav,Iav,Vav,Jav]])
    del data
    return datav

def readolddata(path, decfac, fa,NumCh=4):
    fsamp=50000
    samplen=int(fsamp/(decfac*fa))

    data=np.genfromtxt(path, delimiter='  ', dtype=complex,autostrip=True)

    datdum=pd.DataFrame(data)
    datdum.dropna(inplace=True)
    data=np.array(datdum)
    dum2d=data.copy()
    
    numE=int(data.shape[0]/samplen)
    dum2d=dum2d.reshape(data.shape[0],int(dum2d.size/data.shape[0]))

    del data
    dum3d=dum2d.reshape([samplen,numE,NumCh]).transpose(1,2,0) 
    return dum3d

def SavebyVoltage(data,Energies,Estart,Efinish,directory,fHead,suff):
    
    for n in np.r_[0:data.shape[2]]:
        svname=directory+fHead+str(n)+"_"+str(int(np.round(np.real(data[0,3,n]),3)*1000))+"_mV "+suff
        print(svname)
        Edat=np.concatenate((data[:,:,n],Energies[Estart-1:Efinish,None]),axis=1)
        svdat=np.roll(Edat,1,axis=1)
        np.savetxt(svname, svdat,delimiter='\t',fmt='%.8e')
        
def SavebyEta(data,Energies,Estart,Efinish,directory,fHead,suff,EtaInd):
    
    for n in np.r_[0:data.shape[2]]:
        svname=directory+fHead+str(n)+"_"+str(int(np.round(np.real(data[92,EtaInd,n]),3)*1000))+"_mV "+suff
        print(svname)
        Edat=np.concatenate((data[:,:,n],Energies[Estart-1:Efinish,None]),axis=1)
        svdat=np.roll(Edat,1,axis=1)
        np.savetxt(svname, svdat,delimiter='\t',fmt='%.8e')
      

## Plotting functions	  
def plotXANES(Edata,data,start=0,stop=0,marker='.',in_adjs=[0,0,.1,0,0,.02,0,0],
             startE=7705,stopE=7730,startInE=7718,stopInE=7720,size=(9,6),UseEta=False):
    in_x1adj=in_adjs[0]
    in_x2adj=in_adjs[1]
    in_xint =in_adjs[2]
    in_y1adj=in_adjs[3]
    in_y2adj=in_adjs[4]
    in_yint =in_adjs[5]
    in_xlat=in_adjs[6]
    in_yvert =in_adjs[7]

    title_font = {'fontname':'Arial', 'size':'10', 'color':'black', 'weight':'normal'}
    mpl.rcParams['xtick.labelsize']=10
    mpl.rcParams['ytick.labelsize']=10
    mpl.rcParams['legend.fontsize']=10
    mpl.rcParams['axes.labelsize']=10

#     if UseEta==True:
#         data=CalcEta(data)
#         print(np.max(data[92,:,:]))
    
    AbsMax=np.max(data[:,1,0])
    AbsMin=np.round(np.real(data[np.argwhere(Edata[:,0]>=startE)[0][0],1,0]),2) #np.min(data[:,1,0])
    
    if UseEta==False:
        MaxV=int(round(np.max(data[0,3,:])*1000,0))
        MinV=int(round(np.min(data[0,3,:])*1000,0))
    else:
        MaxV=int(round(np.max(data[:,-1,:])*1000,0))
        MinV=int(round(np.min(data[:,-1,:])*1000,0))
    
    
    ## Plotting full RDF
    fig=plt.figure(constrained_layout=False,figsize=size)
    gs=fig.add_gridspec(5,20)
    f_ax1=fig.add_subplot(gs[:,:-1])
    for n in range(start,data.shape[2]-stop):
        if UseEta==False:
            f_ax1.plot(Edata[:data.shape[0],n],data[:,1,n].T,
                     color=(colorfun(data[0,3,n],np.max(np.abs(data[0,3,:])))),
                     linestyle='-',linewidth=.3,marker='',markersize=3,
                       label=str(str(data[0,3,n])+' mV'))
        else:
            f_ax1.plot(Edata[:data.shape[0],n],data[:,1,n].T,
                     color=(colorfun(data[0,-1,n],np.max(np.abs(data[0,-1,:])))),
                     linestyle='-',linewidth=.3,marker='',markersize=3,
                       label=str(str(data[0,3,n])+' mV'))
    f_ax1.set(xlim=[startE,stopE],ylim=[.97*AbsMin,1.05*AbsMax])#,yticks=[])
    f_ax1.set_ylabel(r'Absorption  / a.u.',**title_font)
    f_ax1.set_xlabel(r'Energy  /  eV',**title_font)
    # ax1minLoc=mpl.ticker.MultipleLocator(.5)
    # f_ax1.xaxis.set_minor_locator(ax1minLoc)


    ## Plotting RDF inset
    f_ax2=fig.add_subplot(gs[0:3,3:11])    
    for n in range(start,data.shape[2]-stop):
        if UseEta==False:
            f_ax2.plot(Edata[:data.shape[0],n],data[:,1,n].T,
                     color=(colorfun(data[0,3,n],np.max(np.abs(data[0,3,:])))),
                     linestyle='-',linewidth=.5,marker=marker,markersize=3,
                       label=str(str(data[0,3,n])+' mV'))
        else:
            f_ax2.plot(Edata[:data.shape[0],n],data[:,1,n].T,
                     color=(colorfun(data[0,-1,n],np.max(np.abs(data[0,-1,:])))),zorder=n,
                     linestyle='-',linewidth=.5,marker=marker,markersize=3,
                       label=str(str(data[0,3,n])+' mV'))
    f_ax2.set(xlim=[startInE,stopInE])

    ## Choosing plot limits for inset

    xind1=np.argwhere(Edata[:,0]>=startInE)[0]
    xind2=np.argwhere(Edata[:,0]<=stopInE)[-1]

    x1=round(Edata[xind1,0][0])
    x2=round(Edata[xind2,0][0])

    y1=round(data[xind1,1,0][0]*.95,2)
    y2=round(data[xind2,1,0][0]*1.05,2)


    f_ax2.set(xlim=[x1+in_x1adj,x2+in_x2adj], xticks=np.arange(x1+in_x1adj,x2+in_x2adj,in_xint),
              ylim=[y1+in_y1adj,y2+in_y2adj], yticks=np.arange(y1+in_y1adj,y2+in_y2adj,in_yint))

    f_ax2.set(xlim=[x1+in_xlat+in_x1adj,x2+in_xlat+in_x2adj], 
              xticks=np.arange(x1+in_x1adj,x2+in_x2adj,in_xint),
              ylim=[y1+in_yvert+in_y1adj,y2+in_yvert+in_y2adj], 
              yticks=np.arange(y1+in_y1adj,y2+in_y2adj,in_yint))

    #Plotting colorbar
    f_ax3=fig.add_subplot(gs[:3,-1])
    cdict = {'red':   [(0.0, 0.0, 0.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 1.0, 1.0)],

                     'green': [(0.0, 0.0, 0.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 0.0, 0.0)],

                     'blue':  [(0.0, 1.0, 1.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 0.0, 0.0)],}

    cmap_name = 'my_list'
    cm = mpl.colors.LinearSegmentedColormap(cmap_name, cdict, N=100)
    norm = mpl.colors.Normalize(vmin=MinV, vmax=MaxV)
    cb1=mpl.colorbar.ColorbarBase(f_ax3, cmap=cm,norm=norm,orientation='vertical')
    cb1.set_ticks([MinV,0,MaxV])
    cb1.set_label('mV',rotation=0, labelpad=-3, verticalalignment='center')
    
    return fig


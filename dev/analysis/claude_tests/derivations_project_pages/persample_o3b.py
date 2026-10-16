import sys, numpy as np, h5py, lal, lalsimulation as ls
from scipy.signal import welch
fs=2048.; dt=1/fs; flow=20.
fn=sys.argv[1]
with h5py.File(fn,'r') as F:
    keys=list(F.keys()); print(keys)
    d=F['H1'][:int(2048*1024)]
f_w,P=welch(d,fs=fs,nperseg=int(4*fs))  # Welch 1967, 4 s segments
hp,hc=ls.SimInspiralChooseTDWaveform(1.4*lal.MSUN_SI,1.4*lal.MSUN_SI,0,0,0,0,0,0,100e6*lal.PC_SI,0,0,0,0,0,dt,flow,flow,lal.CreateDict(),ls.IMRPhenomD)
h=hp.data.data.copy()
n=1<<int(np.ceil(np.log2(len(h)))+1); x=np.zeros(n); x[-len(h)-int(0.5*fs):-int(0.5*fs)]=h
f=np.fft.rfftfreq(n,dt); X=np.fft.rfft(x)
psd=np.interp(f,f_w,P); W=X/np.sqrt(psd*fs/2); W[f<flow]=0; W[f>=fs/2]=0
w=np.fft.irfft(W,n)
# check: real noise whitened the same way has unit variance per sample
m=len(d)//2*2; D=np.fft.rfft(d[:m]); fd=np.fft.rfftfreq(m,dt); Dw=D/np.sqrt(np.interp(fd,f_w,P)*fs/2); Dw[fd<flow]=0
nw=np.fft.irfft(Dw,m); print("whitened-noise variance (band >=20 Hz) %.3f, expected %.3f"%(np.var(nw[int(64*fs):-int(64*fs)]), 1-2*flow/fs))
ipk=np.argmax(np.abs(w)); end=min(ipk+int(0.5*fs),n); seg=w[end-int(4*fs):end]
rt=np.sum(w**2); rw=np.sum(seg**2); print("fraction of SNR^2 from 20 Hz in last 4 s: %.3f"%(rw/rt))
for rho in [4,8]:
    s=seg/np.sqrt(rw)*rho
    print("rho",rho,"rms %.4f"%np.sqrt(np.mean(s**2)),"max|s| %.4f"%np.max(np.abs(s)))

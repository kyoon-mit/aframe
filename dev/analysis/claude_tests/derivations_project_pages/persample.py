import numpy as np, lal, lalsimulation as ls
fs=2048.; dt=1/fs; flow=20.
m1=m2=1.4
hp,hc=ls.SimInspiralChooseTDWaveform(m1*lal.MSUN_SI,m2*lal.MSUN_SI,0,0,0,0,0,0,100e6*lal.PC_SI,0,0,0,0,0,dt,flow,flow,lal.CreateDict(),ls.IMRPhenomD)
h=hp.data.data.copy(); print("len s",len(h)/fs)
# pad to power of two, whiten with aLIGO design PSD
n=1<<int(np.ceil(np.log2(len(h)))+1); x=np.zeros(n); x[-len(h)-int(0.5*fs):-int(0.5*fs)]=h
f=np.fft.rfftfreq(n,dt); X=np.fft.rfft(x)
psd=np.array([ls.SimNoisePSDaLIGOZeroDetHighPower(fi) if fi>=flow else np.inf for fi in f])
psd[0]=np.inf
W=X/np.sqrt(psd*fs/2)   # whitened: unit-variance white noise per sample
W[f<flow]=0
w=np.fft.irfft(W,n)
# check noise normalization: white noise through the same whitening has unit variance
rng=np.random.default_rng(1)
# window: last 4 s ending 0.5 s after peak
ipk=np.argmax(np.abs(w)); end=min(ipk+int(0.5*fs),n); seg=w[end-int(4*fs):end]
rho_tot=np.sqrt(np.sum(w**2)); rho_win=np.sqrt(np.sum(seg**2))
print("fraction of SNR^2 in last 4 s: %.3f"%(rho_win**2/rho_tot**2))
for rho in [4,8]:
    s=seg/rho_win*rho
    print("rho",rho,"rms %.4f"%np.sqrt(np.mean(s**2)),"max|s| %.4f"%np.max(np.abs(s)), "rho/sqrt(N) %.4f"%(rho/np.sqrt(len(s))))

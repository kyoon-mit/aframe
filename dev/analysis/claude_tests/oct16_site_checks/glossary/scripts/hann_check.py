import numpy as np, torch
# Continuous-time check via fine sampling and heavy zero padding.
fs=4096.0; T=1.0; N=int(fs*T)
t=np.arange(N)/fs
for name,w in [('rect',np.ones(N)),('hann_torch_periodic',torch.hann_window(N,dtype=torch.float64).numpy()),('hann_sin2',np.sin(np.pi*t/T)**2)]:
    pad=64*N
    W=np.abs(np.fft.rfft(w,n=pad)); W/=W[0]
    f=np.fft.rfftfreq(pad,1/fs)
    # first local minimum (zero) of |W|
    i=np.argmax((W[1:-1]<W[:-2])&(W[1:-1]<W[2:]))+1
    # highest side lobe beyond first zero
    sl=W[i:int(20*pad/fs)].max()
    # half-power full width
    hp=f[np.argmax(W**2<0.5)]*2
    print(f"{name:20s} first zero at {f[i]:.4f} Hz (x T = {f[i]*T:.4f}); highest side lobe {sl:.4f} ({20*np.log10(sl):.2f} dB); half-power full width {hp:.4f} Hz")
# torch periodic hann equals sin^2(pi n/N)
n=np.arange(N); print('max |torch.hann_window(N) - sin^2(pi n/N)| =', np.abs(torch.hann_window(N,dtype=torch.float64).numpy()-np.sin(np.pi*n/N)**2).max())

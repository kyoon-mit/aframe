import numpy as np
fs, A, sig, f0, ph = 1000.0, 1.0, 3.0, 123.4, 0.7
def crb(N):
    n = np.arange(N); t = n/fs
    arg = 2*np.pi*f0*t + ph
    d = np.stack([np.cos(arg), -A*2*np.pi*t*np.sin(arg), -A*np.sin(arg)])   # d/dA, d/df, d/dphi
    I = d @ d.T / sig**2
    return np.linalg.inv(I)[1, 1]
for N in [1000, 2000, 4000, 8000]:
    c = crb(N); closed = 6*sig**2*fs**2/(np.pi**2*A**2*N*(N**2-1))
    rho = np.sqrt(A**2*N/(2*sig**2))
    print(f"N {N}: CRB sd(f) {np.sqrt(c):.5e} Hz, closed form {np.sqrt(closed):.5e}, sqrt(3)/(pi rho T) {np.sqrt(3)/(np.pi*rho*N/fs):.5e}, rho {rho:.2f}")
r = [np.sqrt(crb(N)) for N in (1000, 8000)]
print("slope of log sd(f) vs log T:", np.log(r[1]/r[0])/np.log(8))

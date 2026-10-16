import numpy as np
from scipy import integrate, optimize
GMsun_c3 = 4.925490947e-6  # s, IAU nominal GM_sun / c^3
def mc(m1,m2): return (m1*m2)**0.6/(m1+m2)**0.2
def tau(M,f): T=M*GMsun_c3; return 5/256*T**(-5/3)*(np.pi*f)**(-8/3)
def ncyc(M,f): T=M*GMsun_c3; return 1/(32*np.pi**(8/3))*T**(-5/3)*f**(-5/3)
print("== A: time from f_low to merger, leading order")
for lab,(m1,m2) in {"1.4+1.4":(1.4,1.4),"1.0+1.0":(1,1),"2.5+2.5":(2.5,2.5),"10+10":(10,10),"30+30":(30,30)}.items():
    M=mc(m1,m2); print(lab, "Mc=%.4f"%M, "tau20=%.3f s"%tau(M,20), "tau10=%.2f s"%tau(M,10), "tau30=%.3f"%tau(M,30), "N20=%.0f"%ncyc(M,20), "N10=%.0f"%ncyc(M,10))
for M in [0.85,1.2,1.8,2.75]:
    print("Mc",M,"tau20=%.2f s"%tau(M,20),"N20=%.0f"%ncyc(M,20))
# check against derivations table: f 4 s before merger, cycles last 4 s
for M,N4 in [(0.85,636),(1.2,513),(1.8,398),(2.75,305)]:
    f4=optimize.brentq(lambda f: tau(M,f)-4, 5, 2000)
    print("Mc",M,"f(4s)=%.1f"%f4,"N4=%.0f"%ncyc(M,f4),"(table %d)"%N4, "N20=%.0f"%ncyc(M,20), "frac before last 4 s=%.4f"%(1-ncyc(M,f4)/ncyc(M,20)))
print("== B: per-sample SNR, N=4*2048")
N=4*2048; print("N",N)
for r in [4,8,50]: print(r, "%.4f"%(r/np.sqrt(N)))
print("== C: populations")
def pl_cdf(x,a,lo,hi):  # density ∝ x^a
    b=a+1; return (x**b-lo**b)/(hi**b-lo**b)
def pl_q(q,a,lo,hi):
    b=a+1; return (lo**b+q*(hi**b-lo**b))**(1/b)
for name,a,hi in [("ours a=-3 [4,50]",-3,50.),("a=-3 [4,inf)",-3,np.inf),("vol a=-4 [4,50]",-4,50.),("vol a=-4 [4,inf)",-4,np.inf)]:
    lo=4.
    if np.isinf(hi):
        cdf=lambda x: 1-(lo/x)**(-a-1); med=lo*2**(1/(-a-1))
    else:
        cdf=lambda x: pl_cdf(x,a,lo,hi); med=pl_q(0.5,a,lo,hi)
    print(name,"median=%.4f"%med,"P(<8)=%.4f"%cdf(8),"P(>16)=%.4f"%(1-cdf(16)),"P(>25)=%.4f"%(1-cdf(25)))
print("uniform [4,50]: median", (4+50)/2, "P(<8)=%.4f"%((8-4)/46), "P(>16)=%.4f"%((50-16)/46))
# bins vs 29 Sep test
edges=[4,6,8,12,16,25,50]; counts=[14476,5021,3589,1238,950,516]; tot=sum(counts)
print("test total",tot, "below 8 frac %.4f"%((counts[0]+counts[1])/tot))
for i in range(6):
    p=pl_cdf(edges[i+1],-3,4,50)-pl_cdf(edges[i],-3,4,50)
    print(edges[i],edges[i+1],"expected %.4f observed %.4f"%(p,counts[i]/tot))
# Monte Carlo check of the ml4gw sampler math
rng=np.random.default_rng(0); u=rng.uniform(size=2_000_000); x=pl_q(u,-3,4,50)
print("MC median %.4f P(<8) %.4f"%(np.median(x),np.mean(x<8)))
print("== D: H1-L1")
try:
    import lal
    h=np.array(lal.CachedDetectors[lal.LHO_4K_DETECTOR].location); l=np.array(lal.CachedDetectors[lal.LLO_4K_DETECTOR].location)
    d=np.linalg.norm(h-l); print("d=%.1f km d/c=%.3f ms"%(d/1e3,d/lal.C_SI*1e3))
except Exception as e: print(e)
print("== E: residual cycles from 20 Hz for chirp-mass error")
for M in [1.2]:
    for eps in [0.0015,0.001,0.01,0.02]:
        dN=ncyc(M,20)-ncyc(M*(1+eps),20)
        print("Mc",M,"eps",eps,"residual cycles 20 Hz->merger %.2f"%dN, " last-4s %.2f"%(ncyc(M,optimize.brentq(lambda f: tau(M,f)-4,5,2000))*(1-(1+eps)**(-5/3))))
print("grid: 100 log-spaced 1.0..2.5 step %.4f%%"%(100*((2.5/1.0)**(1/99)-1)))

import h5py, numpy as np
from sklearn.metrics import roc_auc_score
R='/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630'
D={}
for p in ['snr4_powerlaw','snr8_powerlaw','snr4_uniform']:
    w=h5py.File(f'{R}/{p}/windows.h5'); m=h5py.File(f'{R}/{p}/mf.h5'); names=list(w.attrs['param_names'])
    D[p]=dict(true=w['params/chirp_mass'][:], snr=w['params/snr'][:], ml=w['ml_mean'][:,names.index('chirp_mass')], mls=w['ml_sigma'][:,names.index('chirp_mass')], mf=m['mf_chirp_mass'][:], mfs=m['mf_snr'][:])
def within(d,k,tol,sel):
    return np.mean(np.abs(d[k][sel]-d['true'][sel])/d['true'][sel]<tol)
u=D['snr4_uniform']; s=~np.isnan(u['true'])
print('uniform overall within2: MF %.3f ML %.3f'%(within(u,'mf',0.02,s),within(u,'ml',0.02,s)))
for a,b in [(8,10),(10,12)]:
    k=s&(u['snr']>=a)&(u['snr']<b); print(' uniform',a,b,'MF %.3f ML %.3f'%(within(u,'mf',0.02,k),within(u,'ml',0.02,k)))
# pooled power-law table check
pl={k:np.concatenate([D[p][k] for p in ['snr4_powerlaw','snr8_powerlaw']]) for k in D['snr4_powerlaw']}
s=~np.isnan(pl['true']); print('power-law pooled signals',s.sum())
for a,b in [(4,6),(6,8),(8,10),(10,12),(12,16),(16,25),(25,50)]:
    k=s&(pl['snr']>=a)&(pl['snr']<b)
    print(a,b,k.sum(),'MF',' '.join('%.0f'%(100*within(pl,'mf',t,k)) for t in (0.01,.02,.05,.1)),' ML',' '.join('%.0f'%(100*within(pl,'ml',t,k)) for t in (0.01,.02,.05,.1)))
# ML within 1% in finer bins (all pooled)
al={k:np.concatenate([D[p][k] for p in D]) for k in D['snr4_powerlaw']}; s=~np.isnan(al['true'])
B=[4,5,6,7,8,9,10,12,14,16,20,25,35,50]
print('ML within1 fine bins:',' '.join('%d-%d:%.1f'%(a,b,100*within(al,'ml',0.01,s&(al['snr']>=a)&(al['snr']<b))) for a,b in zip(B[:-1],B[1:])))
print('MF within1 fine bins:',' '.join('%d-%d:%.1f'%(a,b,100*within(al,'mf',0.01,s&(al['snr']>=a)&(al['snr']<b))) for a,b in zip(B[:-1],B[1:])))
for p in D:
    d=D[p]; s=~np.isnan(d['true']); y=s.astype(int)
    print(p,'AUC MF %.3f ML %.3f'%(roc_auc_score(y,d['mfs']),roc_auc_score(y,-d['mls'])))
    if p!='snr8_powerlaw':
        k=s&(d['snr']>=8)&(d['snr']<12); yy=np.r_[np.ones(k.sum()),np.zeros((~s).sum())]
        print('  8-12 AUC MF %.3f ML %.3f'%(roc_auc_score(yy,np.r_[d['mfs'][k],d['mfs'][~s]]),roc_auc_score(yy,-np.r_[d['mls'][k],d['mls'][~s]])))

import h5py, numpy as np
R='/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630'
pops=['snr4_powerlaw','snr8_powerlaw','snr4_uniform']
D={}
for p in pops:
    w=h5py.File(f'{R}/{p}/windows.h5'); m=h5py.File(f'{R}/{p}/mf.h5')
    d={k:w['params/'+k][:] for k in ['chirp_mass','snr','a_1','a_2','tilt_1','tilt_2','mass_1','mass_2','mass_ratio']}
    d['mf']=m['mf_chirp_mass'][:]; d['mf_snr']=m['mf_snr'][:]
    D[p]=d
cat=lambda k:np.concatenate([D[p][k] for p in pops])
true=cat('chirp_mass'); sig=~np.isnan(true)
print('signals',sig.sum(),'background',(~sig).sum())
snr=cat('snr')[sig]; mfs=cat('mf_snr'); mf=cat('mf')[sig]; t=true[sig]
r=mfs[sig]/snr
for lo,hi in [(10,12),(10,50),(12,16),(16,25),(25,50),(10,1e9)]:
    k=(snr>=lo)&(snr<hi)
    print(f'ratio recovered/injected SNR {lo}-{hi}: median {np.median(r[k]):.4f}, mean {r[k].mean():.4f}, n {k.sum()}, 16/84 {np.percentile(r[k],[16,84])}')
# per-pop background
for p in pops:
    d=D[p]; s=~np.isnan(d['chirp_mass']); b=d['mf_snr'][~s]
    print(p,'bkg n',len(b),'median',np.median(b),'mean',b.mean(),'p99',np.percentile(b,99), 'n>30',(b>30).sum(),'n 30-300',((b>=30)&(b<=300)).sum(),'max',b.max())
bk=mfs[~sig]
print('pooled bkg median %.3f mean %.3f p1 %.3f p16 %.3f p84 %.3f p99 %.3f max %.1f n>=30 %d n>300 %d'%(np.median(bk),bk.mean(),*np.percentile(bk,[1,16,84,99]),bk.max(),(bk>=30).sum(),(bk>300).sum()))
print('bkg >20', (bk>20).sum(), '>10',(bk>10).sum(), '>15',(bk>15).sum())
# weak signals score above 7.5, pop snr4_powerlaw as in plot
d=D['snr4_powerlaw']; s=~np.isnan(d['chirp_mass'])
weak=s&(d['snr']>=8)&(d['snr']<12)
print('weak SNR 8-12 (pl4) n',weak.sum(),'frac >7.5',(d['mf_snr'][weak]>7.5).mean(),'median',np.median(d['mf_snr'][weak]))
b=d['mf_snr'][~s]; print('pl4 bkg frac >7.5',(b>7.5).mean())
# median of recovered SNR where injected below 7
for lo,hi in [(4,5),(5,6),(6,7),(7,8)]:
    k=(snr>=lo)&(snr<hi); print('inj',lo,hi,'median rec',np.median(mfs[sig][k]))
# chi_eff
m1=cat('mass_1')[sig]; m2=cat('mass_2')[sig]
ce=(m1*cat('a_1')[sig]*np.cos(cat('tilt_1')[sig])+m2*cat('a_2')[sig]*np.cos(cat('tilt_2')[sig]))/(m1+m2)
err=(mf-t)/t
for lo,hi in [(16,50),(12,50),(25,50)]:
    k=(snr>=lo)&(snr<hi)
    print(f'SNR {lo}-{hi}: n {k.sum()} within1 {np.mean(np.abs(err[k])<0.01):.4f}')
    for name,c in [('|ce|<0.1',np.abs(ce)<0.1),('ce<-0.1',ce<-0.1),('ce>0.1',ce>0.1)]:
        kk=k&c
        print(f'   {name}: n {kk.sum()} frac {kk.sum()/k.sum():.3f} within1 {np.mean(np.abs(err[kk])<0.01):.4f} median err {np.median(err[kk])*100:.3f}%')
print('chi_eff range',ce.min(),ce.max(),'a max',cat('a_1')[sig].max())

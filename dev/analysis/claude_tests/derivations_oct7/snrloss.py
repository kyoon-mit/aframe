import h5py, numpy as np
R='/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630'
for pop in ['snr4_uniform','snr4_powerlaw']:
    w=h5py.File(f'{R}/{pop}/windows.h5'); m=h5py.File(f'{R}/{pop}/mf.h5')
    snr=w['params/snr'][:]; mc=w['params/chirp_mass'][:]; sig=~np.isnan(mc)
    mfs=m['mf_snr'][:]
    bk=np.where(~sig)[0][:200]
    Xb=w['X'][np.sort(bk)]
    print(pop,'background X std per detector',Xb.std(axis=(0,2)), 'mean',Xb.mean())
    idx=np.where(sig&(snr>=10))[0][:800]
    Xc=w['X_clean'][np.sort(idx)]
    idx=np.sort(idx)
    rw=np.sqrt((Xc.astype(np.float64)**2).sum(axis=(1,2)))
    print(pop,'n',len(idx),'window optimal SNR / injected: median %.4f 16/84 %s'%(np.median(rw/snr[idx]),np.percentile(rw/snr[idx],[16,84])))
    print(pop,'MF SNR / window optimal SNR: median %.4f 16/84 %s'%(np.median(mfs[idx]/rw),np.percentile(mfs[idx]/rw,[16,84])))
    print(pop,'MF SNR / injected: median %.4f'%np.median(mfs[idx]/snr[idx]))

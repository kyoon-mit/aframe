import pandas as pd, numpy as np
R='/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/'
D=R+'den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_train_snr8_mc_q_lowband_prob0.96/'
F=R+'free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96/'
models={'free141':F+'test_ck_141/','free248':F+'test_ck_248/','dr339':D+'test_ep339/','dr799':D+'test_ep799/',
        'bank703':R+'template_bank_703_spin3_1024hz_train_snr8_prob0.96/test_ep00/','bank235':R+'template_bank_235_1024hz_train_snr8_prob0.96/test_ep00/'}
pops=['snr4_powerlaw','snr8_powerlaw','snr4_uniform']
bands=[(4,8),(8,10),(10,12),(8,12),(12,16),(16,25),(25,50),(4,50.1)]
res={}
for m,p0 in models.items():
    for p in pops:
        try: d=pd.read_csv(p0+p+'/param_est_results.csv')
        except Exception as e: print(m,p,'missing',e); continue
        b=pd.read_csv(p0+p+'/param_est_results_bkg.csv')
        err=(d.chirp_mass_pred-d.chirp_mass_true).abs()/d.chirp_mass_true
        row={'n':len(d),'nbkg':len(b)}
        for lo,hi in bands:
            s=(d.snr>=lo)&(d.snr<hi)
            row[f'{lo}-{hi}']=(round(100*(err[s]<0.01).mean(),1), round(100*(err[s]<0.02).mean(),1), int(s.sum()))
        res[(m,p)]=row
        print(m,p,row)
import pickle; pickle.dump(res,open('tests.pkl','wb'))

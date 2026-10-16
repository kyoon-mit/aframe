"""Numbers introduced in writeup.md that are not already in a results file."""
import json
import numpy as np
from scipy import integrate, optimize, special

G, C, MSUN = 6.67430e-11, 299792458.0, 1.98847e30
out = {}


def tau(f, mc):
    m = G * mc * MSUN / C**3
    return 5 / 256 * m ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def f_of_tau(t, mc):
    m = G * mc * MSUN / C**3
    return (5 / 256 / t) ** (3 / 8) * m ** (-5 / 8) / np.pi


def fdot(f, mc):
    m = G * mc * MSUN / C**3
    return 96 / 5 * np.pi ** (8 / 3) * m ** (5 / 3) * f ** (11 / 3)


mc = 1.2
out["tau_1.2"] = {str(f): float(tau(f, mc)) for f in (80, 100, 200, 300, 400, 800)}
f4 = f_of_tau(4.0, mc)
out["f_4s_1.2"] = float(f4)
# in-step time fdot^-1/2
out["t_step_1.2"] = {str(f): float(fdot(f, mc) ** -0.5) for f in (100, 200, 300)}
# width at half maximum of a resonator with ring time T: 1/(pi T)
out["fwhm_for_ring"] = {str(T): 1 / (np.pi * T) for T in (4.0, float(tau(80, mc)), float(tau(400, mc)), 0.0057)}
# integral of tau df from f(4 s) to merger, cycles, ratio, for four masses
res = {}
for m_ in (0.85, 1.2, 1.8, 2.75):
    f1 = f_of_tau(4.0, m_)
    I, _ = integrate.quad(lambda f: tau(f, m_), f1, np.inf, limit=200)
    # cycles = int f dt = int f |dtau/df| df
    ncyc, _ = integrate.quad(lambda f: f * 8 / 3 * tau(f, m_) / f, f1, np.inf, limit=200)
    res[str(m_)] = {"f_4s": float(f1), "int_tau_df": I, "cycles": ncyc, "ratio": I / ncyc,
                    "two_pi_int": 2 * np.pi * I, "five_pi_over_4_cycles": 5 * np.pi / 4 * ncyc,
                    "three_pi_over_4_cycles": 3 * np.pi / 4 * ncyc}
out["count"] = res
# lone resonator read at merger: maximize 2a exp(-2 a tau) -> a = 1/(2 tau)
t0 = 1.0
a_grid = np.linspace(0.01, 5, 200001)
val = 2 * a_grid * np.exp(-2 * a_grid * t0)
ab = a_grid[np.argmax(val)]
out["lone_merger_readout"] = {"best_a_tau": float(ab * t0), "ring_over_tau": float(1 / (ab * t0)),
                              "amplitude_kept": float(np.exp(-ab * t0)),
                              "captured_band_times_tau": float(val.max() * t0 / 2 / np.pi * np.pi),
                              "one_over_e": float(1 / np.e)}
# captured band: 2a e^{-2a tau} in angular units per resonator energy; band in Hz = (2a e^{-2a tau})/(2pi)*2pi?
# M1 = E(f)/||K||^2 * 2a e^{-2 a tau}; E is energy per Hz with ||K||^2 = int E df, so 2a e^{-2a tau} is a width in Hz.
out["lone_merger_readout"]["captured_band_times_tau"] = float(val.max() * t0)
# quarter-cycle criterion: x - arctan x = pi/2
xs = optimize.brentq(lambda x: x - np.arctan(x) - np.pi / 2, 0.5, 10)
out["quarter_cycle"] = {"x_star": xs, "c": np.pi / xs, "k": 3 / 8 * np.pi / xs,
                        "amplitude_at_edge": float(1 / np.sqrt(1 + xs**2))}
# resonator group delay a/(a^2+d^2): centre 1/a, at d=a 1/(2a), area pi
a = 7.0
d = np.linspace(-2e5, 2e5, 4000001)
gd = a / (a**2 + d**2)
out["resonator_gd"] = {"centre_times_a": float(gd[len(d) // 2] * a),
                       "at_offset_a_times_a": float(a * a / (a**2 + a**2)),
                       "area_over_pi_numeric": float(np.trapz(gd, d) / np.pi),
                       "area_over_pi_exact_on_grid": float(2 * np.arctan(2e5 / a) / np.pi)}
# GELU odd part
y = np.linspace(-12, 12, 240001)
gelu = lambda v: v * 0.5 * (1 + special.erf(v / np.sqrt(2)))
out["gelu_odd_minus_half_y_max"] = float(np.max(np.abs(0.5 * (gelu(y) - gelu(-y)) - 0.5 * y)))
# linear FM: tau falls from T to 0 across Delta -> int tau df = T Delta / 2
T_, D_ = 2.0, 50.0
I_lin, _ = integrate.quad(lambda f: T_ * (1 - f / D_), 0, D_)
out["linear_fm"] = {"int_tau_df": I_lin, "T_Delta_over_2": T_ * D_ / 2}
out["sqrt_0.61"] = float(np.sqrt(0.61))
# relative costs quoted from verify/v_onset.json (ck_69), recomputed
von = json.load(open("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/verify/v_onset.json"))
k69 = von["sig_only_reg_s4d_max_dropout0_64x32x4:ck_69"]
k46 = von["sig_only_reg_free_kernel:ck_46"]
k135 = von["sig_only_reg_s4d_max_dropout0_64x32x4:ck_135"]
k81 = von["sig_only_reg_s4d_max_dropout0_64x32x4:ck_81"]
rc = lambda ctrl, x: (ctrl - x) / ctrl
out["edgefree_costs"] = {
    "s4d69_T3_0.5s": rc(k69["internal_ramps_seg1024"]["w1"], k69["T3_edgefree_seg1024"]["w1"]),
    "s4d69_T3_0.25s": rc(k69["internal_ramps_seg512"]["w1"], k69["T3_edgefree_seg512"]["w1"]),
    "s4d81_T3_0.25s": rc(k81["internal_ramps_seg512"]["w1"], k81["T3_edgefree_seg512"]["w1"]),
    "s4d135_T3_0.25s": rc(k135["internal_ramps_seg512"]["w1"], k135["T3_edgefree_seg512"]["w1"]),
    "s4d135_T3_0.5s": rc(k135["internal_ramps_seg1024"]["w1"], k135["T3_edgefree_seg1024"]["w1"]),
    "free_T3_0.5s": rc(k46["internal_ramps_seg1024"]["w1"], k46["T3_edgefree_seg1024"]["w1"]),
    "free_T3_0.25s": rc(k46["internal_ramps_seg512"]["w1"], k46["T3_edgefree_seg512"]["w1"]),
}
# T2/T3 relative drops quoted from mechanism/results_T2T3.json
r = json.load(open("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/mechanism/results_T2T3.json"))
s2 = r["tests"]["T2"]["by_estimator"]["s4d_ck69"]
s3 = r["tests"]["T3"]["by_estimator"]["s4d_ck69"]
out["T2T3_recomputed"] = {
    "T2_drop": rc(s2["within1_clean"], s2["within1_manipulated"]),
    "T3_drop": rc(s3["within1_clean"], s3["within1_manipulated"]),
    "taper_drop_T2set": rc(s2["within1_clean"], s2["within1_taper_only"]),
}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/writeup_checks/checks.json", "w"), indent=1, default=float)

# T4 channels against the chirp's starting frequency in the 4,000 clean windows
S = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath"
mcw = np.load(f"{S}/harness/data/clean_chirp_mass.npy").astype(float)
tmw = np.load(f"{S}/harness/data/clean_merger_sample.npy").astype(float) / 2048.0
fstart = f_of_tau(tmw, mcw)
t4 = json.load(open(f"{S}/mechanism/results_T4.json"))
fch = np.array(t4["s4d_ck69"]["channel_freq_Hz"])
nod = np.array(t4["s4d_ck69_kernel_only_noD_descriptive"]["channel_median_delta_s"])
below = fch < fstart.min()
out["t4"] = {"clean_merger_time_min_max": [float(tmw.min()), float(tmw.max())],
             "clean_chirp_mass_min_max": [float(mcw.min()), float(mcw.max())],
             "start_freq_min_max": [float(fstart.min()), float(fstart.max())],
             "channels": int(len(fch)), "channels_below_min_start": int(below.sum()),
             "pairs_where_chirp_passes_f_inside": float((fch[None, :] >= fstart[:, None]).mean()),
             "noD_peak_more_than_3s_before_merger": int((nod < -3.0).sum()),
             "noD_early_among_below": int((nod[below] < -3.0).sum()),
             "noD_early_freqs": [round(float(x), 1) for x in fch[nod < -3.0]],
             "noD_early_among_below_delta_min_max": [float(nod[below & (nod < -3.0)].min()),
                                                     float(nod[below & (nod < -3.0)].max())]}
print(json.dumps(out["t4"], indent=1))
json.dump(out, open(f"{S}/writeup_checks/checks.json", "w"), indent=1, default=float)

# T4 channels peaking at the merger: dominated by the skip term (D^2/||K||^2 > 1) or a dominant ring under 1 ms
tab = {r["channel"]: r for r in json.load(open(f"{S}/kernels/clean_s4d_layer1_table.json"))["channels"]}
chs = t4["s4d_ck69"]["channels"]
mad = t4["s4d_ck69"]["channel_median_abs_delta_s"]
clus = [c for c, m in zip(chs, mad) if m <= 0.1]
ddom = {c for c in clus if tab[c]["D2_over_kernel_norm2"] > 1.0}
short = {c for c in clus if tab[c]["dominant_ring"] < 1e-3}
out["t4_clustered"] = {"n_clustered": len(clus), "skip_dominated": len(ddom), "ring_under_1ms": len(short),
                       "both": len(ddom & short), "either": len(ddom | short)}
print(json.dumps(out["t4_clustered"]))
json.dump(out, open(f"{S}/writeup_checks/checks.json", "w"), indent=1, default=float)

"""(B) Exact numbers behind the bounds, for the two test signals, and Monte
Carlo of the contrast: one full-window matched-filter energy, pooled by the
mean over time or by the maximum over time.

Quantities (unit-norm signal s_hat, SNR rho, N = 8192, circular lags):
  C(d) = sum_t s_t s_{t+d}                      autocorrelation (by FFT)
  first-order bound, stride 1   rho |sum_t s_t| / sqrt(N)
  first-order bound, stride S   rho ||P_S s_hat||, P_S = projection onto S-periodic vectors
  second-order bound, any shift-invariant statistic (stride 1)
                                rho^2 sqrt(sum_d C(d)^2 / (2N)) = rho^2 / sqrt(2 N_eff)
  same, mean of local functions with receptive field R
                                rho^2 sqrt(sum_{|d|<R} C(d)^2 / (2N))
  same, stride S (shift-invariant only under shifts by S)
                                rho^2 sqrt((S / 2N) sum_r sum_d C_r(d)^2),  C_r(d) = sum_{t = r mod S} s_t s_{t+d}
  local energy E_R = max_t sum_{u=t}^{t+R-1} (rho s_u)^2
  all-orders bound for means of local functions (stride 1):
     SNR^2 <= (rho sum s)^2/N + rho^4 sum_{|d|<R} C^2/(2N) + (rho^4 R / N) [(e^E_R - 1)/E_R - 1]
Output: out/b_bounds.txt
"""
import sys
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, ".")
from thm_signals import newtonian, whitened_bns, N, FS

lines = []
def P(s=""):
    print(s, flush=True); lines.append(s)


def autocorr(s):
    S = np.fft.fft(s)
    return np.fft.ifft(np.abs(S) ** 2).real          # C(d), d = 0..N-1 (circular)


def lag_mask(R):
    d = np.arange(N)
    dd = np.minimum(d, N - d)
    return dd < R


def comb_fraction(s, S):
    r = np.arange(N) % S
    sums = np.array([s[r == k].sum() for k in range(S)])
    return (S / N) * np.sum(sums ** 2)             # ||P_S s||^2


def strided_C2(s, S, R=None):
    """sum_r sum_d C_r(d)^2 with C_r(d) = sum_{t = r mod S} s_t s_{t+d} (circular), |d| < R if given."""
    tot = 0.0
    r = np.arange(N) % S
    Fs = np.fft.fft(s)
    m = lag_mask(R) if R else np.ones(N, bool)
    for k in range(S):
        sk = np.where(r == k, s, 0.0)
        Cr = np.fft.ifft(np.conj(np.fft.fft(sk)) * Fs).real      # sum_t sk_t s_{t+d}
        tot += np.sum(Cr[m] ** 2)
    return tot


def local_energy(s, R):
    c = np.concatenate([[0.0], np.cumsum(np.concatenate([s, s[:R]]) ** 2)])
    return np.max(c[R:N + R] - c[:N])


def describe(name, s, rho=8.0):
    P(f"== {name}  (rho = {rho})")
    C = autocorr(s)
    P(f"   sum_t s_hat_t = {s.sum():+.4e};  first-order bound, stride 1: rho |sum s| / sqrt(N) = {rho*abs(s.sum())/np.sqrt(N):.4f}")
    for S in (4, 32):
        f = comb_fraction(s, S)
        P(f"   stride {S:2d}: fraction of signal energy on the comb (multiples of {FS/S:.0f} Hz) = {f:.3e};  first-order bound rho*sqrt = {rho*np.sqrt(f):.4f}")
    tot = np.sum(C ** 2)
    Neff = 1.0 / tot * N                          # (sum C(0))^2 N / sum C^2 with C(0) = 1
    P(f"   N_eff = N / sum_d C(d)^2 = {Neff:.1f};  second-order bound, any stride-1 shift-invariant statistic: rho^2/sqrt(2 N_eff) = {rho**2/np.sqrt(2*Neff):.4f}")
    for R in (49, 65, 899):
        m = lag_mask(R)
        b = rho ** 2 * np.sqrt(np.sum(C[m] ** 2) / (2 * N))
        flat = rho ** 2 * np.sqrt((2 * R - 1) / (2 * N))
        E = rho ** 2 * local_energy(s, R)
        chi = (np.expm1(E) / E - 1.0) if E > 0 else 0.0
        allord = np.sqrt((rho * s.sum()) ** 2 / N + b ** 2 + rho ** 4 * R / N * chi)
        P(f"   R = {R:3d}: sum_|d|<R C^2 / sum_all C^2 = {np.sum(C[m]**2)/tot:.4f}; second-order bound {b:.4f} (flat-signal value rho^2 sqrt((2R-1)/2N) = {flat:.4f}); "
          f"local energy E_R = {E:.3f}; all-orders bound {allord:.3f}")
    for S, R in ((4, 65), (32, 899), (32, None)):
        v = rho ** 2 * np.sqrt(S / (2 * N) * strided_C2(s, S, R))
        P(f"   stride {S}, " + (f"receptive field {R}" if R else "no receptive-field limit") + f": second-order bound {v:.4f}")
    # energy in the last 0.44 s and peak sample
    P(f"   largest sample at rho = {rho}: {rho*np.abs(s).max():.3f}; fraction of rho^2 in the 899 samples ending at the largest local-energy window: {local_energy(s, 899):.3f}")
    return C


def mf_energy(x, z):
    """E(tau) = |sum_u x_u conj(z_{u - tau})|^2, all circular lags tau; z analytic, unit-norm real part."""
    return np.abs(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(z)))) ** 2


def contrast_mc(name, s, draws=4000, seed=3):
    rng = np.random.default_rng(seed)
    z = hilbert(s)                                    # s + i H[s]
    P(f"== contrast (ii) on {name}: one full-window matched-filter energy E(tau) (both quadratures, noise mean 2)")
    P(f"   ||Re z|| = {np.linalg.norm(z.real):.4f}, ||Im z|| = {np.linalg.norm(z.imag):.4f}, <Re z, Im z> = {np.dot(z.real, z.imag):+.2e}")
    rhos = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    stats = {r: ([], []) for r in rhos}
    C = autocorr(s)
    for _ in range(draws // 200):
        n = rng.standard_normal((200, N))
        for r in rhos:
            E = np.array([mf_energy(row, z) for row in (n + r * s)])
            stats[r][0].append(E.max(1)); stats[r][1].append(E.mean(1))
    mx0 = np.concatenate(stats[0.0][0]); mn0 = np.concatenate(stats[0.0][1])
    # exact SNR of the mean-pooled energy: x^T M x with real symmetric circulant M whose
    # eigenvalues are mu_k = (|z_k|^2 + |z_-k|^2) / (2 N) (unnormalized DFT, x^T M x = sum mu_k |x_k|^2 / N)
    zf = np.abs(np.fft.fft(z)) ** 2
    mu = (zf + np.roll(zf[::-1], 1)) / (2 * N)
    sf = np.abs(np.fft.fft(s)) ** 2
    shift_unit = np.sum(mu * sf) / N                 # per rho^2
    var0 = 2 * np.sum(mu ** 2)                       # 2 ||M||_F^2
    thr_mx, thr_mn = np.quantile(mx0, 0.99), np.quantile(mn0, 0.99)
    P(f"   noise only: max over time: mean {mx0.mean():.2f}, std {mx0.std():.2f};  mean over time: mean {mn0.mean():.4f}, std {mn0.std():.5f} (exact std {np.sqrt(var0):.5f})")
    P(f"   {'rho':>4s} {'SNR max-pooled':>15s} {'P_det max (FAP 1%)':>19s} {'SNR mean-pooled':>16s} {'exact':>8s} {'rho^2/sqrt(2N_eff)':>18s} {'P_det mean (FAP 1%)':>20s}")
    Neff = N / np.sum(C ** 2)
    for r in rhos[1:]:
        mx = np.concatenate(stats[r][0]); mn = np.concatenate(stats[r][1])
        smx = (mx.mean() - mx0.mean()) / mx0.std()
        smn = (mn.mean() - mn0.mean()) / mn0.std()
        ex = r ** 2 * shift_unit / np.sqrt(var0)
        P(f"   {r:4.1f} {smx:15.3f} {np.mean(mx > thr_mx):19.3f} {smn:16.4f} {ex:8.4f} {r**2/np.sqrt(2*Neff):18.4f} {np.mean(mn > thr_mn):20.3f}")


def q3(a, b, R, rhos, block=256):
    """Q3(a, b) = sum_{t,t'} a_t b_t' [exp(K) - 1 - K](t, t'), K(t, t') = sum_{j<R} a_{t+j} b_{t'+j},
    circular, for amplitudes rho (a, b unit-norm shapes scaled by rho). Returns {rho: Q3}."""
    out = {r: 0.0 for r in rhos}
    idx = np.arange(N)
    for d0 in range(0, N, block):
        ds = np.arange(d0, min(d0 + block, N))
        bsh = b[(idx[None, :] + ds[:, None]) % N]            # b_{u+d}
        p = a[None, :] * bsh                                   # p_d(u) = a_u b_{u+d}
        ext = np.concatenate([p, p[:, :R]], axis=1)
        cs = np.concatenate([np.zeros((len(ds), 1)), np.cumsum(ext, axis=1)], axis=1)
        K = cs[:, R:N + R] - cs[:, :N]                         # K(t, t+d) = sum_{j<R} p_d(t+j)
        for r in rhos:
            k = r * r * K
            out[r] += r * r * np.sum(p * (np.expm1(k) - k))
    return out


def all_orders(label, s1, s2, R, rhos):
    """Upper bound on the SNR (or separation, if s2 is not s1) of any statistic that is the
    mean over all positions of one function of R consecutive samples (stride 1, circular):
    SNR^2 <= T1 + T2 + (Q3(1,1) - 2 Q3(1,2) + Q3(2,2)) / N; T1, T2 exact (first and second order)."""
    C1, C2 = autocorr(s1), autocorr(s2)
    m = lag_mask(R)
    same = s2 is s1
    qa = q3(s1, s1, R, rhos)
    qb = qa if same else q3(s1, s2, R, rhos)
    qc = qa if same else q3(s2, s2, R, rhos)
    res = {}
    for r in rhos:
        if same:
            T1 = (r * s1.sum()) ** 2 / N
            T2 = r ** 4 * np.sum(C1[m] ** 2) / (2 * N)
            hi = qa[r] / N
        else:
            T1 = (r * (s1.sum() - s2.sum())) ** 2 / N
            T2 = r ** 4 * np.sum((C1 - C2)[m] ** 2) / (2 * N)
            hi = (qa[r] - 2 * qb[r] + qc[r]) / N
        res[r] = (np.sqrt(T1), np.sqrt(T2), hi, np.sqrt(max(T1 + T2 + hi, 0.0)))
    for r in rhos:
        t1, t2, hi, tot = res[r]
        P(f"   {label}, R = {R}, rho = {r:4.1f}: first order {t1:.4f}, second order {t2:.4f}, orders >= 3 add at most {hi:.4f} to SNR^2 -> all orders <= {tot:.4f}")
    return res


def chirp_mass_pairs():
    P("== chirp mass: what a second-order shift-invariant statistic can see of a 1% change (rho = 8)")
    rho = 8.0
    def pair_report(label, s1, s2):
        C1, C2 = autocorr(s1), autocorr(s2)
        z2 = hilbert(s2)
        ov = np.abs(np.fft.ifft(np.fft.fft(s1) * np.conj(np.fft.fft(z2))))   # |<s1, z2 shifted>|, max over phase
        match = ov.max()
        d_full = rho * np.sqrt(2 * (1 - match))
        d2 = rho ** 2 * np.sqrt(np.sum((C1 - C2) ** 2) / (2 * N))
        d2R = {R: rho ** 2 * np.sqrt(np.sum(((C1 - C2)[lag_mask(R)]) ** 2) / (2 * N)) for R in (49, 899)}
        d2S = rho ** 2 * np.sqrt(32 / (2 * N) * strided_C2(s1 - 0 * s1, 32) * 0 + 32 / (2 * N) * strided_diff(s1, s2, 32))
        P(f"   {label}: match (max over time shift and phase) {match:.4f} -> matched-filter separation rho sqrt(2(1-match)) = {d_full:.3f};"
          f" second-order separation bound, any stride-1 shift-invariant statistic {d2:.4f}; mean of local functions R=49: {d2R[49]:.4f}, R=899: {d2R[899]:.4f}; stride 32: {d2S:.4f}")
    # Newtonian, same merger time
    s1, i1 = newtonian(1.2, 3.75)
    s2, i2 = newtonian(1.2 * 1.01, 3.75)
    pair_report(f"Newtonian 1.2 vs 1.212 Msun, same merger time (start {i1['f_start']:.2f} vs {i2['f_start']:.2f} Hz)", s1, s2)
    tc2 = 3.75 * 1.01 ** (-5.0 / 3.0)
    s3, i3 = newtonian(1.2 * 1.01, tc2)
    pair_report(f"Newtonian 1.2 vs 1.212 Msun, merger {3.75 - tc2:.4f} s earlier so both start at the same frequency ({i1['f_start']:.2f} vs {i3['f_start']:.2f} Hz)", s1, s3)
    b1, _ = whitened_bns(1.4, 1.4, 0.25)
    b2, _ = whitened_bns(1.414, 1.414, 0.25)
    pair_report("whitened IMRPhenomD 1.4+1.4 vs 1.414+1.414 (chirp mass +1%), same end 0.25 s after peak", b1, b2)
    end2 = 4.0 - 3.75 * 1.01 ** (-5.0 / 3.0)
    b3, _ = whitened_bns(1.414, 1.414, end2)
    pair_report(f"whitened IMRPhenomD, chirp mass +1%, window ending {end2:.4f} s after peak (same start frequency at leading order)", b1, b3)
    P("   all-orders separation bounds for means of one local function (stride 1):")
    all_orders("BNS +1% chirp mass, same end", b1, b2, 49, (4.0, 8.0, 12.0))
    all_orders("BNS +1% chirp mass, same start frequency", b1, b3, 49, (4.0, 8.0, 12.0))
    all_orders("Newtonian +1% chirp mass, same merger time", s1, s2, 49, (4.0, 8.0, 12.0))
    all_orders("Newtonian +1% chirp mass, same start frequency", s1, s3, 49, (4.0, 8.0, 12.0))
    # coherent stretches of R = 899 samples (the aframe ResNet's receptive field): how well one coherent
    # 0.44 s stretch of s1 separates s1 from s2. Heuristic: the stretch of s1 is compared with the same-length
    # stretch of s2 shifted by tau, maximized over tau and over phase (projection onto the span of the real and
    # imaginary parts of the analytic s2 stretch); separation = rho ||s1_W|| sqrt(2 (1 - match)).
    R = 899
    for lab, a, b in (("same end", b1, b2), ("same start frequency", b1, b3)):
        zb = hilbert(b)
        for seg_end in (7680 + 64, 7680 - 1024, 7680 - 2048, 2 * R):
            aw = a[seg_end - R:seg_end]
            frac = np.sum(aw ** 2)
            best = 0.0
            for tau in range(-400, 401):
                zw = np.roll(zb, -tau)[seg_end - R:seg_end]
                u, v = zw.real, zw.imag
                G = np.array([[u @ u, u @ v], [u @ v, v @ v]])
                c = np.array([aw @ u, aw @ v])
                best = max(best, np.sqrt(c @ np.linalg.solve(G, c)) / np.linalg.norm(aw))
            P(f"   coherent {R}-sample stretch ending at sample {seg_end} ({lab}): holds {frac:.3f} of rho^2; match {best:.4f}; separation at rho = 8: {8*np.sqrt(frac)*np.sqrt(2*(1-best)):.3f}")
    P(f"   leading order: frequency at the window start f ~ Mc^(-5/8) tau^(-3/8), so Mc ~ f^(-8/5) tau^(-3/5); tau from 3.5 to 4 s spans a factor (4/3.5)^(3/5) = {(4/3.5)**0.6:.4f} in Mc at fixed f")


def strided_diff(s1, s2, S):
    tot = 0.0
    r = np.arange(N) % S
    F1, F2 = np.fft.fft(s1), np.fft.fft(s2)
    for k in range(S):
        a = np.fft.ifft(np.conj(np.fft.fft(np.where(r == k, s1, 0.0))) * F1).real
        b = np.fft.ifft(np.conj(np.fft.fft(np.where(r == k, s2, 0.0))) * F2).real
        tot += np.sum((a - b) ** 2)
    return tot


if __name__ == "__main__":
    s_newt = np.load("out/sig_newtonian.npy")
    s_bns = np.load("out/sig_whitened_bns.npy")
    describe("Newtonian toy, Mc 1.2, merger 3.75 s, cut 800 Hz", s_newt)
    describe("whitened IMRPhenomD 1.4+1.4, window ends 0.25 s after peak", s_bns)
    P("== all orders: best possible SNR of a mean over positions of one function of R samples (stride 1)")
    for nm, s in (("Newtonian toy", s_newt), ("whitened BNS", s_bns)):
        for R in (49, 65, 899):
            all_orders(nm, s, s, R, (4.0, 8.0, 12.0))
    contrast_mc("Newtonian toy", s_newt)
    contrast_mc("whitened IMRPhenomD", s_bns)
    chirp_mass_pairs()
    open("out/b_bounds.txt", "w").write("\n".join(lines) + "\n")

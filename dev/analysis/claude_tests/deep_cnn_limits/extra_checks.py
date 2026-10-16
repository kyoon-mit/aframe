"""Comparisons with derivations #sep-fisher and with a longer band.

1. #sep-fisher reports, for a window ending at merger, bound(3.5 s) / bound(30 s)
   = 4.90 to 5.24 (bin medians) at fixed signal amplitude, and SNR(30 s) /
   SNR(3.5 s) = 1.17 to 1.22, with 4 parameters (ln Mc, eta, t_c, phi_c) and a
   3.5PN phase. Same ratios here with the Newtonian phase and 3 parameters.
2. Coherent bound for the whole band from 20 Hz at the same SNR.
"""
import numpy as np

from common import Noise, band, moments
from fisher_seg import coherent

noise = Noise("o3")
print("Newtonian, 3 parameters, O3 average PSD, window ending at merger, cap min(f_ISCO, 1000 Hz):")
for mc in (0.85, 1.2, 1.8, 2.75):
    fhi = band(mc, 3.5, 1000.0)[1]
    f35, f30 = band(mc, 3.5, 1000.0)[0], max(20.0, band(mc, 30.0, 1000.0)[0])
    m35, m30 = moments(f35, fhi, mc, noise)["m0"], moments(f30, fhi, mc, noise)["m0"]
    snr_ratio = np.sqrt(m30 / m35)
    s35 = coherent(mc, noise, snr=8.0, flo=f35, fhi=fhi)["sig_lnmc"]
    s30 = coherent(mc, noise, snr=8.0 * snr_ratio, flo=f30, fhi=fhi)["sig_lnmc"]
    full = coherent(mc, noise, snr=8.0, flo=20.0, fhi=band(mc)[1])["sig_lnmc"]
    four = coherent(mc, noise, snr=8.0)["sig_lnmc"]
    print(f"  Mc {mc}: f_3.5s {f35:.1f} Hz, f_30s {f30:.1f} Hz; SNR(30 s)/SNR(3.5 s) = {snr_ratio:.3f}; "
          f"bound 3.5 s / bound 30 s at fixed amplitude = {s35 / s30:.2f}; "
          f"SNR 8: 3.5 s {100 * s35:.4f}%, 4 s {100 * four:.4f}%, 20 Hz to cap {100 * full:.5f}% "
          f"(4 s / from 20 Hz = {four / full:.1f})")

"""Leading-order frequency, cycles and integral of tau df for the last T seconds before merger."""
import numpy as np
from scipy.integrate import quad

TSUN = 4.925490947e-6  # G Msun / c^3 in s, as in the time-in-band check


def f_of_tau(tau, mc):
    return (5.0 / (256.0 * tau)) ** 0.375 * (mc * TSUN) ** (-0.625) / np.pi


def tau_of_f(f, mc):
    return 5.0 / 256.0 * (mc * TSUN) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


masses = [0.85, 1.2, 1.8, 2.75]
Ts = {"4 s": 4.0, "3.5 s": 3.5, "3.5 s + 1 sample": 3.5 + 1 / 2048, "detector shortest": (7169 - 43) / 2048,
      "detector longest pre": (8192 + 43) / 2048}
print("ratios: (7/8)^(5/8) = %.6f, (8/7)^(3/8) = %.6f" % ((7 / 8) ** 0.625, (8 / 7) ** 0.375))
for name, T in Ts.items():
    print(f"--- T = {T:.6f} s ({name})")
    for mc in masses:
        f1 = f_of_tau(T, mc)
        ncyc = 1.6 * T * f1
        itdf = 0.6 * T * f1
        # quadrature: cycles = int_0^T f dtau ; int tau df from f1 to inf
        q_cyc = quad(lambda t: f_of_tau(t, mc), 0, T, limit=200)[0]
        q_itdf = quad(lambda f: tau_of_f(f, mc), f1, np.inf, limit=200)[0]
        print(f"  Mc {mc:5.2f}: f(T) {f1:8.3f} Hz  cycles {ncyc:8.3f} (quad {q_cyc:8.3f})  "
              f"int tau df {itdf:8.3f} (quad {q_itdf:8.3f})  ratio {q_itdf / q_cyc:.6f}  "
              f"8/sqrt(N) {8 / np.sqrt(ncyc):.4f}  1.55sqrtN {1.55 * np.sqrt(ncyc):.2f}")
# leading-order cycles in the last 21.26 ms (merger pushed past the window end in one detector)
for mc in masses:
    t2 = 43 / 2048
    print(f"Mc {mc}: cycles in last {t2 * 1e3:.2f} ms = {1.6 * t2 * f_of_tau(t2, mc):.2f}, f there {f_of_tau(t2, mc):.1f} Hz")

"""Shares of the logged time per batch outside preparation and the training step (numbers from EXPERIMENTS.md, 2026-10-15 15:40)."""
prep = 2.7
for name, step, batch in (("free kernels", 21.2, 238.0), ("S4D d_state 64", 24.1, 260.0), ("S4D d_state 512", 44.2, 284.0)):
    inside = prep + step
    print(f"{name}: prep+step {inside:.1f} ms of {batch:.0f} ms; neither {100 * (1 - inside / batch):.2f}%; "
          f"prep share of GPU work {100 * prep / inside:.1f}%")
print("batches per epoch", 1564 + 64)

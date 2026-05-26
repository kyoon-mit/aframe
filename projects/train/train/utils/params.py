import torch


def compute_target_params_tensor(
    parameters: dict,
    target_parameters: tuple[str, ...],
) -> torch.Tensor:
    """Compute regression target params from a dict of tensors or numpy arrays.

    Derives ``chirp_mass`` and ``mass_ratio`` from ``mass_1``/``mass_2`` (or
    vice-versa) when needed, then stacks the requested columns into a
    ``(N, len(target_parameters))`` float tensor.
    """
    available = {k: v for k, v in parameters.items()}
    if "chirp_mass" not in available and "mass_1" in available and "mass_2" in available:
        m1, m2 = available["mass_1"], available["mass_2"]
        available["chirp_mass"] = ((m1 * m2) ** 3 / (m1 + m2)) ** 0.2
        available["mass_ratio"] = m2 / m1
    elif "mass_1" not in available and "chirp_mass" in available and "mass_ratio" in available:
        mc, q = available["chirp_mass"], available["mass_ratio"]
        available["mass_1"] = mc * (1 + q) ** 0.2 / q ** 0.6
        available["mass_2"] = mc * q ** 0.4 * (1 + q) ** 0.2
    cols = []
    for name in target_parameters:
        if name not in available:
            raise ValueError(
                f"Unknown target parameter {name!r}. "
                f"Choose from {list(available)}"
            )
        cols.append(available[name])
    return torch.stack(
        [c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in cols],
        dim=-1,
    ).float()

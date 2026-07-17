import torch
from torch import Tensor
from jaxtyping import Array, PyTree

import jax


def tensor_to_jax_array(t: PyTree[Tensor]) -> PyTree[Array]:
    return jax.tree.map(
        lambda x: jax.dlpack.from_dlpack(x.contiguous()),
        t,
        is_leaf=lambda x: isinstance(x, Tensor),
    )


def jax_array_to_tensor(a: PyTree[Array]) -> PyTree[Tensor]:
    return jax.tree.map(
        lambda x: torch.from_dlpack(x),
        a,
        is_leaf=lambda x: isinstance(x, Array),
    )

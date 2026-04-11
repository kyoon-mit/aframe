import torch


class Architecture(torch.nn.Module):
    """
    All architectures should accept an argument
    `num_ifos` as their first argument, since the
    CLI links to this argument from the datamodule
    at initialization time.

    TODO: if you really wanted to enforce this with
    gross overkill, you could make a metaclass
    that inspects the __init__ method of each new
    subclass and ensures that the first parameter
    is an int called `num_ifos`.
    """

    # def __init__(self, num_ifos: int) -> None:
    #     pass


try:
    import jax  # noqa: F401
    import equinox as eqx

    class JaxArchitecture(eqx.Module):
        """
        Base class for all Jax-based architectures.
        """

except ImportError:

    class JaxArchitecture:
        """
        Stub raised when JAX/equinox are not installed.
        Install the optional dependency with: uv sync --extra jax
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "JAX and equinox are required to use JaxArchitecture. "
                "Install them with: uv sync --extra jax"
            )

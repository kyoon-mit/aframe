import hashlib
import logging
import re
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

logger = logging.getLogger(__name__)

# Fixed constants used for every checksum — must never change between
# save and load, otherwise all existing checksums become invalid.
_BATCH_SIZE = 4
_SEED = 0


def compute_model_checksum(
    model: eqx.Module,
    state: eqx.nn.State,
    num_ifos: int,
    time_len: int,
) -> str:
    """Return an 8-char hex checksum for *model* + *state*.

    A fixed random batch of shape ``(_BATCH_SIZE, num_ifos, time_len)``
    is passed through the model in inference mode (dropout/BatchNorm off).
    The raw bytes of the output array are hashed with MD5 and the first
    8 hex characters are returned.
    """
    # Import here to avoid circular imports at module load time.
    from train.utils.jax.training import jax_inference

    k1, k2 = jr.split(jr.PRNGKey(_SEED), 2)
    X = jr.normal(k1, (_BATCH_SIZE, num_ifos, time_len))
    keys = jr.split(k2, _BATCH_SIZE)
    outputs, _ = jax_inference(model, X, state, keys)
    # log outputs
    logger.info("Computing checksum from model outputs:\n%s", outputs)
    raw = jnp.asarray(outputs).tobytes()
    return hashlib.md5(raw).hexdigest()[:8]


# Regex that matches the trailing ``_{num_ifos}_{time_len}_{checksum}``
# segment appended by JAXCheckpointManager.  Everything before it is the
# base name (step / metric info).
_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)_([0-9a-f]{8})$")


def parse_checkpoint_meta(
    path: Path | str,
) -> tuple[int, int, str] | None:
    """Extract ``(num_ifos, time_len, checksum)`` embedded in *path*'s stem.

    Returns ``None`` if the filename does not contain a recognisable suffix
    (e.g. a checkpoint written by an older version of the manager).
    """
    stem = Path(path).stem
    m = _SUFFIX_RE.search(stem)
    if m is None:
        return None
    num_ifos = int(m.group(1))
    time_len = int(m.group(2))
    checksum = m.group(3)
    return num_ifos, time_len, checksum


def verify_checkpoint(
    path: Path | str,
    model: eqx.Module,
    state: eqx.nn.State,
) -> None:
    """Recompute the checksum for *model*+*state* and warn if it differs
    from the one embedded in *path*.  A missing suffix is silently ignored
    (backward-compatible with old checkpoints).
    """
    meta = parse_checkpoint_meta(path)
    if meta is None:
        logger.debug(
            f"Checkpoint {path} has no embedded checksum"
            " — skipping verification."
        )
        return

    num_ifos, time_len, saved_checksum = meta
    actual_checksum = compute_model_checksum(model, state, num_ifos, time_len)

    if actual_checksum != saved_checksum:
        logger.warning(
            f"Checkpoint integrity check FAILED for {path}:\n"
            f"  expected checksum: {saved_checksum}\n"
            f"  computed checksum: {actual_checksum}\n"
            "The loaded weights may not match what was originally saved."
        )
    else:
        logger.info(
            f"Checkpoint integrity verified (checksum={actual_checksum})."
        )

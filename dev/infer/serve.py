"""Step 0: serve an exported model repo with Triton via hermes.

Run inside the infer uv env (`uv run python serve.py [--repo ...]`). Uses the
same hermes.aeriel.serve path aframe's own infer task uses, so it matches your
build. Writes this node's address for step 1, then holds the server open until
the Slurm job is cancelled or times out (which tears the server down cleanly).

The IP file is the readiness signal step 1 waits on: it is cleared at startup
and (re)written only once Triton reports READY, so step 1 can never read a
stale address left behind by a dead server. It holds "ip grpc_port http_port"
so the client knows exactly where to connect.

Custom ports matter for concurrency: SLURM can pack several serve jobs onto one
gpu_requeue node, and two Tritons on the same host collide on the default
8000/8001/8002. Each shard therefore serves on its own port triple. hermes'
built-in wait() hardcodes localhost:8001, so with custom ports we poll the
server's own HTTP health endpoint ourselves instead of using wait=True.
"""

import argparse
import os
import socket
import time
import urllib.request
from pathlib import Path

from hermes.aeriel.serve import serve

BASE = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s"  # noqa: E501
DEFAULT_REPO = f"{BASE}/chirp_mass_snr_8_50_60-64s_d64_s64_l4_on_disk_id2"
DEFAULT_IP_FILE = f"{BASE}/triton_ip.txt"
DEFAULT_LOG = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/server.log"  # noqa: E501

# aframe's standard Triton image (pipelines/sandbox/configs/base.cfg). Override
# with $TRITON_IMAGE if you have a .sif path or a different tag.
IMAGE = os.environ.get("TRITON_IMAGE", "hermes/tritonserver:23.01")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ip-file", default=DEFAULT_IP_FILE)
    parser.add_argument("--log-file", default=DEFAULT_LOG)
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--grpc-port", type=int, default=8001)
    parser.add_argument("--metrics-port", type=int, default=8002)
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=1200,
        help="seconds to wait for the server to report ready",
    )
    return parser.parse_args()


def wait_until_ready(ip, http_port, timeout):
    """Poll Triton's HTTP health endpoint until ready (or raise on timeout)."""
    url = f"http://{ip}:{http_port}/v2/health/ready"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"server not ready after {timeout}s ({url})")


def main():
    arguments = parse_arguments()
    ip = socket.gethostbyname(socket.gethostname())
    ip_file = Path(arguments.ip_file)
    ip_file.unlink(missing_ok=True)
    print(
        f"starting {arguments.repo}\n  on {ip} "
        f"(http={arguments.http_port} grpc={arguments.grpc_port} "
        f"metrics={arguments.metrics_port}, image={IMAGE})",
        flush=True,
    )

    server_args = [
        "--http-port",
        str(arguments.http_port),
        "--grpc-port",
        str(arguments.grpc_port),
        "--metrics-port",
        str(arguments.metrics_port),
    ]
    # The exported models are KIND_GPU, so the server needs a GPU. Slurm gives
    # us one via --gres=gpu:1 (visible as device 0 inside the job). wait=False
    # because hermes' wait() only knows the default port; we poll our own.
    with serve(
        arguments.repo,
        IMAGE,
        gpus=[0],
        server_args=server_args,
        log_file=arguments.log_file,
        wait=False,
    ):
        wait_until_ready(ip, arguments.http_port, arguments.ready_timeout)
        ip_file.write_text(
            f"{ip} {arguments.grpc_port} {arguments.http_port}\n"
        )
        print(
            f"triton READY; published -> {ip_file} "
            f"({ip} grpc={arguments.grpc_port})",
            flush=True,
        )
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()

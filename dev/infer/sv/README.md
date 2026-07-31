# Sensitive-Volume pipeline (one command)

    bash run_sv.sh --ckpt <last.ckpt> --config <run/config.yaml> \
                   --group reg|class --name <run_name> [--livetime 1wk|1mo]

Traces the ckpt (arch/dims auto from config), streams branches as co-located
gpu_test server+client shards (%1, no OOM, no eviction), then sweeps SV metrics.
Everything runs in SLURM so it survives session idle. Outputs under
triton_ts/merger_4s/<group>/<name>/step2_aggregated_<lt>/<metric>/<integ>/.

See memory: project_sv_pipeline_gotchas, reference_sv_runbook.

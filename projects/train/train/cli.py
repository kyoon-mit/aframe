from lightning.pytorch.cli import LightningCLI

from train.callbacks import WandbSaveConfig
from train.data import BaseAframeDataset
from train.model import AframeBase


class AframeCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        # hack into init to hardcode
        # the WandbSaveConfig callback
        kwargs["parser_kwargs"] = {"parser_mode": "omegaconf"}
        kwargs["save_config_callback"] = WandbSaveConfig
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        # link data arguments to model;
        # some models require information about
        # sequence length before hand, so we need
        # to link sample rate and kernel length
        parser.link_arguments(
            "data.num_ifos",
            "model.init_args.arch.init_args.num_ifos",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.init_args.windowing.kernel_length",
            "model.init_args.arch.init_args.kernel_length",
            apply_on="parse",
        )
        parser.link_arguments(
            (
                "data.init_args.sample_rate",
                "data.init_args.model_input_sample_rate",
            ),
            "model.init_args.arch.init_args.sample_rate",
            apply_on="parse",
            compute_fn=lambda sr, msr: msr if msr is not None else sr,
        )

        parser.link_arguments(
            "data.init_args.valid_stride",
            "model.init_args.metric.init_args.stride",
        )

        # NOTE: metric_X / metric_X_spec are optional extra metrics that only
        # exist on some models (e.g. the time+spectrogram supervised model).
        # We can't link their stride here: link_arguments registers the link
        # successfully (the source exists), but jsonargparse then fails to
        # validate any model class that lacks these params (a bare
        # StopIteration when it can't find the target action). Configs that
        # use these metrics set stride explicitly instead.

        # TODO: This is a workaround for linking num_chirp_masses between
        # the model and the architecture for in_channels.
        try:
            parser.link_arguments(
                "data.init_args.num_chirp_masses",
                "model.init_args.arch.init_args.num_chirp_masses",
            )
        except Exception:
            pass

        parser.link_arguments(
            "data.init_args.sample_rate",
            "data.init_args.waveform_sampler.init_args.sample_rate",
            apply_on="parse",
        )

        parser.link_arguments(
            "data.init_args.fduration",
            "data.init_args.waveform_sampler.init_args.fduration",
            apply_on="parse",
        )

        parser.link_arguments(
            "data.init_args.ifos",
            "data.init_args.waveform_sampler.init_args.ifos",
            apply_on="parse",
        )

        parser.link_arguments(
            (
                "model.init_args.metric.init_args.pool_length",
                "data.init_args.valid_stride",
            ),
            "trainer.num_sanity_val_steps",
            compute_fn=lambda x, y: int(x / y),
            apply_on="parse",
        )


def main(args=None):
    cli = AframeCLI(
        AframeBase,
        BaseAframeDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
    )
    return cli


if __name__ == "__main__":
    main()

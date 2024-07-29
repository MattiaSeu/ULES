import click
import os
from os.path import join, dirname, abspath
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from datasets.datasets import StatDataModule
from models.ULES import Ules as ULES
from models.ULES_DB import Ules as ULES_DB
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)), 'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights '
                   'from the checkpoint file without resuming training',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training or test',
              default=None)
@click.option('--data_ratio',
              '-dt_rt',
              type=click.IntRange(10, 100),
              help='percentage of dataset to be used',
              default=100)
@click.option('--gpus',
              '-g',
              help='number of gpus to be used',
              default=1)
@click.option('--only_bb/--bb+head',
              show_default=True,
              help='if triggered, load all state dict, otherwise just backbone',
              default=False)
@click.option('--rgb_only_ft/--full_ft',
              show_default=False,
              help='use to fine tune on rgb only if using a double backbone',
              default=False)
@click.option('--double_backbone/--single_backbone',
              show_default=True,
              help='if triggered, use double backbone',
              default=True)

def main(config, weights, checkpoint, data_ratio, gpus, only_bb, rgb_only_ft, double_backbone):
    cfg = yaml.safe_load(open(config))
    torch.manual_seed(cfg['experiment']['seed'])

    # use the comment block below if you don't plan on using command line
    dataset_name = "VisNir"
    # data_ratio = 100
    if weights == "yes":
        weights = 'unsup_pretrain/checkpoints/epoch=99-visnir3_db.ckpt'
    # rgb_only_ft = True
    # double_backbone = False

    if dataset_name in cfg["data"].keys():
        image_size = cfg["data"][dataset_name]['image_size']
        mean = cfg["data"][dataset_name]['mean']
        std = cfg["data"][dataset_name]['std']
    else:
        raise Exception("No dataset named {}".format(dataset_name))

    # Load data and model
    mean = None
    std = None
    head_only = False
    data = StatDataModule(cfg, dataset_name, data_ratio, image_size=image_size, mean=mean, std=std)
    model = ULES_DB(cfg, dataset_name, rgb_only_ft, double_backbone, head_only, mean, std, unfreeze_epoch=200)

    if weights:
        checkpoint = torch.load(weights)
        state_dict = checkpoint["state_dict"]
        loopable_state_dict = dict(state_dict)

        # following loop is to adapt from pixel pro learner weights
        for k in loopable_state_dict.keys():
            if k.startswith("backbone.0."):
                new_k = k.replace("backbone.0.", "backbone.conv1.")
                state_dict[new_k] = state_dict.pop(k)
            if k.startswith("backbone.1."):
                new_k = k.replace("backbone.1.", "backbone.bn1.")
                state_dict[new_k] = state_dict.pop(k)
        loopable_state_dict = dict(state_dict)

        # remove an unused key
        for k in loopable_state_dict.keys():
            if k.endswith("num_batches_tracked"):
                del state_dict[k]
        loopable_state_dict = dict(state_dict)

        # all these following loops are to adapt the key names to our architecture
        for k in loopable_state_dict.keys():
            k_str = str(k)
            k_split = k_str.split(sep=".")
            if len(k_split) == 3:
                continue
            k_suffix = k_split[1]
            if k.startswith("backbone." + k_suffix + "."):
                k_suffix_new = str(int(k_suffix) - 3)
                new_k = k.replace("backbone." + k_suffix + ".", "backbone.layer" + k_suffix_new + ".")
                state_dict[new_k] = state_dict.pop(k)
        loopable_state_dict = dict(state_dict)

        for k in loopable_state_dict.keys():
            if k.startswith("backbone"):
                new_k = k.replace("backbone.", "model.backbone.")
                state_dict[new_k] = state_dict.pop(k)
        loopable_state_dict = dict(state_dict)

        for k in loopable_state_dict.keys():
            if k.startswith("model.target_encoder.net"):
                new_k = k.replace("model.target_encoder.net.", "model.")
                state_dict[new_k] = state_dict.pop(k)
        loopable_state_dict = dict(state_dict)

        for k in loopable_state_dict.keys():
            if k.startswith("model.target_encoder_rgb"):
                if "net.backbone" in k:
                    new_k = k.replace("model.target_encoder_rgb.net.backbone.", "model.backbone_rgb.")
                else:
                    new_k = k.replace("model.target_encoder_rgb.net.", "model.backbone_rgb.")
                state_dict[new_k] = state_dict.pop(k)
        loopable_state_dict = dict(state_dict)

        # in the pretraining arch I decided to call the range view backbone "gray" because it was a grayscale image
        # in the end they are brought back to a specific color space anyway, so I decided to use "range" here
        for k in loopable_state_dict.keys():
            if k.startswith("model.target_encoder_gray"):
                if "net.backbone" in k:
                    new_k = k.replace("model.target_encoder_gray.net.backbone.", "model.backbone_range.")
                else:
                    new_k = k.replace("model.target_encoder_gray.net.", "model.backbone_range.")
                state_dict[new_k] = state_dict.pop(k)
        loopable_state_dict = dict(state_dict)

        # this block is to only use the rgb weights of the pretrained double backbone
        if rgb_only_ft and not double_backbone:
            for k in loopable_state_dict.keys():
                if k.startswith("model.backbone_rgb"):
                    new_k = k.replace("model.backbone_rgb.", "model.backbone.")
                    state_dict[new_k] = state_dict.pop(k)
            loopable_state_dict = dict(state_dict)

        # remove everything that isn't backbone

        if only_bb:
            for k in loopable_state_dict.keys():
                # if not k.startswith("model.backbone"):
                if not "backbone" in k:
                    del state_dict[k]
        loopable_state_dict = dict(state_dict)

        # remove the first layer if we're only using the image for image+range pretraining
        # if range_pt:
        #     for k in loopable_state_dict.keys():
        #         if k.startswith("model.backbone.conv1"):
        #             del state_dict[k]

        odd_keys = model.load_state_dict(state_dict, strict=False)  # stores mismatching keys for warning
        # warn the user of mismatching keys
        if len(odd_keys[0]) > 1:
            print("The following keys are missing:")
            for miss_key in odd_keys[0]:
                print(miss_key)
            if odd_keys[1]:
                print("The following keys are unexpected:")
                for unexpected_key in odd_keys[1]:
                    print(unexpected_key)
        # if any("backbone" in s for s in odd_keys[0]):
        #     raise LookupError("No backbone key found. Check weights compatibility with model")

    # model.model.head[4] = torch.nn.Conv2d(512, 23, kernel_size=(1, 1), stride=(1, 1))
    # Add callbacks:
    if cfg['train']['mode'] == "train":
        if double_backbone and not rgb_only_ft:
            version_name = "db_no_pt_" if weights is None else "db_ft_"
        elif double_backbone and rgb_only_ft:
            if weights:
                version_name = "db_ft_rgb_"
        elif not double_backbone and not rgb_only_ft:
            version_name = "sb_no_pt_" if weights is None else "sb_ft_"
        elif not double_backbone and rgb_only_ft:
            if weights:
                version_name = "sb_ft_range_"
        version_name = version_name + str(data_ratio) + "%"
        #version_name = version_name + "crop_jaccard_norm_" + str(data_ratio) + "%"
    elif cfg['train']['mode'] == "eval":
        version_split = checkpoint.replace("checkpoints/", "")
        version_split = version_split.split("_")
        if len(version_split) == 4:
            version_name = "test_no_pt_" + version_split[2]
        elif len(version_split) == 3:
            version_name = "test_ft_" + version_split[1]
    elif cfg['train']['mode'] == "infer":
        version_name = "infer_db_ft"


    # version_name = "trial_100%"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd(), name='experiments/visnir/',
                                             version=version_name, default_hp_metric=False)

    # Setup trainer
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/",
                                          filename="{epoch}-%s" % version_name,
                                          monitor="sem_loss")


    # create your own theme!
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".10e",
        )
    )

    # class UnfreezeCallback(Callback):
    #     def __init__(self, unfreeze_epoch):
    #         self.unfreeze_epoch = unfreeze_epoch
    #
    #     def on_epoch_end(self, trainer, pl_module):
    #         if trainer.current_epoch == self.unfreeze_epoch:
    #             pl_module.unfreeze_backbone()
    #             pl_module.trainer.logger.info("Backbone unfreezed")

    torch.set_float32_matmul_precision('high')
    trainer = Trainer(devices=gpus,
                      logger=tb_logger,
                      log_every_n_steps=10,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[checkpoint_callback, progress_bar])#, accumulate_grad_batches=2)
    # Train
    if cfg['train']['mode'] == "train":
        trainer.fit(model, datamodule=data)  # .ckpt_path=checkpoint)
        # trainer.save_checkpoint("checkpoints/%s/%s.ckpt" % (dataset_name, version_name))
    elif cfg['train']['mode'] == "eval":
        trainer.test(model, ckpt_path=checkpoint, datamodule=data)
    elif cfg['train']['mode'] == "infer":
        predictions = trainer.predict(model, datamodule=data)
        print(predictions)


if __name__ == "__main__":
    main()

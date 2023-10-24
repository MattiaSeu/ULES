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


@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)), 'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--data_ratio',
              '-dt_rt',
              type=click.Choice([10, 20, 50, 100]),
              help='percentage of dataset to be used',
              default=100)
@click.option('--gpus',
              '-g',
              help='number of gpus to be used',
              default=1)
@click.option('--only_bb',
              '-bb',
              is_flag=True,
              show_default=True,
              help='if flag true, load all state dict, otherwise just backbone',
              default=True)
def main(config, weights, checkpoint, data_ratio, gpus, only_bb):
    cfg = yaml.safe_load(open(config))
    torch.manual_seed(cfg['experiment']['seed'])
    # use the comment block below if you don't plan on using command line
    data_ratio = 10
    weights = 'checkpoints/epoch=199-step=333200.ckpt'

    # Load data and model
    data = StatDataModule(cfg, data_ratio)

    if weights is None:
        model = ULES(cfg)
    else:
        model = ULES(cfg)
        checkpoint = torch.load(weights)
        state_dict = checkpoint["state_dict"]
        loopable_state_dict = dict(state_dict)
        # following loop is to adapt from pixel pro learner weights
        for k in loopable_state_dict.keys():
            if k.startswith("model.online_encoder.net"):
                new_k = k.replace("model.online_encoder.net.", "model.")
                state_dict[new_k] = state_dict.pop(k)
        loopable_state_dict = dict(state_dict)

        # remove everything that isn't backbone
        if only_bb:
            for k in loopable_state_dict.keys():
                if not k.startswith("model.backbone"):
                    del state_dict[k]

        odd_keys = model.load_state_dict(state_dict, strict=False) # stores mismatching keys for warning
        # warn the user of mismatching keys
        if len(odd_keys[0]) > 1:
            print("The following keys are missing:")
            for miss_key in odd_keys[0]:
                print(miss_key)
            if odd_keys[1]:
                print("The following keys are unexpected:")
                for unexpected_key in odd_keys[1]:
                    print(unexpected_key)
        if any("backbone" in s for s in odd_keys[0]):
            raise LookupError("No backbone key found. Check weights compatibility with model")

    # Add callbacks:
    version_name = "no_pt_" if weights is None else "ft_"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd(), name='experiments/',
                                             version=version_name + str(data_ratio) + "%", default_hp_metric=False)

    # Setup trainer
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints",
                                          filename="{epoch}-%s%s%%_final.ckpt" % (version_name, data_ratio),
                                          save_top_k=1, monitor="sem_loss")

    trainer = Trainer(devices=gpus,
                      logger=tb_logger,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[checkpoint_callback])
    # Train
    trainer.fit(model, datamodule=data)  # .ckpt_path=checkpoint)
    trainer.save_checkpoint("checkpoints/%s%s%%_final.ckpt" % (version_name, data_ratio))


if __name__ == "__main__":
    main()

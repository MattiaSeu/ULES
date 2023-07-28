import click
from os.path import join, dirname, abspath
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from datasets.datasets import StatDataModule
from models.ULES import Ules as ULES


@click.command()
### Add your options here
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
@click.option('--reduced_data',
              '-redata',
              is_flag=True,
              show_default=True,
              help='finetune on 10% of dataset only',
              default=False)

def main(config, weights, checkpoint, reduced_data):
    cfg = yaml.safe_load(open(config))
    torch.manual_seed(cfg['experiment']['seed'])

    # reduced_data = True
    # weights = 'checkpoints/vicreg_city_updated.ckpt'

    # Load data and model
    data = StatDataModule(cfg, reduced_data)

    if weights is None:
        model = ULES(cfg)
    else:  # this works only if we pre-train the whole network (encoder + decoder)
        model = ULES.load_from_checkpoint(weights, cfg=cfg, strict=False)

    # Add callbacks:
    # lr_monitor = LearningRateMonitor(logging_interval='step')


    tb_logger = pl_loggers.TensorBoardLogger('experiments/' + cfg['experiment']['id'],
                                             default_hp_metric=False)

    # Setup trainer
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="sem_loss", save_last=True)

    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      log_every_n_steps=10,
                      resume_from_checkpoint=checkpoint,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[checkpoint_callback],
                      accumulate_grad_batches=4)
    # Train
    trainer.fit(model, data)


if __name__ == "__main__":
    main()

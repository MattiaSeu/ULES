from os.path import join, dirname, abspath
import sys
import click
import pytorch_lightning as pl
import torchvision
from self_supervised_models import SelfSupervisedMethod
from model_params import VICRegParams, ModelParams
from torchsummary import summary
from pytorch_lightning.callbacks import ModelCheckpoint
from model_params import VICRegParams


@click.command()
@click.option('--data_path',
              '-d',
              type=str,
              help='path to the dataset folder',
              default="~/data")
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--loss_type',
              '-l',
              type=str,
              help='type of loss to be used',
              default="contrastive")
@click.option('--extra',
              '-e',
              is_flag=True,
              show_default=True,
              help='use the extra coarse data',
              default=True)
@click.option('--img_log',
              '-i',
              is_flag=True,
              show_default=True,
              help='adds an image logger during training to check the inputs are correct',
              default=False)
@click.option('--gpus',
              '-g',
              help='number of gpus to be used',
              default=-1)
def main(data_path, loss_type, extra, img_log, checkpoint, gpus):
    hparams = VICRegParams(
        encoder_arch='FCN_resnet50',
        dataset_name="cityscapes",
        batch_size=4,
        lr=0.001,
        optimizer_name="adam",
        embedding_dim=20,  # this number needs to match the network end embedding size (not sure if still true?)
        # mlp_hidden_dim=256,
        extra=extra,  # when true, uses extended coarse dataset instead of fine
        data_path=data_path,  # dataset root folder path
        loss_type=loss_type,
        image_log=img_log
    )
    checkpoint = "checkpoints/last.ckpt"
    model = SelfSupervisedMethod(hparams)
    summary(model)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="loss", save_last=True,
                                          every_n_epochs=5)
    trainer = pl.Trainer(gpus=gpus, max_epochs=100, callbacks=[checkpoint_callback], resume_from_checkpoint=checkpoint)
    trainer.fit(model)


if __name__ == "__main__":
    main()

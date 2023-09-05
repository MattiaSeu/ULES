from os.path import join, dirname, abspath
import click
import pytorch_lightning as pl
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
def main(data_path, loss_type, extra):
    hparams = VICRegParams(
        encoder_arch='FCN_resnet50',
        dataset_name="cityscapes",
        batch_size=8,
        lr=0.001,
        optimizer_name="adam",
        embedding_dim=2048,  # this number needs to match the network end embedding size (not sure if still true?)
        extra=extra,  # when true, uses extended coarse dataset instead of fine
        data_path=data_path,  # dataset root folder path
        loss_type=loss_type
    )

    model = SelfSupervisedMethod(hparams)
    summary(model)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="loss", save_last=True)
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    main()

import os
import pytorch_lightning as pl
from self_supervised_models import SelfSupervisedMethod
from model_params import VICRegParams, ModelParams
from torchsummary import summary
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ["DATA_PATH"] = "~/data"

from model_params import VICRegParams

hparams = VICRegParams(
    encoder_arch='FCN_resnet50',
    dataset_name="cityscapes",
    batch_size=8,
    embedding_dim=2048,  # this number needs to match the network end embedding size (not sure if still true?)
    extra=True,  # when true, uses extended coarse dataset instead of fine
)

model = SelfSupervisedMethod(hparams)
summary(model)
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="loss", save_last=True)
trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback])
trainer.fit(model)

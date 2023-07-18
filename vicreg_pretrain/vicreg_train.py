import os
import pytorch_lightning as pl
from self_supervised_models import SelfSupervisedMethod
from model_params import VICRegParams, ModelParams
from torchsummary import summary

os.environ["DATA_PATH"] = "~/data"

from model_params import VICRegParams

hparams = VICRegParams(
    encoder_arch='FCN_resnet50',
    dataset_name="cityscapes",
    batch_size=16,
    embedding_dim=2048,  # this number needs to match the network end embedding size (not sure if still true?)
    extra=False,  # when true, uses extended coarse dataset instead of fine
)

model = SelfSupervisedMethod(hparams)
summary(model)
trainer = pl.Trainer(gpus=1, max_epochs=100)
trainer.fit(model)
trainer.save_checkpoint("vicreg_pt.ckpt")

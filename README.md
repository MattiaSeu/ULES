# Unsupervised Learning Enhanced Segmentation

The provided requirements can be used to setup a conda environment with `conda env create -f requirements.yml -n env_name`

Segmentation training or finetuning is run with `train_ules.py` from root directory with options to load config params `-c`, load weights `-w`, restart
a previous training from checkpoint `-ckpt` and finetune on a smaller dataset `-redata`. Config parameters can be adjusted in `config/config.yaml`.

For pre-training, use `vicreg_train.py` in the `vicreg_pretrain` folder; parameters can be adjusted from inside the script
itself, refer to `model_params.py` for reference. 

It is assumed that the Cityscapes dataset is stored in a `data` folder on the home directory, this can be adjusted either in the 
`config` folder or the pre-training script above.
#block(name=[pixelpro_pretrain], threads=4, memory=12000, subtasks=8,  gpus=3, hours=48)

python train_ules.py --data_path "/cache/mseu" -dt_rt 100 --gpus -1 -w "unsup_pretrain/checkpoints/last-v1.ckpt"
python train_ules.py --data_path "/cache/mseu" -dt_rt 50 --gpus -1 -w "unsup_pretrain/checkpoints/last-v1.ckpt"
python train_ules.py --data_path "/cache/mseu" -dt_rt 20 --gpus -1 -w "unsup_pretrain/checkpoints/last-v1.ckpt"
python train_ules.py --data_path "/cache/mseu" -dt_rt 10 --gpus -1 -w "unsup_pretrain/checkpoints/last-v1.ckpt"
python train_ules.py --data_path "/cache/mseu" -dt_rt 100 --gpus -1
python train_ules.py --data_path "/cache/mseu" -dt_rt 50 --gpus -1
python train_ules.py --data_path "/cache/mseu" -dt_rt 20 --gpus -1
python train_ules.py --data_path "/cache/mseu" -dt_rt 10 --gpus -1

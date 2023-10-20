#block(name=[pixelpro_pretrain], threads=4, memory=12000, subtasks=1,  gpus=3, hours=48)

python pixelpro.py --data_path "/cache/mseu" --num_workers 4 --batch_size 4 -ckpt "checkpoints/last-v1.ckpt"

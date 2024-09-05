
CUDA_VISIBLE_DEVICES=4 python3 eval_recall.py --seq 00 --model ./checkpoints/model/00.ckpt --model_mae ./checkpoints/model/00_mae.ckpt --velo_desc_folder ./data/lidar_desc --img_desc_folder ./data/img_desc --pose_file ./data/pose_kitti/00.txt


CUDA_VISIBLE_DEVICES=4 python3 eval_recall.py --seq 02 --model ./checkpoints/model/02.ckpt --model_mae ./checkpoints/model/02_mae.ckpt --velo_desc_folder ./data/lidar_desc --img_desc_folder ./data/img_desc --pose_file ./data/pose_kitti/02.txt


CUDA_VISIBLE_DEVICES=4 python3 eval_recall.py --seq 05 --model ./checkpoints/model/05.ckpt --model_mae ./checkpoints/model/05_mae.ckpt --velo_desc_folder ./data/lidar_desc --img_desc_folder ./data/img_desc --pose_file ./data/pose_kitti/05.txt


CUDA_VISIBLE_DEVICES=4 python3 eval_recall.py --seq 06 --model ./checkpoints/model/06.ckpt --model_mae ./checkpoints/model/06_mae.ckpt --velo_desc_folder ./data/lidar_desc --img_desc_folder ./data/img_desc --pose_file ./data/pose_kitti/06.txt


CUDA_VISIBLE_DEVICES=4 python3 eval_recall.py --seq 08 --model ./checkpoints/model/08.ckpt --model_mae ./checkpoints/model/08_mae.ckpt --velo_desc_folder ./data/lidar_desc --img_desc_folder ./data/img_desc --pose_file ./data/pose_kitti/08.txt




 
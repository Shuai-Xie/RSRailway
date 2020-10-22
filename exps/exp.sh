# train input 要能被 4 整除
python main.py --data_dir /datasets/rs_detect/railway/train \
 --num_epoch 100 --batch_size 16 --dataset railway --phase train \
 --input_h 448 --input_w 800 \
 --exp dec_res101_epoch100_data1501

# test
python main.py --data_dir /datasets/rs_detect/DOTA/val_split \
 --resume model_50.pth --dataset dota --phase test

# eval
python main.py --data_dir dataPath --conf_thresh 0.1 --batch_size 16 --dataset dota --phase eval

# seg
python train_seg.py

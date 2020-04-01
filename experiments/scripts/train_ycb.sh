export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=3

python3 ./tools/train-gen.py --dataset ycb\
  --dataset_root /home/dell/dy/DenseFusion/datasets/ycb/YCB_Video_Dataset --nepoch 301 --workers 32 --batch_size=8 --resume_posenet pose_model_55_0.0.pth --start_epoch 56

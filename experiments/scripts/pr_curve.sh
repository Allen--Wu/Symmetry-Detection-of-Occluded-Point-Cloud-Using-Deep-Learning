
export PYTHONUNBUFFERED="True"

CUDA_VISIBLE_DEVICES=""
python3 ./tools/pr-curve.py --dataset ycb\
  --dataset_root /home/dell/dy/DenseFusion/datasets/ycb/YCB_Video_Dataset

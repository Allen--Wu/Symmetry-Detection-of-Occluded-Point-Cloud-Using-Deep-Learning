export PYTHONUNBUFFERED="True"

CUDA_VISIBLE_DEVICES=""
python3 ./tools/pr-curve-occ.py --dataset ycb\
  --dataset_root /home/dell/dy/DenseFusion/datasets/ycb/YCB_Video_Dataset

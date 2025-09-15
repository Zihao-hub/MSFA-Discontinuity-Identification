# MSFA-Discontinuity-Identification
A multi-scale feature aggregation deep learning network for simultaneous identification of discontinuity traces and planes in large-scale complex rock masses


# Install
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.6 and Python 3.7:

conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch



# Run
You can run modes with following codes.

python train_partseg.py --model rock_part_seg_msg --log_dir rock_part_seg_msg
python test_partseg.py --log_dir rock_part_seg_msg

# Reference By
https://github.com/charlesq34/pointnet

https://github.com/charlesq34/pointnet2

https://github.com/yanx27/Pointnet_Pointnet2_pytorch

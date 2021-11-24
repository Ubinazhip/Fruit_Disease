#!/bin/sh

#python3 test.py --checkpoint ./work_dir1/detectors/fold3/epoch_15.pth --test_csv my_test_epoch15.csv
#python3 ./mmdetection/tools/train.py ./mmdetection/configs/detectors/customized_detectorRs2.py --gpu-ids 3
python3 ./mmdetection/tools/train.py ./mmdetection/configs/vfnet/customized_vfnet.py --gpu-ids 3



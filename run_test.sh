#!/bin/sh




python3 ./UniverseNet/tools/test.py ./UniverseNet/configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco.py ./work_dir1/universnet/fold3_again/epoch_16.pth --eval mAP --file_name final_res16.csv

#python3 ./mmdetection/tools/test.py ./UniverseNet/configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco.py ./work_dir1/detectors/fold3_resume/epoch_25.pth --eval mAP


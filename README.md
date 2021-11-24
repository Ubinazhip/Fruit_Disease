# Fruit_Disease
clone this repository: <br/>
git clone https://github.com/Ubinazhip/Fruit_Disease.git <br/>
work inside this repository <br/>
Move the folders of train and test images to this direcory: Train_Images folder and Test_Images folder <br/>
<br/>
Create a conda virtual environment and activate it <br/>
conda create -n fruit python=3.8.5 -y <br/>
conda activate fruit

Install PyTorch and torchvision following the [https://pytorch.org/] <br/>
conda install pytorch torchvision -c pytorch <br/>
example: <br/>
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch <br/>

Install mmcv-full. <br/>
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html <br/> 
example: <br/>
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html <br/>

git clone https://github.com/shinya7y/UniverseNet.git <br/>
cd UniverseNet <br/>
pip install -r requirements/build.txt <br/>
pip install -v -e .  <br/>
inside this UniverseNet folder, there is a folder ./configs delete it and move folder configs in this repository to ./UniverseNet. In another it is kind customized configs, so we need to replace original configs folder <br/>
and the files test.py and custom_utils.py in this repository move to  UniverseNet/tools/  . Note that there is already test.py file in UniverseNet/tools/, so my test.py should replace it. <br>  

move the model https://drive.google.com/drive/folders/1AbCL1P28_KeaRW777xt7ArbK4Pv2vPSK?usp=sharing to the folder ./load_models . Note that this file is easily available online 
in https://github.com/shinya7y/UniverseNet/tree/master/configs/universenet - so we can use it as a pretrained model. <br/>
pip install pandas <br/>
pip install tqdm <br/>
pip install -U albumentations <br/>

inside this directory run in terminal : </br>
python3 ./UniverseNet/tools/train.py ./UniverseNet/configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco.py --gpu-ids 1
just put your gpu ids <br/> 
After which it should start to train <br/>

for testing: <br/>
python3 ./UniverseNet/tools/test.py ./UniverseNet/configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco.py ./work_dir1/universnet/fold3/epoch_16.pth --eval mAP --file_name test_aslan.csv <br/>
So the result will be saved in test_aslan.csv <br/>

Note that I am reproducing my second best model(I chose 2 models, as it was required), which is also should give me 4th place. I am not reproducing my best model, since I noticed some errors on it and it requires more time to converge.

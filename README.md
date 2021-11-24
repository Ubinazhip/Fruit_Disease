# Fruit_Disease

Create a conda virtual environment and activate it
conda create -n fruit python=3.8.5 -y
conda activate fruit

Install PyTorch and torchvision following the [https://pytorch.org/] <br/>
conda install pytorch torchvision -c pytorch <br/>
example <br/>
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch <br/>

Install mmcv-full. <br/>
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html <br/> 
example: <br/>
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html <br/>

git clone https://github.com/shinya7y/UniverseNet.git <br/>
cd UniverseNet <br/>
pip install -r requirements/build.txt <br/>
pip install -v -e .  <br/>

pip install pandas <br/>
pip install tqdm <br/>
pip install -U albumentations <br/>

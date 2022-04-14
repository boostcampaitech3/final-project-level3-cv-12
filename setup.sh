conda install pytorch=1.7 cudatoolkit=11.0 torchvision=0.8.0 -c pytorch
pip install openmim
mim install mmdet
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install -r requirements/build.txt
pip install -v -e .
pip install -r requirements/albu.txt

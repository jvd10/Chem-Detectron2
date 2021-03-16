#!/bin/sh
time conda install -q -y -c conda-forge rdkit
#python3 conda_path.py
pip install -U torch torchvision
pip install git+https://github.com/facebookresearch/fvcore.git
pip install absl-py==0.10.0
pip install bounded-pool-executor==0.0.3
pip install cachetools==4.1.1
pip install certifi==2020.6.20
pip install chardet==3.0.4
pip install cloudpickle==1.6.0
pip install cycler==0.10.0
pip install Cython==0.29.21
pip install future==0.18.2
pip install fvcore==0.1.2.post20201009
pip install google-auth==1.22.1
pip install google-auth-oauthlib==0.4.1
pip install grpcio==1.32.0
pip install idna==2.10
pip install importlib-metadata==2.0.0
pip install joblib==0.17.0
pip install kiwisolver==1.2.0
pip install Markdown==3.3
pip install matplotlib==3.3.2
pip install mkl-fft==1.2.0
pip install mkl-random==1.1.1
pip install mkl-service==2.3.0
pip install mock==4.0.2
pip install numpy==1.19.2
pip install oauthlib==3.1.0
pip install olefile==0.46
pip install opencv-python
pip install portalocker==2.0.0
pip install pqdm
pip install protobuf==3.13.0
pip install pyasn1==0.4.8
pip install pyasn1-modules==0.2.8
pip install pycairo==1.19.1
pip install pycocotools==2.0.2
pip install pydot==1.4.1
pip install pyparsing==2.4.7
pip install python-dateutil==2.8.1
pip install pytz==2020.1
pip install PyYAML==5.1
pip install requests==2.24.0
pip install requests-oauthlib==1.3.0
pip install rsa==4.6
pip install scikit-learn==0.23.2
pip install scipy
pip install six==1.15.0
pip install tabulate==0.8.7
pip install tensorboard==2.3.0
pip install tensorboard-plugin-wit==1.7.0
pip install termcolor==1.1.0
pip install threadpoolctl==2.1.0
pip install tqdm==4.50.2
pip install typing-extensions==3.7.4.3
pip install urllib3==1.25.10
pip install Werkzeug==1.0.1
pip install yacs==0.1.8
pip install zipp==3.3.0
pip install ipykernel
pip uninstall torch
pip uninstall torch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
# check the above link, the following is for cuda == 11.1, torch == 1.8
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

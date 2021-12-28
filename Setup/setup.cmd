python-3.9.5-amd64.exe

pip install --no-index --find-links . jupyter
pip install --no-index --find-links . torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113
pip install --no-index --find-links . gym[atari,accept-rom-license]

cuda_11.3.0_465.89_win10.exe

@REM unzip cudnn-11.3-windows-x64-v8.2.0.53.zip and copy the subfolders to CUDA sdk folder




# RNN_DVIS

![image](https://github.com/user-attachments/assets/b0cb5d5a-1368-428c-bfb9-38d495e03d64)

### 환경 설정
```
conda create --name dvis python=3.8 -y
conda activate dvis # 가상환경 세팅
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U opencv-python

# install detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd ./detectron2
pip install -e .

# install panoptic api
pip install git+https://github.com/cocodataset/panopticapi.git

git clone git@https://github.com/zhang-tao-whu/DVIS.git
cd DVIS
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
### 데이터 셋
1. SOTA를 통해 원하는 데이터 셋(ytvis19, ovis...)의 CODALAB 접속
2. participate 후 다운로드
3. 데이터 셋 등록 형식은 링크 참고

### Training
example...
```
python train_net_video.py \
  --num-gpus 8 \
  --config-file ./configs/youtubevis_2019/DVIS_Online_R50.yaml \
  --eval-only MODEL.WEIGHTS ./pretrained_model/pth_file/DVIS_online_ytvis19_r50.pth
```

### Inferencing
example...
```
python train_net_video.py \
  --num-gpus 4 \
  --config-file ./configs/youtubevis_2019/DVIS_Online_R50.yaml \
  --resume MODEL.WEIGHTS ./pretrained_model/pth_file/minvis_ytvis19_swin_large.pth
```

### Additional...
make video 코드 적어두기

# MLFlow - Train Server

## 1. Install requirements

```
bash requirements.sh
```

## 2. run `train.py`
기본적으로 AirFlow scheduler를 통해 학습이 되며 학습코드를 따로 사용하고 싶으실 때 아래와 같이 사용하시면 됩니다.

```
python train.py
```

```
optional arguments:
  -h, --help            show this help message and exit
  --batch BATCH         set batch size. default=8
  --val_batch VAL_BATCH
                        set val and test batch size. default=8
  --lr LR               set learning rate. default=5e-4
  --epochs EPOCHS       set training epoch. default=50
  --accelerator {cpu,gpu,auto}
                        set training device. default=gpu
  --precision {32,16}   set training mode. if 16, use mixed-precision. default=16
  --seed SEED           set random seed. default=42
  --regist_name REGIST_NAME
                        set regist name. default=register_model
  --bn_type BN_TYPE     set batchnorm type. default=torchbn
  --num_classes NUM_CLASSES
                        set num-classes. default=19
  --backbone BACKBONE   set model backbone. default=hrnet48
  --pretrained PRETRAINED
  --experiment_name EXPERIMENT_NAME
  --cutmix              use cutmix
```

## 3. Tracking
로컬 webserver로 접속
```
mlflow server
```
외부에서 webserver 접속

```
mlflow server -h 0.0.0.0 -p 12345
```

## 4. Register
checkpoint의 origin.pth 사용하여 best.pth 생성    
 
개선된 모델 생성시 Production 레벨로 변경
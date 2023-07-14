import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
from torchvision import datasets, transforms
import argparse
from datetime import datetime
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import seed_everything
import os
import pytz
import numpy as np
from PIL import Image
from collections import namedtuple
from dataset import CustomCityscapesSegmentation
from models.light import PLModel
import pytorch_lightning as pl


def get_arg():
    parser = argparse.ArgumentParser(description="mlflow-pytorch test")
    parser.add_argument("--batch",type=int, default=2)
    parser.add_argument("--lr",type=float, default=1e-3)
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--accelerator", choices=['cpu','gpu','auto'],default='gpu')
    parser.add_argument("--precision", choices=['32','16'],default='16')
    parser.add_argument("--seed", type=int , default=42)
    parser.add_argument("--regist_name", type=str, default="register_model")
    parser.add_argument("--bn_type", type=str, default= 'torchbn')
    parser.add_argument("--num_classes", type=int, default= 19)
    parser.add_argument("--backbone", type=str, default='hrnet48')
    parser.add_argument("--pretrained", type=str, default=None)
    
    args = parser.parse_args()
    return args


def run(args):
    seed_everything(args.seed)
    
    # 실험 이름 설정
    mlflow.set_experiment(experiment_name="mlflow_test")
    
    # 실험 환경 path 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    tracking_uri = 'file://'+os.path.join(base_path,'mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    # 실험 이름에 따라 실험 뽑기
    experiment = mlflow.get_experiment_by_name("mlflow_test")
    
    # 이전 실험들을 불러와서 제일 높은 max값 불러오기
    client = MlflowClient()
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id)
    if len(runs):
        best_runs = max(runs, key=lambda r: r.data.metrics['best_score'])
        best_score = best_runs.data.metrics['best_score']
    else:
        best_score= 0 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch
    
    train_data = CustomCityscapesSegmentation(
        data_dir = os.path.join(base_path,'data','cityscape'),
        image_set="val",
        transform = transforms.ToTensor(),
        target_transform = transforms.PILToTensor(),
        )
    test_data = CustomCityscapesSegmentation(
        data_dir = os.path.join(base_path,'data','cityscape'),
        image_set="test",
        transform = transforms.ToTensor(),
        target_transform = transforms.PILToTensor(),
        )
    
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                               batch_size = batch_size, shuffle = True,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                                batch_size = batch_size, shuffle = False,
                                               )
    
    # lighiting model 선언
    model = PLModel(args=args)
    
    
    run_name = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y%m%d%H%M")
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        
        artifact_path = run.info.artifact_uri[7:]
        # checkpoint callback 선언
        checkpoint_callback = ModelCheckpoint(
            dirpath = artifact_path, 
            save_top_k=1, monitor="val_score", mode="max",
        )
        # logger
        mlf_logger = MLFlowLogger(
            experiment_name="mlflow_test",
            run_id=run.info.run_id,
            tracking_uri=tracking_uri
        )
        
        # lightning Trainer
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=args.accelerator,
            precision=args.precision,
            logger= mlf_logger,
            callbacks=[checkpoint_callback]
            )
        
        # model train
        trainer.fit(model,train_loader,test_loader)
        
        # 현재 score뽑기
        active_run = mlflow.get_run(run_id=run.info.run_id)
        
        current_score = active_run.data.metrics['best_score']
        
        # load best model using checkpoint
        model.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, args=args)
        mlflow.pytorch.log_model(
            pytorch_model = model, 
            artifact_path = "model",
            )
        
        # retrain시 best보다 좋아졌다면 register version
        if current_score > best_score:
            
            # 모델 저장
            now = mlflow.register_model(
                # model log가 있는 곳에서
                model_uri = 'model',
                name=args.regist_name,
                tags = {'score': current_score}
                )
            
            # 현재 버전 Production
            client.transition_model_version_stage(
                name=args.regist_name,
                stage = 'Production',
                version=now.version
                )
            print(f"model '{args.regist_name}' {now.version} is changing 'Production'")
            
            # 마지막 버전 가져오기 -> return list
            last_version = client.get_latest_versions(
                                                    name=args.regist_name,
                                                    stages=['Production'])[0]
            
            pre_version = int(last_version.version)-1
            # 이전 버전이 있었다면 staging변경
            if pre_version > 0:
                # 이전 버전 Staging
                client.transition_model_version_stage(
                    name=args.regist_name,
                    stage = "Staging",
                    version=pre_version
                    )
                print(f"pre-version {pre_version} is changing 'Staging'")
            


if __name__=='__main__':
    args = get_arg()
    run(args)
    
    
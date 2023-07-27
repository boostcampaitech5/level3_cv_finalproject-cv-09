import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import argparse
from datetime import datetime
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import seed_everything, request_weight
import os
import pytz
from models.light import PLModel
import pytorch_lightning as pl
from dataset.dataset import get_data_loader


def get_arg():
    parser = argparse.ArgumentParser(description="mlflow-pytorch test")
    parser.add_argument("--batch",type=int, default=8)
    parser.add_argument("--val_batch",type=int, default=8)
    parser.add_argument("--lr",type=float, default=5e-4)
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--accelerator", choices=['cpu','gpu','auto'],default='gpu')
    parser.add_argument("--precision", choices=['32','16'],default='16')
    parser.add_argument("--seed", type=int , default=42)
    parser.add_argument("--regist_name", type=str, default="register_model")
    parser.add_argument("--bn_type", type=str, default= 'torchbn')
    parser.add_argument("--num_classes", type=int, default= 19)
    parser.add_argument("--backbone", type=str, default='hrnet48')
    parser.add_argument("--pretrained", type=str, default='/opt/ml/level3_cv_finalproject-cv-09/MLflow/checkpoint/origin.pth')
    parser.add_argument("--experiment_name",type=str, default='krload')
    parser.add_argument('--cutmix', action='store_true', help="use cutmix")
    
    args = parser.parse_args()
    return args


def run(args):
    seed_everything(args.seed)
    
    # 실험 환경 path 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    tracking_uri = 'file://'+os.path.join(base_path,'mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    
    # clinet 설정
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # experiment 생성
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name=args.experiment_name)
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
    experiment_id = experiment.experiment_id
    
    # regist name의 production level의 score 뽑기
    registered_models = mlflow.search_model_versions()
    best_score = 0
    pre_version = -1
    if registered_models:
        for m in registered_models:
            if m.current_stage == 'Production' and m.name == args.regist_name:
                best_score = float(m.tags['score'])
                pre_version = int(m.version)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data loader 가져오기
    train_loader, val_loader, test_loader = get_data_loader(args)
    
    # lighiting model 선언
    model = PLModel(args=args)
    
    # start mlflow logging
    run_name = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y%m%d%H%M")
    with mlflow.start_run(experiment_id=experiment_id,run_name=run_name) as run:
        mlflow.pytorch.autolog()
        print(f"running in {run.info.run_id}")
        artifact_path = run.info.artifact_uri[7:]
        # checkpoint callback 선언
        checkpoint_callback = ModelCheckpoint(
            dirpath = artifact_path, 
            save_top_k=1, monitor="val_score", mode="max",
        )
        # logger
        mlf_logger = MLFlowLogger(
            experiment_name=args.experiment_name,
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
        if args.epochs:
            trainer.fit(model,train_loader,val_loader)
            
            # load best model using checkpoint
            model.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, args=args)
        
        # model test
        trainer.test(model,test_loader)
        # 현재 score뽑기
        active_run = mlflow.get_run(run_id=run.info.run_id)
        
        current_score = active_run.data.metrics['test_score']
        
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
            
            # checkpoint에 weight저장
            save_path = args.pretrained.replace("origin","best")
            torch.save({"state_dict":model.net.state_dict()}, f=save_path)
            
            request_weight(save_path)
            
            # 이전 버전이 있었다면 staging변경
            if pre_version > -1:
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
    
    
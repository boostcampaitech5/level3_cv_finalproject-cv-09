import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import argparse
import pytorch_lightning as pl
import torchmetrics
from datetime import datetime
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import seed_everything
import os
import pytz

def get_arg():
    parser = argparse.ArgumentParser(description="mlflow-pytorch test")
    parser.add_argument("--batch",type=int, default=16)
    parser.add_argument("--lr",type=float, default=1e-3)
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--accelerator", choices=['cpu','gpu','auto'],default='gpu')
    parser.add_argument("--precision", choices=['32','16'],default='16')
    parser.add_argument("--seed", type=int , default=42)
    parser.add_argument("--regist_name", type=str, default="register_model")
    
    args = parser.parse_args()
    return args


class PLModel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.net = torchvision.models.mobilenet_v2(num_classes = 10)
        self.args = args
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.best_score = 0
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self,x):
        return self.net(x)

    # on_epoch는 epoch 마다 on_step은 step 마다
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits,y)
        pred = logits.argmax(dim=1)
        
        self.train_acc(pred,y)
        self.training_step_outputs.append(loss)
        # self.log("train_loss",loss,on_epoch=True,on_step=False)
        # self.log("train_acc",self.train_acc,on_epoch=True,on_step=False)
        return loss
    
    def on_train_epoch_end(self) -> None:
        all_outs = torch.stack(self.training_step_outputs)
        acc = self.train_acc.compute()
        
        self.log("train_loss",all_outs.mean().item())
        self.log("train_acc",acc.detach().cpu())
        self.training_step_outputs.clear()
        self.train_acc.reset()
        
    
    def validation_step(self,batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits,y)
        pred = logits.argmax(dim=1)
        
        self.val_acc(pred,y)
        self.validation_step_outputs.append(loss)
        # self.log("val_loss",loss,on_epoch=True,on_step=False)
        # self.log("val_acc",self.val_acc,on_epoch=True,on_step=False)
        
        return loss
    
        
    def on_validation_epoch_end(self) -> None:
        all_outs = torch.stack(self.validation_step_outputs)
        acc = self.val_acc.compute()
        self.best_score = max(self.best_score,acc)
        
        self.log("val_loss",all_outs.mean().item())
        self.log("val_acc",acc.detach().cpu())
        self.log("best_score",self.best_score)
        
        self.validation_step_outputs.clear()
        self.val_acc.reset()
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.args.lr)


def run(args):
    seed_everything(args.seed)
    
    # 실험 환경 path 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    tracking_uri = 'file://'+os.path.join(base_path,'mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    
    # 실험 이름 설정
    mlflow.set_experiment(experiment_name="mlflow_test")
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
    
    train_data = datasets.CIFAR10(root = os.path.join(base_path,'data'),
                            train = True,
                            download= True,
                            transform = transforms.ToTensor())
    test_data = datasets.CIFAR10(root = os.path.join(base_path,'data'),
                            train = False,
                            transform = transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                               batch_size = batch_size, shuffle = True,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                                batch_size = batch_size, shuffle = False,
                                                num_workers=8)
    
    # lighiting model 선언
    model = PLModel(args=args)
    
    
    run_name = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y%m%d%H%M")
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        
        artifact_path = run.info.artifact_uri[7:]
        # checkpoint callback 선언
        checkpoint_callback = ModelCheckpoint(
            dirpath = artifact_path, 
            save_top_k=1, monitor="val_acc", mode="max",
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
    
    
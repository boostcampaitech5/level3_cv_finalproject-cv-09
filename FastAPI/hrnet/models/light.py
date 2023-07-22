from models.hrnet import HRNet_W48_OCR
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn


class PLModel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.net = HRNet_W48_OCR(args)
        self.args = args
        self.train_score = torchmetrics.Dice(multiclass=True,num_classes=args.num_classes)
        self.val_score = torchmetrics.Dice(multiclass=True,num_classes=args.num_classes)
        self.best_score = 0
        
        self.training_step_outputs = []
        self.validation_step_outputs = []

        if args.pretrained:
            self.load_model(args.pretrained)
    
    def load_model(self,path):
        state_dict = torch.load(path)['state_dict']
        self.net.load_state_dict(state_dict)
    
    def forward(self,x):
        return self.net(x)

    # on_epoch는 epoch 마다 on_step은 step 마다
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).type(dtype=torch.long)
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits,y)
        pred = logits.argmax(dim=1)
        
        self.train_score(pred,y)
        self.training_step_outputs.append(loss)
        # self.log("train_loss",loss,on_epoch=True,on_step=False)
        # self.log("train_score",self.train_score,on_epoch=True,on_step=False)
        return loss
    
    def on_train_epoch_end(self) -> None:
        all_outs = torch.stack(self.training_step_outputs)
        score = self.train_score.compute()
        
        self.log("train_loss",all_outs.mean().item())
        self.log("train_score",score.detach().cpu())
        self.training_step_outputs.clear()
        self.train_score.reset()
        
    
    def validation_step(self,batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).type(torch.long)
        logits = self.forward(x)
        print(type(logits), type(y))
        loss = nn.functional.cross_entropy(logits,y)
        pred = logits.argmax(dim=1)
        
        self.val_score(pred,y)
        self.validation_step_outputs.append(loss)
        # self.log("val_loss",loss,on_epoch=True,on_step=False)
        # self.log("val_score",self.val_score,on_epoch=True,on_step=False)
        
        return loss
    
        
    def on_validation_epoch_end(self) -> None:
        all_outs = torch.stack(self.validation_step_outputs)
        score = self.val_score.compute()
        self.best_score = max(self.best_score,score)
        
        self.log("val_loss",all_outs.mean().item())
        self.log("val_score",score.detach().cpu())
        self.log("best_score",self.best_score)
        
        self.validation_step_outputs.clear()
        self.val_score.reset()
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.args.lr)
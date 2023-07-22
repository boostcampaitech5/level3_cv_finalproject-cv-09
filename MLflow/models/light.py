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
        # self.train_score = torchmetrics.Dice(multiclass=True,num_classes=args.num_classes)
        self.train_score = torchmetrics.JaccardIndex(task='multiclass',num_classes=args.num_classes)
        self.val_score = torchmetrics.JaccardIndex(task='multiclass',num_classes=args.num_classes,average='none')
        self.test_score = torchmetrics.JaccardIndex(task='multiclass',num_classes=args.num_classes,average='none')
        
        self.training_step_outputs = []
        self.validation_step_outputs = []

        if args.pretrained:
            self.load_model(args.pretrained)
    
    def load_model(self,path):
        state_dict = torch.load(path)['state_dict']
        self.net.load_state_dict(state_dict)
    
    def forward(self,x):
        return self.net(x)

    # train , on_epoch는 epoch 마다 on_step은 step 마다
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).type(dtype=torch.long)
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits,y)
        pred = logits.argmax(dim=1)
        
        self.train_score(pred.detach(),y.detach())
        self.training_step_outputs.append(loss.detach().cpu())
        
        return loss
        
    def on_train_epoch_end(self) -> None:
        all_outs = torch.stack(self.training_step_outputs)
        score = self.train_score.compute()
        
        self.log("train_loss",all_outs.mean())
        self.log("train_score",score.mean())
        self.training_step_outputs.clear()
        self.train_score.reset()
        
    # validation
    def validation_step(self,batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).type(torch.long)
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits,y)
        pred = logits.argmax(dim=1)
        
        self.val_score(pred.detach(),y.detach())
        self.validation_step_outputs.append(loss.detach().cpu())
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        all_outs = torch.stack(self.validation_step_outputs)
        score = self.val_score.compute()
        
        self.log("val_loss",all_outs.mean())
        self.log("val_score",score.mean())
        self.log_dict({f"class{key}":value.item() for key,value in enumerate(score)})
        
        self.validation_step_outputs.clear()
        self.val_score.reset()
        
    # test
    def test_step(self,batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).type(torch.long)
        logits = self.forward(x)
        pred = logits.argmax(dim=1)
        
        self.test_score(pred.detach(),y.detach())
        
    def on_test_epoch_end(self) -> None:
        score = self.test_score.compute()
        
        self.log("test_score",score.mean())
        self.log_dict({f"test class{key}":value.item() for key,value in enumerate(score)})
        self.test_score.reset()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.args.lr)
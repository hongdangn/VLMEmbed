import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        
    def forward(self, model_trainer, input_data):
        model = model_trainer.model
            
        qry_reps = model.encode_input(input_data['qry'])
        pos_reps = model.encode_input(input_data['pos'])
        
        scores = model.compute_similarity(qry_reps, pos_reps)
        scores = scores.view(qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (qry_reps.size(0) // pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / model_trainer.temperature, target)
        
        return {
            "loss": contrastive_loss,
            "contrastive_loss": contrastive_loss,
        }
    
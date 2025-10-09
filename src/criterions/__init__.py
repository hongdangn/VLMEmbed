# from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .contrastive_loss import ContrastiveLoss

criterion_list = {
    # "contrastive_rkd": ContrastiveLossWithRKD,
    "contrastive_loss": ContrastiveLoss
}

def build_criterion(args):
    if not args.criterion_type:
        raise ValueError(f"{args.criterion_type} is not specified")
    
    return criterion_list[args.criterion_type](args)


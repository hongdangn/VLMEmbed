import json
from prepare_one_model_training import TrainOneModelCollator, OneModelTrainer, TrainOneModelDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
from src import model
from src.utils import print_rank, print_master
from src.criterions import build_criterion
import time 
import os
import sys
from tqdm import tqdm 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.optim import AdamW

import deepspeed
import wandb 
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig

def push_to_hub(repo_name=None, token=None, commit_message="Upload model", 
                local_dir="./temp_model", private=False):
    try:
        if not repo_name:
            raise ValueError("must specify a repo name to push to hub")
        
        if not os.path.exists(local_dir):
            raise ValueError(f"local_dir {local_dir} does not exist")
        
        print_rank(f"Pushing model to the hub at {repo_name}...")
        api = HfApi()
        create_repo(repo_name, token=token, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_name, 
            token=token, 
            commit_message=commit_message
        )

        print_rank(f"Model has been pushed to the hub at: {repo_name}")
        return True
        
    except Exception as e:
        print_rank(f"Error pushing to hub: {str(e)}")
        return False


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def finetune(
    model_args: ModelArguments, 
    data_args: DataArguments,
    training_args: TrainingArguments,
    model_trainer: OneModelTrainer, 
    train_dataset: TrainOneModelDataset,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    collator: TrainOneModelCollator,
    criterion: nn.Module,
):
    print_rank("Start finetuning...")
    start_time = time.time()
    
    is_distributed = dist.is_initialized()
    if is_distributed:
        sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=collator,
            sampler=sampler, 
            drop_last=True,
            pin_memory=True,
        )   
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=collator,
            shuffle=True, 
            drop_last=True,
            pin_memory=True,
        )
    
    model = model_trainer.model
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            try:
                p.data = p.data.to(torch.bfloat16)
            except Exception:
                # If bf16 not supported, ignore and continue
                pass
    
    # print_rank(next(model_engine.parameters()).device)
    model.train()
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank(f"Number of parameters in student model: {num_trainable}")

    
    step = 0
    logging_output = {
        'epoch': 0, 
        'global_step': 0, 
        'loss': [],
        'contrastive_loss': [],
        'micro_step_time': [],
        'step_time': []
    }
    
    for epoch in range(training_args.num_train_epochs):
        logging_output['epoch'] = epoch + 1
        print_rank("Start iteration of epoch {}".format(epoch + 1))
        end_epoch = False
        epoch_step = 0
        epoch_loss, epoch_contrastive_loss, epoch_kd_loss = 0, 0, 0
        losses, contrastive_losses, kd_losses = [], [], []
        
        if is_distributed and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        
        
        progress_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}",
        )
        
        train_iter = iter(train_dataloader)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            
        for batch in progress_bar:
            optimizer.zero_grad()
            device_batch = {key: batch_to_device(val, device) 
                                            for key, val in batch.items()}
            st_time = time.time()
            
            loss_dict = model_trainer(criterion, device_batch)
            
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            contrastive_loss = loss_dict['contrastive_loss']
            # kd_loss = loss_dict['kd_loss']
            
            losses.append(loss_dict['loss'].detach().item())
            contrastive_losses.append(contrastive_loss.detach().item())
            # kd_losses.append(kd_loss.detach().item())
            logging_output['micro_step_time'].append(time.time() - st_time)
            
            step += 1
            epoch_step += 1
            logging_output['global_step'] = step
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            batch_loss = sum(losses)/len(losses)
            batch_contrastive_loss = sum(contrastive_losses)/len(contrastive_losses)
            # batch_kd_loss = sum(kd_losses)/len(kd_losses)
                        
            progress_bar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "contrastive_loss": f"{batch_contrastive_loss:.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
            })
            progress_bar.update(1)
                # if "wandb" in training_args.report_to:
                #     wandb.log({
                #         "train/loss": batch_loss,
                #         "train/contrastive_loss": batch_contrastive_loss,
                #         # "train/kd_loss": batch_kd_loss,
                #         "train/lr": lr_scheduler.get_last_lr()[0],
                #         "train/epoch": epoch + 1,
                #         "train/global_step": step,
                #     })
                    
                #     logging_output['micro_step_time'] = []
                #     logging_output['step_time'] = []
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_contrastive_loss = epoch_contrastive_loss / len(train_dataloader)
        # avg_kd_loss = epoch_kd_loss / max(1, epoch_step)
        
        print_rank(
            f"Epoch {epoch + 1} completed. Avg Loss: {avg_epoch_loss:.4f} | "
            f"Avg Contrastive Loss: {avg_contrastive_loss:.4f} "
        )
            
            # if "wandb" in training_args.report_to:
            #     wandb.log({
            #         "epoch/avg_loss": avg_epoch_loss,
            #         "epoch/avg_contrastive_loss": avg_contrastive_loss,
            #         # "epoch/avg_kd_loss": avg_kd_loss,
            #         "epoch/epoch": epoch + 1,
            #     })
            # Save checkpoint
        if training_args.save_strategy == "epoch":
            ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            model.encoder.save_pretrained(ckpt_dir)                
            print_rank(f"Checkpoint saved at {ckpt_dir}")
            
            model_config = AutoConfig.from_pretrained(model_args.model_name) if model_args.model_name else None
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name) if model_args.model_name else None
            processor = AutoProcessor.from_pretrained(model_args.model_name) if model_args.model_name else None
            model_config.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)

            print_rank(f"Epoch {epoch + 1} finished.")
    total_time = time.time() - start_time
    print_rank(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save final model
    if training_args.save_strategy == "epoch":
        final_ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-final")
        os.makedirs(final_ckpt_dir, exist_ok=True)

        model.encoder.save_pretrained(final_ckpt_dir)
        print_rank(f"Final model saved at {final_ckpt_dir}")
        
        if model_args.model_name:
            model_config = AutoConfig.from_pretrained(model_args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
            processor = AutoProcessor.from_pretrained(model_args.model_name)
            model_config.save_pretrained(final_ckpt_dir)
            tokenizer.save_pretrained(final_ckpt_dir)
            processor.save_pretrained(final_ckpt_dir)

        # if "wandb" in training_args.report_to:
        #     wandb.finish()

    return logging_output

def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(data_args.subset_name)
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    train_dataset = TrainOneModelDataset(data_args, model_args)
    print_rank(f"Number of training samples: {len(train_dataset)}")
    model_trainer = OneModelTrainer(model_args, 
                                    training_args, 
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    collator = TrainOneModelCollator(
        processor=model_trainer.get_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    optimizer = AdamW(
        model_trainer.model.parameters(),
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # Initialize learning rate scheduler
    steps_per_epoch = len(train_dataset) // (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    )
    total_steps = steps_per_epoch * training_args.num_train_epochs
        
    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps ,
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    else:
        # Default constant learning rate
        from transformers import get_constant_schedule
        lr_scheduler = get_constant_schedule(optimizer)
        
    criterion = build_criterion(training_args)
    print(model_args.model_backbone)
    
    logging_output = finetune(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        model_trainer=model_trainer,
        train_dataset=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        collator=collator,
        criterion=criterion,
    )
    
    print_rank("Training completed successfully!")
    return logging_output

if __name__ == "__main__":
    main()
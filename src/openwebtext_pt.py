import os
import json
import argparse
import sys
sys.path.append(".")
from collections import defaultdict

import wandb
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
import numpy as np
import tiktoken
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import itertools

from utils.util import (
    set_seed,
    get_lr_scheduler,
    load_model,
    delete_old_checkpoint,
    load_checkpoint,
    save_checkpoint,
    setup_logger,
    wandb_log
)

from utils.models import (
    create_attention_mask,
    DownstreamLanguageModel,
    DownstreamLlamaLM,
    create_llama_model
)
from utils.training_args import OpenWebTextTrainingArgs, create_openwebtext_parser, args_to_dataclass
from utils.dataset_utils import OpenWebTextDataset

# Initialize logger
log = setup_logger('openwebtext_train')


def build_dataloader(args, split="train"):
    """
    Build dataloader for OpenWebText dataset using binarized data files
    """
    if not args.data_dir:
        raise ValueError("--data_dir must be specified for OpenWebText training")
    
    log.info(f"Using binarized OpenWebText data from {args.data_dir}")
    
    # Create dataset
    dataset = OpenWebTextDataset(
        data_dir=args.data_dir,
        split=split,
        block_size=args.new_seq_len if args.new_seq_len is not None else args.pt_seq_len,
        max_samples=args.max_samples
    )
    
    # Initialize tiktoken tokenizer (GPT-2) for consistency
    enc = tiktoken.get_encoding("gpt2")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(split == "train"),
    )
    
    return dataloader, enc

def eval_epoch(args, model, dataloader):
    # compute in-context loss on the dataset
    model.eval()

    criterion = CrossEntropyLoss(reduction='none')
    
    total_loss = 0.0
    count = 0
    # Determine dtype for autocast based on mixed_precision setting
    autocast_dtype = None
    if args.mixed_precision == 'bf16':
        autocast_dtype = torch.bfloat16
    elif args.mixed_precision == 'fp16':
        autocast_dtype = torch.float16

    base_mask = create_attention_mask(args.new_seq_len if args.new_seq_len is not None else args.pt_seq_len, additive=True).to(args.device) # pre-compute mask 
    contextual_loss = torch.zeros(args.new_seq_len if args.new_seq_len is not None else args.pt_seq_len)
    num_batches = 0

    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(args.device)
            targets = targets.to(args.device)
            
            B = input_ids.shape[0]
            attention_mask = base_mask.repeat(B, 1, 1, 1).to(args.device)
            
            with torch.autocast(device_type=args.device.type, enabled=args.autocast, dtype=autocast_dtype):
                logits = model(input_ids, attention_mask=attention_mask)
                
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(logits, targets) # compute loss with reduction='none'
            
            loss = loss.reshape(B, -1)
            loss = torch.sum(loss, dim=0)
            contextual_loss += loss.detach().cpu().numpy()
            num_batches += B
    
    return contextual_loss / num_batches

def val_epoch(args, model, dataloader, criterion):
    model.eval()
    
    total_loss = 0.0
    count = 0
    base_mask = create_attention_mask(args.new_seq_len if args.new_seq_len is not None else args.pt_seq_len, additive=True).to(args.device) # pre-compute mask

    # Determine dtype for autocast based on mixed_precision setting
    autocast_dtype = None
    if args.mixed_precision == 'bf16':
        autocast_dtype = torch.bfloat16
    elif args.mixed_precision == 'fp16':
        autocast_dtype = torch.float16

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validation", unit="batch", leave=False, ncols=80)
        for input_ids, targets in val_pbar:
            input_ids = input_ids.to(args.device)
            targets = targets.to(args.device)
            
            B = input_ids.shape[0]
            attention_mask = base_mask.repeat(B, 1, 1, 1).to(args.device)
            
            with torch.autocast(device_type=args.device.type, enabled=args.autocast, dtype=autocast_dtype):
                # Forward pass with causal mask
                logits = model(input_ids, attention_mask=attention_mask)
                
                # Reshape for loss calculation
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                # Calculate loss
                loss = criterion(logits, targets)
            
            total_loss += loss.detach().cpu().item()
            count += 1
            
            # Update validation progress bar
            avg_val_loss = total_loss / count
            val_pbar.set_postfix({'val_loss': f'{avg_val_loss:.6f}'})
    
    return total_loss / count

# wandb_log is now imported from utils.util

# -------------------------- Main Script --------------------------

def init_args(args: OpenWebTextTrainingArgs) -> OpenWebTextTrainingArgs:
    """Initialize arguments with runtime setup"""
    # Convert device to torch.device
    args.to_device()
    
    # Ensure save directory exists
    args.save_dir = os.path.join(args.save_dir)
    args.set_runtime_paths()
    
    # Load wandb info from previous run if resuming
    if os.path.exists(args.metrics_path) and not args.wandb_resume_run_id and args.wandb_enable and args.resume:
        with open(args.metrics_path, 'r') as f:
            metrics = json.load(f)
        args.wandb_name = metrics['wandb_name']
        args.wandb_resume_run_id = metrics['wandb_run_id']

    if args.owt_pretraining:
        args.interval_path = os.path.join(args.save_dir, "interval_save")
        if not os.path.exists(args.interval_path):
            os.makedirs(args.interval_path)
        else:
            log.info(f"Interval save directory already exists: {args.interval_path}")
    return args

def main(args: OpenWebTextTrainingArgs):
    # Initialize args
    args = init_args(args)
    set_seed(args.seed)
    
    log.info(f"Starting OpenWebText training with seed: {args.seed}")
    log.info(f"Device: {args.device}")
    log.info(f"Save directory: {args.save_dir}")
    log.info(f"Data directory: {args.data_dir}")
    log.info(f"Mixed precision: {args.mixed_precision}")
    log.info(f"Autocast enabled: {args.autocast}")
    
    # Initialize save path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        log.info(f"Created save directory: {args.save_dir}")
    
    # Initialize wandb
    if args.wandb_enable:
        if args.wandb_resume_run_id is None:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=args.__dict__
            )
            log.info(f"Initialized new wandb run: {args.wandb_name}")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=args.__dict__,
                id=args.wandb_resume_run_id,
                resume="allow"
            )
            log.info(f"Resumed wandb run: {args.wandb_name} (ID: {args.wandb_resume_run_id})")
    
    # Load and create model for OpenWebText language modeling
    if args.model_type == 'gpt2':
        model_arc = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
        log.info(f"Creating model with architecture: {model_arc}")
        pretrain_model = create_gpt2_model(args.model_name, vocab_size=args.pt_vocab_size, seq_length=args.pt_seq_len, params=model_arc)
    elif args.model_type == 'llama':
        pretrain_model = create_llama_model(
            vocab_size=args.pt_vocab_size,
            seq_length=args.pt_seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd
        )
    
    # Load pretrained weights if specified
    assert not (args.pretrain and args.owt_pretraining), "Cannot fine-tune and do OWT pretraining at the same time"

    if args.pretrain and not args.pretrained_from_owt:
        pretrain_model = load_model(pretrain_model, args.model_path, args.model_file, pretrain_pos='pos' in args.reinit_modules)
        log.info('Pretrained model loaded')
    
    # Apply LoRA if enabled
    if args.lora == 1:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            use_rslora=True
        )
        pretrain_model = get_peft_model(pretrain_model, lora_config)
    
    # Load and create dataloaders
    train_loader, tokenizer = build_dataloader(args, "train")
    val_loader, _ = build_dataloader(args, "validation")

    if args.model_type == 'gpt2':
        # Create language modeling head
        model = DownstreamLanguageModel(
                    gpt=pretrain_model,
                    vocab_size=tokenizer.n_vocab, # Use tiktoken vocab size
                    seq_length=args.new_seq_len,
                    frozen_modules=args.freeze_modules,
                    reinit_modules=args.reinit_modules+['embed'],
                    weight_tying=args.weight_tying == 1,
                )
    elif args.model_type == 'llama':
        model = DownstreamLlamaLM(
                    model=pretrain_model,
                    vocab_size=tokenizer.n_vocab,
                    frozen_modules=args.freeze_modules,
                    reinit_modules=args.reinit_modules+['embed'],
                    weight_tying=args.weight_tying == 1,
                    fixed_embed_init=getattr(args, 'fixed_embed_init', 0) == 1,
                )
    
    if args.pretrain and args.pretrained_from_owt:
        model = load_model(model, args.model_path, args.model_file, pretrain_pos='pos' in args.reinit_modules, strict=False)
        log.info('Pretrained model loaded from checkpoint')
        log.info(f"Warning: Strict loading is disabled for pretrained model but vocab is the same / encoding is the same")
        log.info(f"Warning: Embedding weights are truncated to 50257")
    
    # Handle reinitialization based on reinit_modules list
    if 'attn' in args.reinit_modules:
        model.reinit_attention_weights(args.reinit_layer_idxs) # reinitialize attention weights
    if 'mlp' in args.reinit_modules:
        model.reinit_mlp_weights(args.reinit_layer_idxs) # reinitialize MLP weights
    if 'ln' in args.reinit_modules:
        model.reinit_layer_norm_weights(args.reinit_layer_idxs) # reinitialize layer norm weights
    if 'embed' in args.reinit_modules:
        model.reinit_embeddings()
    
    if args.lora == 1:
        model.enable_lora()
    
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model architecture:\n{model}")
    log.info(f"Total parameters: {total_params:,}")
    
    log.info(f"Training on {len(train_loader)} batches per epoch")
    log.info(f"Validation on {len(val_loader)} batches per epoch")
    log.info(f"Validation frequency: every {args.val_freq} iterations")
    
    # Set up training loop
    criterion = CrossEntropyLoss()  # No need to ignore padding tokens since we don't use padding

    # Set up effective training steps
    effective_train_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps

    if args.owt_pretraining:
        effective_train_steps_per_epoch = args.max_pretraining_steps

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=int(effective_train_steps_per_epoch * args.warmup),
        total_steps=int(effective_train_steps_per_epoch * args.epochs),
        decrease_mode=args.lr_decay_mode
    )
    
    # Initialize stage tracking for two-stage training
    current_stage = 1
    current_iterations = 0
    total_iterations = len(train_loader) * args.epochs
    stage_transition_point = int(total_iterations * args.two_stage_training)

    if args.owt_pretraining:
        total_iterations = args.max_pretraining_steps # we will override the total iterations to only do pretraining on the last X Iterations
        stage_transition_point = total_iterations # we will not do any stage transition
    
    # Set up initial stage 1 configuration
    if args.two_stage_training > 0:
        log.info("Setting up stage 1: unfreezing embeddings and layer norms, freezing core")
        model._unfreeze_embs()  # unfreeze embeddings
        model._freeze_core()    # freeze core
        model._unfreeze_ln()    # unfreeze layer norm
    
    # Initialize metrics
    count = 0
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_loss_el = float('inf')
    metrics = {}
    metrics['train/loss'] = []
    metrics['val/loss'] = []
    metrics['val/loss_mean'] = []
    metrics['train/loss_mean'] = []
    
    if args.patience == -1:
        args.patience = float('inf')
    patience = args.patience

    # Load checkpoint if resuming from checkpoint, but not resuming the metrics and re-starting training from here
    if args.resume_from_checkpoint:
        model, *_ = load_checkpoint(
            model, None, None, args.model_path, args.model_file, args.device
        )
        # Determine current stage and set up accordingly
        if current_iterations >= stage_transition_point:
            current_stage = 2
            log.info("Resuming from checkpoint in stage 2")
            model._freeze_unfreeze_modules()
        else:
            current_stage = 1
            log.info("Resuming from checkpoint in stage 1")
            model._unfreeze_embs()
            model._freeze_core()
            model._unfreeze_ln()
        
        # Test (remove later)
        if 'embed' in args.reinit_modules:
            model.reinit_embeddings()
    
    # Load checkpoint if resuming
    if args.resume:
        model, optimizer, scheduler, count, _, best_val_loss, best_val_loss_el, metrics, current_iterations, _ = load_checkpoint(
            model, optimizer, scheduler, args.save_dir, args.load_dir, args.device, total_iterations=True
        )
        log.info(f"Loading checkpoint from iteration {current_iterations}")
        # Determine current stage based on iteration count
        if current_iterations >= stage_transition_point:
            current_stage = 2
            model._freeze_unfreeze_modules()
            log.info("Resuming in stage 2")
        else:
            current_stage = 1
            model._unfreeze_embs()
            model._freeze_core()
            model._unfreeze_ln()
            log.info("Resuming in stage 1")

    # Store wandb info in metrics
    if args.wandb_enable:
        metrics['wandb_name'] = args.wandb_name
        metrics['wandb_run_id'] = wandb_run.id

    # Determine dtype for autocast based on mixed_precision setting
    autocast_dtype = None
    if args.mixed_precision == 'bf16':
        autocast_dtype = torch.bfloat16
    elif args.mixed_precision == 'fp16':
        autocast_dtype = torch.float16
    
    # Compile model
    if args.device.type != "mps" and torch.__version__ >= "2.0" and torch.cuda.device_count() == 1:
        compiled_model = torch.compile(model)
        torch._inductor.config.debug = False
        torch._inductor.config.verbose_progress = False
        torch._inductor.config.triton.unique_kernel_names = True
    else:
        compiled_model = model
    
    compiled_model.to(args.device)
    log.info(f"device count: {torch.cuda.device_count()}")

    if torch.cuda.device_count() > 1:
        log.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        compiled_model = DataParallel(compiled_model)
    else:
        log.info(f"Using single GPU/CPU: {args.device}")
    
    # Training loop
    log.info(f"training on {len(train_loader)} batches")
    
    # Training with validation every X iterations
    compiled_model.train()
    
    if args.autocast:
        scaler = torch.GradScaler(device=args.device.type)
    
    epoch_loss = 0.0
    batch_count = 0
    accumulated_loss = 0.0
    accumulation_step = 0
    base_mask = create_attention_mask(args.new_seq_len if args.new_seq_len is not None else args.pt_seq_len, additive=True).to(args.device) # pre-compute mask
    
    log.info(f"current_iterations: {current_iterations}")

    # Create progress bar for training loop
    if args.owt_pretraining:
        train_loader = itertools.islice(train_loader, len(train_loader) - int(args.max_pretraining_steps * args.gradient_accumulation_steps), None)

    elif args.resume and current_iterations > 0:
        train_loader = itertools.islice(train_loader, current_iterations * args.gradient_accumulation_steps, None)

    pbar = tqdm(
        enumerate(train_loader), 
        total=(effective_train_steps_per_epoch-current_iterations) * args.gradient_accumulation_steps, 
        desc="Training",
        unit="batch",
        leave=True,
        ncols=100
    )

    if args.iteration_overide is not None:
        total_iterations = args.iteration_overide
    
    for itr_idx, (input_ids, targets) in pbar:
        # Check for stage transition
        if current_iterations >= stage_transition_point and current_stage == 1:
            log.info(f"Transitioning to stage 2 at iteration {current_iterations}")
            compiled_model._freeze_unfreeze_modules()  # freeze unfreeze modules
            current_stage = 2
            pbar.set_description(f"Training (Stage {current_stage})")

        input_ids = input_ids.to(args.device)
        targets = targets.to(args.device)

        B = input_ids.shape[0]
        attention_mask = base_mask.repeat(B, 1, 1, 1).to(args.device)
        
        with torch.autocast(device_type=args.device.type, enabled=args.autocast, dtype=autocast_dtype):
            # Forward pass with causal mask
            logits = compiled_model(input_ids, attention_mask=attention_mask)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(logits, targets)
        
        # Accumulate loss for logging (unscaled)
        accumulated_loss += loss.detach().cpu().item()
        accumulation_step += 1
        
        if args.autocast:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if accumulation_step % args.gradient_accumulation_steps == 0:
            if args.autocast:
                if args.grad_clip_enable or args.log_grad:
                    scaler.unscale_(optimizer)
                
                if args.grad_clip_enable:
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                
                if args.log_grad and current_iterations % args.log_grad_freq == 0:
                    grad_norm = torch.norm(
                        torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
                    )
                    wandb_log({'iteration': current_iterations, 'train/grad_norm': grad_norm.detach().cpu().item(), "lr": scheduler.get_last_lr()[0]}, args)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_enable:
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                
                if args.log_grad and current_iterations % args.log_grad_freq == 0:
                    grad_norm = torch.norm(
                        torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
                    )
                    wandb_log({'iteration': current_iterations, 'train/grad_norm': grad_norm.detach().cpu().item(), "lr": scheduler.get_last_lr()[0]}, args)
                
                optimizer.step()

            # log accumulated loss
            avg_accumulated_loss = accumulated_loss / args.gradient_accumulation_steps
            wandb_log({'iteration': current_iterations, 'train/loss': avg_accumulated_loss}, args)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
            epoch_loss += avg_accumulated_loss
            batch_count += 1
            current_iterations += 1

            # Reset accumulation
            accumulated_loss = 0.0 
            
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = epoch_loss / batch_count
            pbar.set_postfix({
                'loss': f'{loss.detach().cpu().item():.6f}',
                'avg_loss': f'{avg_loss:.6f}',
                'lr': f'{current_lr:.2e}',
                'stage': current_stage
            })

            if args.owt_pretraining and current_iterations in args.pt_save_interval:
                save_checkpoint(
                    current_iterations, count, model, optimizer, scheduler,
                    best_val_loss, best_val_loss_el, metrics, args.interval_path, total_iterations=current_iterations
                )
        
        # Run validation every X iterations
        if accumulation_step % args.gradient_accumulation_steps == 0 and current_iterations % args.val_freq == 0:
            val_loss = val_epoch(args, compiled_model, val_loader, criterion)
            log.info(f"Iteration {current_iterations} | train loss: {avg_accumulated_loss:.8f} | val loss: {val_loss:.8f}")
            
            # Update progress bar with validation loss
            pbar.set_postfix({
                'loss': f'{loss.detach().cpu().item():.6f}',
                'avg_loss': f'{avg_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'lr': f'{current_lr:.2e}',
                'stage': current_stage
            })
            
            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                log.info(f"New best validation loss: {val_loss:.8f}")
                save_checkpoint(
                    current_iterations, count, model, optimizer, scheduler,
                    best_val_loss, best_val_loss_el, metrics, args.save_dir, best=True, 
                    total_iterations=current_iterations
                )
                delete_old_checkpoint(args.save_dir)
            
            # Save latest checkpoint
            save_checkpoint(
                current_iterations, count, model, optimizer, scheduler,
                best_val_loss, best_val_loss_el, metrics, args.save_dir, total_iterations=current_iterations
            )
            delete_old_checkpoint(args.save_dir)
            
            # Save metrics
            if os.path.exists(args.metrics_path):
                os.remove(args.metrics_path)

            os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
            
            metrics['train/loss_mean'].append({current_iterations: epoch_loss / batch_count})
            metrics['val/loss_mean'].append({current_iterations: val_loss})

            with open(args.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            wandb_log({"iteration": current_iterations, "train/loss_mean": epoch_loss / batch_count, "val/loss_mean": val_loss}, args)
            
            # Early stopping
            if val_loss - best_val_loss_el >= 1e-10:
                count += 1
                if count > patience:
                    log.info(f"Early stop at iteration {current_iterations}")
                    wandb_log({"iteration": current_iterations, "early stop": current_iterations}, args)
                    break
            else:
                count = 0
                best_val_loss_el = val_loss
            # Update progress bar with current metrics     

    # Log epoch summary
    avg_epoch_loss = epoch_loss / batch_count
    log.info(f"Training completed | average train loss: {avg_epoch_loss:.8f}")
    
    wandb.finish()

def eval_icl(args):
    # load a trained OWT model and evaluate on the in-context learning tasks
    # Set device
    args.device = torch.device(args.device)

    train_loader, tokenizer = build_dataloader(args, "train")
    val_loader, _ = build_dataloader(args, "validation")
    # Build model just like in main
    if args.model_type == 'gpt2':
        model_arc = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
        pretrain_model = create_gpt2_model(args.model_name, vocab_size=args.pt_vocab_size, seq_length=args.pt_seq_len, params=model_arc)
        model = DownstreamLanguageModel(
            gpt=pretrain_model,
            vocab_size=args.pt_vocab_size,
            seq_length=args.new_seq_len if args.new_seq_len else args.pt_seq_len,
            frozen_modules=args.freeze_modules,
            reinit_modules=args.reinit_modules,
            weight_tying=args.weight_tying == 1,
        )
    elif args.model_type == 'llama':
        base_model = create_llama_model(
            vocab_size=tokenizer.n_vocab,
            seq_length=args.new_seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )
        model = DownstreamLlamaLM(
            model=base_model,
            vocab_size=tokenizer.n_vocab,
            frozen_modules=args.freeze_modules,
            reinit_modules="embed",
            weight_tying=args.weight_tying == 1,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Use the pre-existing load_checkpoint utility to load the weights (we don't need optimizer, scheduler etc for eval)
    # This will load the latest checkpoint in the directory
    model, *_ = load_checkpoint(
        model=model,
        optimizer=None,
        scheduler=None,
        save_dir=args.model_path,
        load_dir=args.model_file,
        device=args.device
    )
    model.to(args.device)
    model.eval()
    log.info(f"Loaded fine-tuned OWT model for ICL evaluation from {args.model_path}/{args.model_file}")

    # Evaluate using eval_epoch
    contextual_loss = eval_epoch(args, model, val_loader).tolist()

    log.info(f"ICL evaluation complete. Contextual loss (averaged over evaluation set): {contextual_loss}")

    if os.path.exists(f"{args.save_dir}/icl_graph"):
        os.remove(f"{args.save_dir}/icl_graph/contextual_loss.json")
    
    os.makedirs(f"{args.save_dir}/icl_graph", exist_ok=True)   
    with open(f"{args.save_dir}/icl_graph/contextual_loss.json", "w") as f:
        json.dump(contextual_loss, f)
        
if __name__ == "__main__":
    # Create parser and parse arguments
    parser = create_openwebtext_parser()
    parsed_args = parser.parse_args()
    
    # Convert to dataclass
    training_args = args_to_dataclass(parsed_args, is_v2l=False)
    
    # Run main script
    if training_args.icl_eval:
        eval_icl(training_args)
    else:
        main(training_args) 
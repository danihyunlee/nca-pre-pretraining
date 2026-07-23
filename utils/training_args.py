"""
Shared training arguments dataclass for training scripts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import argparse
import torch

import os

#############################################################
#  Base Training Arguments - Hierarchical Structure
#############################################################

@dataclass
class BaseTrainingArgs:
    """
    Base dataclass for all training arguments.
    Contains core fields common to all training scenarios.
    """
    # Core settings
    seed: int = 0
    device: Union[str, int] = 'cuda:0'
    distributed: bool = False
    num_workers: int = 2

    # Wandb logging (identical across all classes)
    wandb_project: str = ''
    wandb_enable: bool = False
    wandb_name: Optional[str] = None
    wandb_resume_run_id: Optional[str] = None

    # Precision settings (identical across all classes)
    compile: bool = False
    autocast: bool = False
    mixed_precision: str = 'none'  # options = 'bf16', 'fp16', 'none'

    def __post_init__(self):
        """Post-initialization validation"""
        # Convert device to string format if it's an int
        if isinstance(self.device, int):
            self.device = f'cuda:{self.device}'

        # Validate mixed_precision setting
        if self.mixed_precision not in ['bf16', 'fp16', 'none']:
            raise ValueError("mixed_precision must be one of: 'bf16', 'fp16', 'none'")

        # Enable autocast automatically if mixed_precision is specified
        if self.mixed_precision != 'none':
            self.autocast = True

        assert not (self.autocast is True and self.mixed_precision == 'none'), \
            "autocast cannot be enabled if mixed_precision is none"

    def to_device(self):
        """Convert device string to torch.device"""
        if isinstance(self.device, str):
            self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        return self.device

    def set_runtime_paths(self):
        """Set runtime paths - override in subclasses"""
        pass


@dataclass
class ModelTrainingArgs(BaseTrainingArgs):
    """
    Extended base class for model training with architecture and hyperparameters.
    Extends BaseTrainingArgs with model-specific fields.
    """
    # Model architecture (standardized names)
    model_type: str = 'llama'
    n_layer: int = 24
    n_head: int = 32
    n_embd: int = 2048

    # Module management (identical across all model training classes)
    freeze_modules: List[str] = field(default_factory=list)
    reinit_modules: List[str] = field(default_factory=list)
    reinit_layer_idxs: List[int] = field(default_factory=lambda: [0, 0])
    weight_tying: int = 0
    fixed_embed_init: int = 0

    # Training hyperparameters
    batch_size: int = 16
    val_freq: int = 500
    epochs: int = 50
    lr: float = 1e-4
    lr_decay_mode: str = 'cosin'
    weight_decay: float = 0.00
    warmup: float = 1  # epoch
    patience: int = -1

    # Gradient control
    grad_clip: float = 1.0
    grad_clip_enable: bool = False
    log_grad: bool = False
    log_grad_freq: int = 100

    # Training state
    resume: bool = False
    resume_from_checkpoint: bool = False

    # Paths (standardized)
    save_dir: str = ''
    load_dir: Optional[str] = None
    model_path: str = ''
    model_file: Optional[str] = None

    # Runtime attributes
    metrics_path: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Extended post-initialization"""
        # Call parent __post_init__
        super().__post_init__()

        # Ensure reinit_layer_idxs is a list of exactly 2 elements
        if len(self.reinit_layer_idxs) != 2:
            raise ValueError("reinit_layer_idxs must contain exactly 2 elements [start, end]")

    def set_runtime_paths(self):
        """Set runtime paths based on save_dir"""
        if self.save_dir:
            self.metrics_path = f"{self.save_dir}/metrics.json"

            # Set default wandb_name if not provided
            if not self.wandb_name:
                self.wandb_name = f'{self.save_dir}'


#############################################################
#  Legacy Training Arguments (Backward Compatibility)
#############################################################

@dataclass
class TrainingArgs:
    """
    Dataclass for universal training arguments for all scripts
    """
    # SEEDS & DEVICE
    seed: int = 0
    device: Union[str, int] = 'cuda:0'
    distributed: bool = False
    num_workers: int = 2
    wandb_project: str = ''
    wandb_enable: bool = False
    wandb_name: Optional[str] = None
    wandb_resume_run_id: Optional[str] = None

    # PRECISION
    compile: bool = False
    autocast: bool = False
    mixed_precision: str = 'none' # options = 'bf16', 'fp16', 'none'

    # MODEL CONFIGURATIONS
    model_type: str = 'llama'
    model_name: str = 'llama-large'
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 2048

    freeze_modules: List[str] = field(default_factory=list)
    reinit_modules: List[str] = field(default_factory=list)
    reinit_layer_idxs: List[int] = field(default_factory=lambda: [0, 0])
    weight_tying: int = 0
    fixed_embed_init: int = 0

    # HYPERPARAMETERS (GENERAL)
    batch_size: int = 16
    val_freq: int = 500
    epochs: int = 50

    lr: float = 1e-4
    lr_decay_mode: str = 'cosin'
    weight_decay: float = 0.00
    warmup: float = 1 # epoch
    patience: int = -1
    grad_clip: float = 1.0
    grad_clip_enable: bool = False
    log_grad: bool = False
    log_grad_freq: int = 100
    resume: bool = False
    resume_from_checkpoint: bool = False

### ORIGINAL TRAINING ARGUMENTS ###

@dataclass
class OpenWebTextTrainingArgs(ModelTrainingArgs):
    """
    Dataclass for OpenWebText training arguments.
    Extends ModelTrainingArgs with OWT-specific fields.
    """
    # Data settings
    data_dir: str = ""  # Required, will be validated
    max_samples: Optional[int] = None

    # Model parameters
    vocab_size: int = 50257  # GPT-2 default
    new_seq_len: Optional[int] = None
    model_path: str = ''
    model_file: Optional[str] = None
    pt_vocab_size: int = 64000  # Aligned to launcher defaults
    pt_seq_len: int = 1024  # Aligned to launcher defaults
    pretrain: int = 1
    pretrained_from_owt: int = 0

    # Model name override defaults
    model_name: str = 'tiny'

    # V2L-specific parameters
    v2l_model_name: str = 'openai/imagegpt-large'

    # Training hyperparameters - override defaults
    epochs: int = 1
    warmup: float = 0.1
    val_freq: int = 100
    gradient_accumulation_steps: int = 32
    log_grad_freq: int = 10
    grad_clip_enable: int = 0

    # LoRA parameters
    lora: int = 0
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # System settings - override defaults
    save_dir: str = 'data/model/openwebtext'
    load_dir: str = ''
    wandb_project: str = 'OpenWebText'

    # Checkpointing settings
    checkpoint_every_n_iterations: int = 500
    keep_last_n_checkpoints: int = 3
    save_best_checkpoint: bool = True
    interval_save: bool = False
    iteration_overide: Optional[int] = None

    # OWT pretraining
    owt_pretraining: bool = False
    max_pretraining_steps: int = 50_000
    pt_save_interval: List[int] = field(default_factory=lambda: [2_500, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000])

    # Training options
    two_stage_training: float = 0.0
    icl_eval: bool = False

    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Call parent __post_init__
        super().__post_init__()

    # Inherits set_runtime_paths() and to_device() from ModelTrainingArgs


def create_parser(for_v2l: bool = False) -> argparse.ArgumentParser:
    """
    Create argument parser for OpenWebText training.

    Args:
        for_v2l: If True, includes V2L-specific parameters and defaults

    Returns:
        Configured argument parser
    """
    if for_v2l:
        description = "Vision2Language transfer learning on OpenWebText"
        default_project = "Vision2Language-OpenWebText"
        default_save_dir = "data/model/v2l_openwebtext"
        default_device = "cuda:0"
        default_device_type = str
        default_warmup = 0.1
    else:
        description = "Transfer learning on OpenWebText language modeling"
        default_project = "OpenWebText"
        default_save_dir = "data/model/openwebtext"
        default_device = 0
        default_device_type = int
        default_warmup = 0.1

    parser = argparse.ArgumentParser(description=description)

    # Basic settings
    parser.add_argument('--seed', type=int, default=0, help='rng seed')
    parser.add_argument('--data_dir', type=str, required=True, help='directory containing binarized OpenWebText data files (train.bin, val.bin)')
    parser.add_argument('--max_samples', type=int, default=None, help='maximum number of samples to use')

    # V2L-specific model parameter (only for V2L)
    if for_v2l:
        parser.add_argument('--v2l_model_name', type=str, default='openai/imagegpt-large', help='vision2language model name')
        parser.add_argument('--vocab_size', type=int, default=50257, help='vocabulary size for language modeling (GPT-2 default)')

    # Standard model parameters (only for non-V2L)
    if not for_v2l:
        parser.add_argument('--model_type', type=str, default='llama', help='model type')
        parser.add_argument('--model_path', type=str, default='', help='file location for pre-trained model')
        parser.add_argument('--model_file', type=str, default=None, help='file name for the model')
        parser.add_argument('--pt_vocab_size', type=int, default=64000, help='vocabulary size for pre-trained model')  # Aligned to launcher defaults
        parser.add_argument('--pt_seq_len', type=int, default=1024, help='sequence length for training')  # Aligned to launcher defaults
        parser.add_argument('--pretrain', type=int, default=1, help='load pretrained model weights')

        # pretrained from owt
        parser.add_argument('--pretrained_from_owt', type=int, default=0)

        # Model architecture (only for standard training)
        parser.add_argument('--model_name', type=str, default='tiny', help='model name')
        parser.add_argument('--n_layer', type=int, default=24, help='number of layers')  # Aligned to launcher defaults
        parser.add_argument('--n_head', type=int, default=32, help='number of heads')  # Aligned to launcher defaults
        parser.add_argument('--n_embd', type=int, default=2048, help='embedding dimension')  # Aligned to launcher defaults

    # Common model configuration
    parser.add_argument('--new_seq_len', type=int, default=None, help='sequence length for training')
    parser.add_argument('--freeze_modules', type=str, nargs='*', default=[],
                       help='list of modules to freeze: pos, attn, mlp, ln, core, embs')
    parser.add_argument('--reinit_modules', type=str, nargs='*', default=[],
                       help='list of modules to reinitialize: pos, attn, mlp, ln, core')
    parser.add_argument('--reinit_layer_idxs', type=int, nargs=2, default=[0, 0],
                       help='layer indices to reinitialize (start end) -- e.g., --reinit_layer_idxs 0 4')
    parser.add_argument('--weight_tying', type=int, default=0, help='enable weight tying for input and output projections')
    parser.add_argument('--fixed_embed_init', type=int, default=0, help='init fresh input/output projections with normal(0, 0.02) instead of constructor defaults')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')  # Aligned to launcher defaults
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='number of steps to accumulate gradients before optimizer step')  # Aligned to launcher defaults
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--lr_decay_mode', type=str, default='cosin', choices=['cosin', 'linear', 'const'], help='learning rate decay mode')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('--patience', type=int, default=-1, help='early stopping patience (-1 for no early stopping)')
    parser.add_argument('--warmup', type=float, default=default_warmup, help='warmup epochs for scheduler (can also be fraction of total steps)')
    parser.add_argument('--val_freq', type=int, default=100, help='validation frequency (every X iterations)')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay for training')

    # LoRA parameters
    parser.add_argument('--lora', type=int, default=0, help='enable LoRA')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--iteration_overide', type=int, default=None, help='iteration overide')
    parser.add_argument('--pt_save_interval', type=int, nargs='*', default=[2_500, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000], help='interval save interval')

    # System settings
    parser.add_argument('--device', type=default_device_type, default=default_device,
                       help='device for training (CUDA device index for standard, device string for V2L)')
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='save directory')
    parser.add_argument('--resume', action='store_true', help='resume previous training')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='resume from checkpoint (not resuming metrics)')
    parser.add_argument('--load_dir', type=str, default='', help='file location for latest checkpoint')

    # Wandb logging
    parser.add_argument('--wandb_project', type=str, default=default_project)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_enable', action='store_true', help='enable wandb')
    parser.add_argument('--wandb_resume_run_id', type=str, default=None)

    # OWT pretraining
    parser.add_argument('--owt_pretraining', action='store_true', help='enable OWT pretraining')
    parser.add_argument('--max_pretraining_steps', type=int, default=50_000, help='maximum number of steps for pretraining')

    # Checkpointing settings
    parser.add_argument('--checkpoint_every_n_iterations', type=int, default=500,
                       help='save checkpoint every N effective iterations')
    parser.add_argument('--keep_last_n_checkpoints', type=int, default=3,
                       help='keep only the last N checkpoints (excluding best checkpoint)')
    parser.add_argument('--save_best_checkpoint', action='store_true', default=True,
                       help='whether to save best validation checkpoint separately')
    parser.add_argument('--interval_save', action='store_true', default=False,
                       help='save checkpoint every N iterations')

    # Training options
    parser.add_argument('--autocast', action='store_true', help='enable autocast')
    parser.add_argument('--mixed_precision', type=str, default='none', choices=['bf16', 'fp16', 'none'],
                       help='mixed precision training type: bf16 (bfloat16), fp16 (float16), or none (disabled)')
    parser.add_argument('--log_grad', default=0, type=int, help='enable gradient reporting')
    parser.add_argument('--log_grad_freq', type=int, default=10, help='frequency of gradient reporting')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--grad_clip_enable', default=0, type=int, help='enable gradient clipping')
    parser.add_argument('--2-stage-training', dest='two_stage_training', type=float, default=0.0,
                       help='ratio of total iterations for embedding-only training (0.0 disables two-stage training)')

    parser.add_argument('--icl_eval', action='store_true', help='enable ICL evaluation')

    return parser


def create_openwebtext_parser() -> argparse.ArgumentParser:
    """Create argument parser for standard OpenWebText training"""
    return create_parser(for_v2l=False)


def create_v2l_parser() -> argparse.ArgumentParser:
    """Create argument parser for Vision2Language training"""
    return create_parser(for_v2l=True)


def args_to_dataclass(args: argparse.Namespace, is_v2l: bool = False) -> OpenWebTextTrainingArgs:
    """
    Convert argparse.Namespace to OpenWebTextTrainingArgs dataclass

    Args:
        args: Parsed arguments from argparse
        is_v2l: Whether this is for V2L training (affects some field mappings)

    Returns:
        OpenWebTextTrainingArgs instance
    """
    # Create a dictionary from args, handling special cases
    args_dict = vars(args).copy()

    # Handle V2L-specific mappings
    if is_v2l:
        # Map v2l_model_name to model_name for consistency
        if 'v2l_model_name' in args_dict:
            args_dict['v2l_model_name'] = args_dict['v2l_model_name']

        # Set vocab_size as pt_vocab_size for consistency
        if 'vocab_size' in args_dict:
            args_dict['pt_vocab_size'] = args_dict['vocab_size']
            args_dict['vocab_size'] = args_dict['vocab_size']

    # Filter out any keys that aren't in the dataclass
    dataclass_fields = {f.name for f in OpenWebTextTrainingArgs.__dataclass_fields__.values() if f.init}
    filtered_dict = {k: v for k, v in args_dict.items() if k in dataclass_fields}

    # Create and return the dataclass
    training_args = OpenWebTextTrainingArgs(**filtered_dict)
    training_args.set_runtime_paths()
    return training_args


def dataclass_to_args(training_args: OpenWebTextTrainingArgs) -> argparse.Namespace:
    """
    Convert OpenWebTextTrainingArgs dataclass back to argparse.Namespace

    Args:
        training_args: OpenWebTextTrainingArgs instance

    Returns:
        argparse.Namespace with all the fields
    """
    return argparse.Namespace(**training_args.__dict__)

# ===== Arguments for NCA pre-training ==== #

@dataclass
class NCATrainingArgs(ModelTrainingArgs):
    """
    Dataclass for NCA training arguments.
    Extends ModelTrainingArgs with NCA-specific fields.
    """
    # NCA parameters
    identity_bias: float = 0.0
    temperature: float = 0.0
    grid: int = 12
    num_colors: int = 10

    filter_rules: bool = False
    filter_rules_threshold: float = 0.4
    filter_rules_upper_bound: Optional[float] = None
    filter_rules_mode: str = 'shannon'
    filter_rules_percentile: bool = False

    # data generation parameters
    seq_len: Optional[int] = None
    patch: int = 3
    dT: int = 1
    train_num_rules: int = 5000
    train_num_sim: int = 500
    val_num_rules: int = 2000
    val_num_sim: int = 100
    min_grid: int = 1
    generate_train: bool = False
    vocab_size: int = 16384
    input_vocab_size: Optional[int] = None
    init_rollout_steps: int = 0

    # representation parameters
    token: bool = False

    # model parameters - override defaults
    model_name: str = 'llama-1.3b'
    n_head: int = 16  # Override parent default
    mask_prob: float = 0.0
    input_bias: float = 0.1
    reinit_modules: List[str] = field(default_factory=lambda: ['embed'])

    # training parameters - override defaults
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 3e-5
    warmup: float = 10
    iteration_overide: Optional[int] = None
    grad_accumulation_steps: int = 1

    # evaluation parameters
    eval_enable: bool = False
    eval_num_rules: int = 20
    eval_num_sim: int = 10
    eval_min_grids: int = 1
    eval_freq: int = 5
    eval_mode: bool = False
    eval_dir: str = 'data/model/nca/eval'
    eval_num_examples: int = 3
    eval_icl_step: int = 1

    # interval saving
    interval_save: bool = False
    intervals: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # wandb - override defaults
    wandb_project: str = 'nca-pretraining'

    # Other
    generate_rules: int = 1

    def __post_init__(self):
        # Call parent __post_init__
        super().__post_init__()

        # NCA-specific validation
        if not self.save_dir:
            raise ValueError("save_dir is required")

    def set_runtime_paths(self):
        """Set runtime paths with NCA-specific additions"""
        # Call parent implementation first
        super().set_runtime_paths()

        # NCA-specific paths
        if self.interval_save:
            self.interval_save_path = f"{self.save_dir}/interval_save"
            if not os.path.exists(self.interval_save_path):
                os.makedirs(self.interval_save_path)

    # Inherits to_device() from ModelTrainingArgs

def create_nca_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--identity_bias', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--grid', type=int, default=12)
    parser.add_argument('--num_colors', type=int, default=10)
    parser.add_argument('--vocab_size', type=int, default=16384)
    parser.add_argument('--input_vocab_size', type=int, default=None)
    parser.add_argument('--init_rollout_steps', type=int, default=0) # number of initial steps

    parser.add_argument('--filter_rules', action='store_true', default=False)
    parser.add_argument('--filter_rules_threshold', type=float, default=0.4)
    parser.add_argument('--filter_rules_upper_bound', type=float, default=None)
    parser.add_argument('--filter_rules_mode', type=str, default='shannon')
    parser.add_argument('--filter_rules_percentile', action='store_true', default=False)

    parser.add_argument('--token', action='store_true', default=False)

    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--patch', type=int, default=3)
    parser.add_argument('--dT', type=int, default=1)
    parser.add_argument('--train_num_rules', type=int, default=5000)
    parser.add_argument('--train_num_sim', type=int, default=500)
    parser.add_argument('--val_num_rules', type=int, default=2000)
    parser.add_argument('--val_num_sim', type=int, default=100)
    parser.add_argument('--min_grid', type=int, default=1)

    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--model_name', type=str, default='llama-1.3b')
    parser.add_argument('--n_layer', type=int, default=24)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_embd', type=int, default=2048)
    parser.add_argument('--mask_prob', type=float, default=0.0)
    parser.add_argument('--input_bias', type=float, default=0.1)

    parser.add_argument('--freeze_modules', type=str, nargs='*', default=[], choices=['', 'pos', 'core', 'core-attn', 'core-mlp', 'core-ln', 'core-attn-ln', 'embs', 'ln'], help='list of modules to freeze: pos, core, core-attn (this means core minus attention), core-mlp (this means core minus mlp), core-ln (this means core minus layer norm), core-attn-ln (this means core minus attention and layer norm), embs, ln')
    parser.add_argument('--reinit_modules', type=str, nargs='*', default=['embed'], choices=['', 'embed', 'pos', 'attn', 'mlp', 'ln', 'core'], help='list of modules to reinitialize: embed, pos, attn, mlp, ln, core')
    parser.add_argument('--reinit_layer_idxs', type=int, nargs=2, default=[0, 0], help='layer indices to reinitialize (start end) -- e.g., --reinit_layer_idxs 0 4')
    parser.add_argument('--weight_tying', type=int, default=0, help='enable weight tying')
    parser.add_argument('--fixed_embed_init', type=int, default=0, help='init fresh input/output projections with normal(0, 0.02) instead of constructor defaults')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_freq', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--warmup', type=float, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grad_clip_enable', action='store_true', default=False)
    parser.add_argument('--log_grad', action='store_true', default=False)
    parser.add_argument('--log_grad_freq', type=int, default=100)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)

    # regeneration parameters
    parser.add_argument('--generate_train', action='store_true', default=False)
    parser.add_argument('--generate_rules', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--load_dir', type=str, default=None)

    parser.add_argument('--eval_enable', action='store_true', default=False)
    parser.add_argument('--eval_num_rules', type=int, default=20)
    parser.add_argument('--eval_num_sim', type=int, default=10)
    parser.add_argument('--eval_min_grids', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=5, help='frequency of evaluation')

    # evaluation parameters
    parser.add_argument('--eval_mode', action='store_true', default=False)
    parser.add_argument('--eval_num_examples', type=int, default=3) # number of examples to plot
    parser.add_argument('--eval_dir', type=str, default='data/model/nca/eval')
    parser.add_argument('--eval_icl_step', type=int, default=1)

    parser.add_argument('--wandb_project', type=str, default='nca-pretraining')
    parser.add_argument('--wandb_enable', action='store_true', default=False)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_resume_run_id', type=str, default=None)

    parser.add_argument('--interval_save', action='store_true', default=False)
    parser.add_argument('--intervals', type=int, nargs='*', default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--autocast', action='store_true', default=False)
    parser.add_argument('--mixed_precision', type=str, default='none', choices=['bf16', 'fp16', 'none'],
                       help='mixed precision training type: bf16 (bfloat16), fp16 (float16), or none (disabled)')

    return parser

def nca_args_to_dataclass(args: NCATrainingArgs):
    args_dict = vars(args).copy()

    # Map CLI argument names to standardized dataclass field names
    if 'learning_rate' in args_dict:
        args_dict['lr'] = args_dict.pop('learning_rate')
    if 'num_epochs' in args_dict:
        args_dict['epochs'] = args_dict.pop('num_epochs')

    # filter out any keys that aren't in the dataclass
    dataclass_fields = {f.name for f in NCATrainingArgs.__dataclass_fields__.values() if f.init}
    filtered_dict = {k: v for k, v in args_dict.items() if k in dataclass_fields}

    # Create and return the dataclass
    training_args = NCATrainingArgs(**filtered_dict)
    return training_args

def nca_dataclass_to_args(training_args: NCATrainingArgs) -> argparse.Namespace:
    return argparse.Namespace(**training_args.__dict__)

# ===== Arguments for Language Fine-Tuning ==== #

@dataclass
class BaseFTArgs(ModelTrainingArgs):
    """
    Base class for fine-tuning arguments.
    Extends ModelTrainingArgs with FT-specific fields (pretrain vocab, LoRA).

    Note: save_path/load_path renamed to save_dir/load_dir (inherited from ModelTrainingArgs).
    Parser mappings handle backward compatibility for CLI flags.
    """
    # Pretrain model parameters
    pt_vocab_size: int = 16384
    pretrain: int = 1

    # LoRA parameters
    lora: int = 0
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Module management - override default
    reinit_modules: List[str] = field(default_factory=lambda: ['embed'])

    # Evaluation
    eval_enable: bool = False

def create_base_ft_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--wandb_project', type=str, default='')
    parser.add_argument('--wandb_enable', action='store_true', default=False)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_resume_run_id', type=str, default=None)

    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--autocast', action='store_true', default=False)
    parser.add_argument('--mixed_precision', type=str, default='none', choices=['bf16', 'fp16', 'none'])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_freq', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay_mode', type=str, default='cosin', choices=['cosin', 'linear', 'const'], help='learning rate decay mode')
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--warmup', type=float, default=1)
    parser.add_argument('--patience', type=int, default=-1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grad_clip_enable', action='store_true', default=False)
    parser.add_argument('--log_grad', action='store_true', default=False)
    parser.add_argument('--log_grad_freq', type=int, default=100)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=False)

    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--n_layers', type=int, default=24)
    parser.add_argument('--n_heads', type=int, default=32)
    parser.add_argument('--n_embed', type=int, default=2048)
    parser.add_argument('--pt_vocab_size', type=int, default=16384)
    parser.add_argument('--pretrain', type=int, default=1)

    parser.add_argument('--lora', type=int, default=0)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    parser.add_argument('--freeze_modules', type=str, nargs='*', default=[], choices=['', 'pos', 'core', 'core-attn', 'core-mlp', 'core-ln', 'core-attn-ln', 'embs', 'ln'], help='list of modules to freeze: pos, core, core-attn (this means core minus attention), core-mlp (this means core minus mlp), core-ln (this means core minus layer norm), core-attn-ln (this means core minus attention and layer norm), embs, ln')
    parser.add_argument('--reinit_modules', type=str, nargs='*', default=['embed'], help='list of modules to reinitialize: embed, pos, attn, mlp, ln, core')
    parser.add_argument('--reinit_layer_idxs', type=int, nargs=2, default=[0, 0], help='layer indices to reinitialize (start end) -- e.g., --reinit_layer_idxs 0 4')
    parser.add_argument('--weight_tying', type=int, default=0, help='enable weight tying')
    parser.add_argument('--fixed_embed_init', type=int, default=0, help='init fresh input/output projections with normal(0, 0.02) instead of constructor defaults')

    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--metrics_path', type=str, default='')
    parser.add_argument('--eval_enable', action='store_true', default=False)
    return parser

@dataclass
class LanguageTrainingArgs(BaseFTArgs):
    """
    Dataclass for Language Training arguments
    """
    # general data parameters
    task: str = 'dyck'
    tokenizer_type: str = 'owt'
    data_dir: str = ''
    seq_len: int = 1024
    vocab_size: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    grad_accumulation_steps: int = 1

    # data generation parameters
    num_train: int = int(1e5)
    num_val: int = int(1e3)

    # champ parameters
    n_bit: int = 24
    n_champ_int: int = 255

    # dyck parameters
    generate_dyck: bool = False
    num_symbols: int = 64
    n: int = int(1e6)
    target_length: int = 1024
    min_depth: int = 1
    max_depth: int = 16
    p_open: float = 0.5
    interval_save: bool = False
    intervals: List[int] = field(default_factory=lambda: [2500, 5000, 10000, 20000, 30000, 40000, 50000])
    save_freq: int = 500

    # bigbench-lite parameters
    n_shot: List[int] = field(default_factory=lambda: [0, 5])
    max_samples: int = 200
    min_samples: int = 100

    def __post_init__(self):
        # Call parent __post_init__
        super().__post_init__()

        # Language-specific validation
        if not self.save_dir:
            raise ValueError("save_dir is required")

    # Inherits set_runtime_paths() and to_device() from BaseFTArgs/ModelTrainingArgs

def create_language_ft_parser() -> argparse.ArgumentParser:
    parser = create_base_ft_parser()
    parser.add_argument('--task', type=str, default='dyck', choices=['champ', 'dyck', 'shuffle_dyck', 'codeparrot', 'math', 'full-codeparrot', 'gsm8k', 'metamathqa', 'bigbench-lite', 'c4'])
    parser.add_argument('--tokenizer_type', type=str, default='owt', choices=['owt', 'math', 'code'])
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--num_train', type=int, default=int(1e5))
    parser.add_argument('--num_val', type=int, default=int(1e3))
    parser.add_argument('--steps_per_epoch', type=int, default=None) # maximum number of steps per epoch
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=500)

    # champernowne constant parameters
    parser.add_argument('--n_bit', type=int, default=24)
    parser.add_argument('--n_champ_int', type=int, default=255)
    parser.add_argument('--interval_save', action='store_true', default=False)
    parser.add_argument('--intervals', type=int, nargs='*', default=[2500, 5000, 10000, 20000, 30000, 40000, 50000])

    # dyck parameters
    parser.add_argument('--generate_dyck', action='store_true', default=False)
    parser.add_argument('--num_symbols', type=int, default=64)
    parser.add_argument('--n', type=int, default=int(1e6))
    parser.add_argument('--target_length', type=int, default=1024)
    parser.add_argument('--min_depth', type=int, default=1)
    parser.add_argument('--max_depth', type=int, default=16)
    parser.add_argument('--p_open', type=float, default=0.5)
    parser.add_argument('--vocab_size', type=int, default=None)

    # instruction ft parameters
    parser.add_argument('--n_shot', nargs=2, type=int, default=[0, 5])
    parser.add_argument('--min_samples', type=int, default=100)
    parser.add_argument('--max_samples', type=int, default=200)
    return parser

def language_ft_args_to_dataclass(args: LanguageTrainingArgs):
    args_dict = vars(args).copy()

    # Map CLI argument names to standardized dataclass field names
    field_mappings = {
        'n_layers': 'n_layer',
        'n_heads': 'n_head',
        'n_embed': 'n_embd',
        'save_path': 'save_dir',
        'load_path': 'load_dir'
    }

    for old_name, new_name in field_mappings.items():
        if old_name in args_dict:
            args_dict[new_name] = args_dict.pop(old_name)

    # Filter out runtime attributes (init=False fields) and non-dataclass fields
    dataclass_fields = {f.name for f in LanguageTrainingArgs.__dataclass_fields__.values() if f.init}
    filtered_dict = {k: v for k, v in args_dict.items() if k in dataclass_fields}

    # Create and return the dataclass
    training_args = LanguageTrainingArgs(**filtered_dict)
    return training_args

def language_ft_dataclass_to_args(training_args: LanguageTrainingArgs) -> argparse.Namespace:
    return argparse.Namespace(**training_args.__dict__)

@dataclass
class MathEvalArgs(BaseFTArgs):
    """
    Dataclass for Math Evaluation arguments
    """

    data_dir: str = ''
    seq_len: int = 1024
    vocab_size: Optional[int] = None
    pretrained_tokenizer: str="owt"
    temperature: float = 0.0
    top_p: float = 1.0
    passes: int = 64
    stop_string: str = '####'
    max_len: int = 200
    start_idx: int = 0
    end_idx: Optional[int] = None
    eval_passes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

def create_math_eval_parser() -> argparse.ArgumentParser:
    parser = create_base_ft_parser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--pretrained_tokenizer', type=str, default="owt")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--passes', type=int, default=64)
    parser.add_argument('--eval_passes', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--stop_string', type=str, default='####')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    return parser

def math_eval_args_to_dataclass(args: MathEvalArgs):
    args_dict = vars(args).copy()

    # Map CLI argument names to standardized dataclass field names
    field_mappings = {
        'n_layers': 'n_layer',
        'n_heads': 'n_head',
        'n_embed': 'n_embd',
        'save_path': 'save_dir',
        'load_path': 'load_dir'
    }

    for old_name, new_name in field_mappings.items():
        if old_name in args_dict:
            args_dict[new_name] = args_dict.pop(old_name)

    dataclass_fields = {f.name for f in MathEvalArgs.__dataclass_fields__.values() if f.init}
    filtered_dict = {k: v for k, v in args_dict.items() if k in dataclass_fields}
    training_args = MathEvalArgs(**filtered_dict)
    return training_args

def math_eval_dataclass_to_args(training_args: MathEvalArgs) -> argparse.Namespace:
    return argparse.Namespace(**training_args.__dict__)

# ===== Arguments for Human Evaluation ==== #
@dataclass
class HumanEvalArgs(BaseFTArgs):
    """
    Dataclass for Human Evaluation arguments
    """
    seq_len: int = 1024
    vocab_size: Optional[int] = None
    pretrained_tokenizer: str="code"
    temperature: float = 0.0
    top_p: float = 1.0
    passes: int = 10
    start_idx: int = 0
    end_idx: Optional[int] = None
    max_len: int = 100
    eval_passes: List[int] = field(default_factory=lambda: [1, 10])

def create_human_eval_parser() -> argparse.ArgumentParser:
    parser = create_base_ft_parser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--pretrained_tokenizer', type=str, default="code")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--passes', type=int, default=10)
    parser.add_argument('--eval_passes', nargs='+', type=int, default=[1, 10])
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--max_len', type=int, default=100)
    return parser

def human_eval_args_to_dataclass(args: HumanEvalArgs):
    args_dict = vars(args).copy()

    # Map CLI argument names to standardized dataclass field names
    field_mappings = {
        'n_layers': 'n_layer',
        'n_heads': 'n_head',
        'n_embed': 'n_embd',
        'save_path': 'save_dir',
        'load_path': 'load_dir'
    }

    for old_name, new_name in field_mappings.items():
        if old_name in args_dict:
            args_dict[new_name] = args_dict.pop(old_name)

    dataclass_fields = {f.name for f in HumanEvalArgs.__dataclass_fields__.values() if f.init}
    filtered_dict = {k: v for k, v in args_dict.items() if k in dataclass_fields}
    training_args = HumanEvalArgs(**filtered_dict)
    return training_args

def human_eval_dataclass_to_args(training_args: HumanEvalArgs) -> argparse.Namespace:
    return argparse.Namespace(**training_args.__dict__)

# ===== Arguments for BigBench Evaluation ==== #
@dataclass
class BigBenchEvalArgs(BaseFTArgs):
    """
    Dataclass for BigBench Evaluation arguments
    """
    data_dir: str = ''
    seq_len: int = 1024
    vocab_size: Optional[int] = None
    pretrained_tokenizer: str="owt"
    temperature: float = 0.0
    top_p: float = 1.0
    passes: int = 1
    max_len: int = 200
    start_idx: int = 0
    end_idx: Optional[int] = None
    n_shot: List[int] = field(default_factory=lambda: [0, 0])
    eval_passes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    max_per_task: int = 30
    eval_mode: str = 'generative'
    few_shot_prompts_path: Optional[str] = None
    min_samples: int = 100

def create_bigbench_eval_parser() -> argparse.ArgumentParser:
    parser = create_base_ft_parser()
    parser.add_argument('--n_shot', nargs=2, type=int, default=[0, 0])
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--pretrained_tokenizer', type=str, default="owt")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--passes', type=int, default=1)
    parser.add_argument('--eval_passes', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--max_per_task', type=int, default=30)
    parser.add_argument('--eval_mode', type=str, default='generative', choices=['generative', 'logprob'])
    parser.add_argument('--few_shot_prompts_path', type=str, default=None)
    parser.add_argument('--min_samples', type=int, default=100)
    return parser

def bigbench_eval_args_to_dataclass(args: BigBenchEvalArgs):
    args_dict = vars(args).copy()

    # Map CLI argument names to standardized dataclass field names
    field_mappings = {
        'n_layers': 'n_layer',
        'n_heads': 'n_head',
        'n_embed': 'n_embd',
        'save_path': 'save_dir',
        'load_path': 'load_dir'
    }

    for old_name, new_name in field_mappings.items():
        if old_name in args_dict:
            args_dict[new_name] = args_dict.pop(old_name)

    dataclass_fields = {f.name for f in BigBenchEvalArgs.__dataclass_fields__.values() if f.init}
    filtered_dict = {k: v for k, v in args_dict.items() if k in dataclass_fields}
    training_args = BigBenchEvalArgs(**filtered_dict)
    return training_args

def bigbench_eval_dataclass_to_args(training_args: BigBenchEvalArgs) -> argparse.Namespace:
    return argparse.Namespace(**training_args.__dict__)

# ===== Arguments for Physics Fine-Tuning ==== #

@dataclass
class PhysicsTrainingArgs(BaseFTArgs):
    """
    Dataclass for physics downstream transfer experiments (e.g., active matter).
    """

    data_root: str = "hf://datasets/polymathic-ai/"
    dataset_name: str = "active_matter_cloud_optimized"
    train_split: str = "train"
    val_split: str = "validation"
    seq_len: int = 4
    patch_size: int = 16
    channels: int = 3
    steps_per_epoch: Optional[int] = None
    grad_accumulation_steps: int = 1
    val_freq: int = 1
    normalize: bool = True
    drop_last: bool = False
    embedding_dropout: float = 0.0
    prefetch_factor: int = 2
    num_traj_limit: Optional[int] = None
    steps_per_trajectory: Optional[int] = None

    def __post_init__(self):
        # Call parent __post_init__
        super().__post_init__()

        # Physics-specific validation
        if not self.save_dir:
            raise ValueError("save_dir is required")

    # Inherits set_runtime_paths() and to_device() from BaseFTArgs/ModelTrainingArgs


def create_physics_ft_parser() -> argparse.ArgumentParser:
    parser = create_base_ft_parser()
    parser.add_argument('--data_root', type=str, default="hf://datasets/polymathic-ai/", help='root location for HuggingFace datasets')
    parser.add_argument('--dataset_name', type=str, default="active_matter_cloud_optimized", help='dataset identifier within the data root')
    parser.add_argument('--train_split', type=str, default='train', help='split name used for training')
    parser.add_argument('--val_split', type=str, default='validation', help='split name used for validation')
    parser.add_argument('--seq_len', type=int, default=4, help='temporal sequence length fed to the transformer')
    parser.add_argument('--patch_size', type=int, default=16, help='square patch size used for ViT-style embedding')
    parser.add_argument('--channels', type=int, default=3, help='number of channels in the active matter frames')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='optional cap on optimization steps per epoch')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='number of micro-steps before optimizer update')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='normalize pixel values to [0, 1]')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', help='disable pixel normalization')
    parser.set_defaults(normalize=True)
    parser.add_argument('--drop_last', action='store_true', default=False, help='drop last partial temporal chunk')
    parser.add_argument('--embedding_dropout', type=float, default=0.0, help='dropout applied after patch embedding')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='prefetch factor for the underlying datamodule')
    parser.add_argument('--num_traj_limit', type=int, default=None, help='optional limit on number of trajectories loaded per epoch')
    parser.add_argument('--steps_per_trajectory', type=int, default=None, help='override for number of temporal steps per trajectory')
    return parser


def physics_ft_args_to_dataclass(args: argparse.Namespace) -> PhysicsTrainingArgs:
    args_dict = vars(args).copy()

    # Map CLI argument names to standardized dataclass field names
    field_mappings = {
        'n_layers': 'n_layer',
        'n_heads': 'n_head',
        'n_embed': 'n_embd',
        'save_path': 'save_dir',
        'load_path': 'load_dir'
    }

    for old_name, new_name in field_mappings.items():
        if old_name in args_dict:
            args_dict[new_name] = args_dict.pop(old_name)

    dataclass_fields = {f.name for f in PhysicsTrainingArgs.__dataclass_fields__.values() if f.init}
    filtered_dict = {k: v for k, v in args_dict.items() if k in dataclass_fields}
    return PhysicsTrainingArgs(**filtered_dict)


def physics_ft_dataclass_to_args(training_args: PhysicsTrainingArgs) -> argparse.Namespace:
    return argparse.Namespace(**training_args.__dict__)
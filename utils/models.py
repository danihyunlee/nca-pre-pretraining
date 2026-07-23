from dataclasses import dataclass

import torch
import torch.nn as nn

from typing import Optional

from transformers import LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaRMSNorm, LlamaAttention, LlamaMLP
from transformers.cache_utils import DynamicCache

from utils.util import setup_logger
logger = setup_logger('models')

def create_attention_mask(seq_length, additive=False):
    """
    If not additive, returns a lower triangular mask of True's.

    If additive, returns a float mask of upper triangular of -inf's and lower triangular of 0's.
    Args:
        seq_length: int
    """
    mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0).unsqueeze(0)
    
    if(additive):
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    
    return mask.bool()

def convert_to_additive_mask(mask):
    """
    Convert a mask to an additive mask.
    Args:
        mask: (batch_size, seq_length, seq_length) mask tensor
    Returns:
        additive_mask: (batch_size, seq_length, seq_length) additive mask tensor

    """
    return mask.float().masked_fill(mask == 0, 0.0).masked_fill(mask == 1, float('-inf'))

#############################################################
#  LLama Style Model
#############################################################

@dataclass
class CustomLlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    use_cache: bool
    rope_theta: int
    attention_dropout: float
    hidden_dropout: float
    output_vocab: int

def create_llama_model(vocab_size: int, seq_length: int, n_layer: int, n_head: int, n_embd: int, attn: str = 'sdpa', output_vocab:int = None):
    config = CustomLlamaConfig(vocab_size=vocab_size,
                               hidden_size=n_embd,
                               intermediate_size=4*n_embd,
                               num_hidden_layers=n_layer,
                               num_attention_heads=n_head,
                               max_position_embeddings=seq_length,
                               use_cache=False,
                               rope_theta=1e4,
                               attention_dropout=0.1,
                               hidden_dropout=0.1,
                               output_vocab=output_vocab if output_vocab else vocab_size)
    
    model = CustomLlamaModel(config=config, attn=attn)
    return model

class CustomLlamaModel(nn.Module):
    def __init__(self, config, attn='sdpa', weight_tying=False):
        super().__init__()
        self.config = LlamaConfig(vocab_size=config.vocab_size,
                                  hidden_size=config.hidden_size,
                                  intermediate_size=config.intermediate_size,
                                  num_hidden_layers=config.num_hidden_layers,
                                  num_attention_heads=config.num_attention_heads,
                                  max_position_embeddings=config.max_position_embeddings,
                                  use_cache=config.use_cache,
                                  rope_theta=config.rope_theta,
                                  attention_dropout=config.attention_dropout,
                                  hidden_dropout=config.hidden_dropout,
                                  attn_implementation=attn,
                                  output_vocab=config.output_vocab)
        
        _llama = LlamaModel(self.config)

        self.vocab_size = config.vocab_size
        self.output_vocab = config.output_vocab
        self.n_embd = config.hidden_size
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.seq_length = config.max_position_embeddings
        self.attn = attn

        self.input_proj = _llama.embed_tokens
        self.wpe = _llama.rotary_emb # RoPE embeddings
        self.layers = _llama.layers
        self.norm = _llama.norm
        self.output_proj = nn.Linear(self.n_embd, self.output_vocab)

        if weight_tying:
            self.input_proj.weight = self.output_proj.weight # weight-tying

    def forward(self, input_ids, 
                attention_mask=None, 
                output_attentions=False):
        
        device = input_ids.device
        input_embeds = self.input_proj(input_ids)

        position_ids = torch.arange(input_embeds.shape[1], device=device).unsqueeze(0)
        position_embeds = self.wpe(input_embeds, position_ids)

        hidden_states = input_embeds

        attentions = []

        for layer in self.layers[:self.n_layer]:
            out = layer(hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeds,
                        output_attentions=output_attentions)
            hidden_states = out[0]

            if output_attentions:
                attentions.append(out[1])

        hidden_states = self.norm(hidden_states)
        logits = self.output_proj(hidden_states)

        if output_attentions:
            return logits, attentions
        return logits

    def freeze(self):
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.wpe.parameters():
            param.requires_grad = False
        for param in self.layers.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.input_proj.parameters():
            param.requires_grad = True
        for param in self.wpe.parameters():
            param.requires_grad = True
        for param in self.layers.parameters():
            param.requires_grad = True

## ---- Downstream Llama Models -----

class BaseDownstreamLlamaModel(nn.Module):
    def __init__(self, model: CustomLlamaModel, frozen_modules=None, reinit_modules=None):
        super().__init__()
        self.layers = model.layers
        self.norm = model.norm
        self.wpe = model.wpe

        self.seq_len = model.seq_length
        self.n_embd = model.n_embd
        self.n_layer = model.n_layer
        
        # Store as sets for efficient lookup
        self.frozen_modules = set(frozen_modules) if frozen_modules else set()
        self.reinit_modules = set(reinit_modules) if reinit_modules else set()

        if 'pos' in self.reinit_modules:
            self.wpe = LlamaRotaryEmbedding(model.config)

    def _freeze_core(self):
        for param in self.layers.parameters():
            param.requires_grad = False
        
        print('Freezing intermediate Llama layers')

    def _unfreeze_core(self):
        for param in self.layers.parameters():
            param.requires_grad = True
        
        print('Unfreezing intermediate Llama layers')
    
    def _unfreeze_attn(self):
        # look for LlamaAttention in the modules and unfreeze
        for name, module in self.named_modules():
            if isinstance(module, LlamaAttention):
                for param in module.parameters():
                    param.requires_grad = True
        
        print('Unfreezing attention in intermediate Llama layers')
    
    def _unfreeze_mlp(self):
        # look for LlamaMLP in the modules and unfreeze
        for name, module in self.named_modules():
            if isinstance(module, LlamaMLP):
                for param in module.parameters():
                    param.requires_grad = True
        
        print('Unfreezing MLP in intermediate Llama layers')
    
    def _freeze_pos(self):
        for param in self.wpe.parameters():
            param.requires_grad = False

        print('Freezing RoPE embeddings -- should have no effect')
    
    def _unfreeze_pos(self):
        for param in self.wpe.parameters():
            param.requires_grad = True
        
        print('Unfreezing RoPE embeddings -- should have no effect')
    
    def _freeze_ln(self):
        """
        Freeze all parameters of all LlamaRMSNorm modules in the model
        """
        frozen_count = 0
        for name, module in self.named_modules():
            # Check for LlamaRMSNorm specifically using isinstance
            if isinstance(module, LlamaRMSNorm):
                for param in module.parameters():
                    param.requires_grad = False
                frozen_count += 1

        print(f'Freezing {frozen_count} LlamaRMSNorm modules')

    def _unfreeze_ln(self):
        """
        Unfreeze all parameters of all LlamaRMSNorm modules in the model
        """
        unfrozen_count = 0
        for name, module in self.named_modules():
            # Check for LlamaRMSNorm specifically using isinstance
            if isinstance(module, LlamaRMSNorm):
                for param in module.parameters():
                    param.requires_grad = True
                unfrozen_count += 1

        print(f'Unfreezing {unfrozen_count} LlamaRMSNorm modules')

    def enable_lora(self):
        for name, param in self.layers.named_parameters():
            if(not(name.startswith('lora'))):
                param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d) or module.__class__.__name__ == 'Conv1D':
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, ImageGPTLayerNorm) or isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    
    def reinit_attention_weights(self, layer_idx=[0, 0]):
        print(f'Reinitializing attention weights in layers {layer_idx[0]} to {layer_idx[1]}')
        for i in range(layer_idx[0], layer_idx[1]):
            block = self.layers[i]
            # Reinitialize attention query, key, value projections
            self._init_weights(block.self_attn.q_proj)
            self._init_weights(block.self_attn.k_proj)
            self._init_weights(block.self_attn.v_proj)
            # Reinitialize attention output projection
            self._init_weights(block.self_attn.o_proj)
            
            # set requires_grad to True
            for param in block.self_attn.parameters():
                param.requires_grad = True
    
    def reinit_mlp_weights(self, layer_idx=[0, 0]):
        print(f'Reinitializing MLP weights in layers {layer_idx[0]} to {layer_idx[1]}')
        for i in range(layer_idx[0], layer_idx[1]):
            block = self.layers[i]
            # Reinitialize MLP first linear layer (c_fc)
            self._init_weights(block.mlp.gate_proj)
            self._init_weights(block.mlp.up_proj)
            # Reinitialize MLP second linear layer (c_proj)
            self._init_weights(block.mlp.down_proj)        

            # set requires_grad to True
            for param in block.mlp.parameters():
                param.requires_grad = True

    def reinit_layer_norm_weights(self, layer_idx=[0, 0]):
        print(f'Reinitializing layer norm weights in layers {layer_idx[0]} to {layer_idx[1]}')
        for i in range(layer_idx[0], layer_idx[1]):
            block = self.layers[i]
            self._init_weights(block.input_layernorm)
            self._init_weights(block.post_attention_layernorm)
            
            # set requires_grad to True
            for param in block.input_layernorm.parameters():
                param.requires_grad = True
            for param in block.post_attention_layernorm.parameters():
                param.requires_grad = True

class DownstreamLlamaModel(BaseDownstreamLlamaModel):
    def __init__(self, model: CustomLlamaModel,
                       input_dim=9,
                       output_dim=1,
                       num_classes=8,
                       frozen_modules=None,
                       reinit_modules=None,
                       input_bias=0.0):
        super().__init__(model, frozen_modules=frozen_modules, reinit_modules=reinit_modules)
        self.num_classes = num_classes
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, model.n_embd, bias=True)
        self.output_proj = nn.Linear(model.n_embd, self.num_classes*self.output_dim)

        self._freeze_unfreeze_modules()

        self.set_input_bias(input_bias)

    def _freeze_unfreeze_modules(self):
        """
        Apply freezing/unfreezing based on self.frozen_modules
        """
        # Handle core (GPT layers)
        if 'core' in self.frozen_modules:
            self._freeze_core()
        else:
            self._unfreeze_core()
        
        if 'core-attn' in self.frozen_modules:
            self._freeze_core()
            self._unfreeze_attn()
        
        if 'core-mlp' in self.frozen_modules:
            self._freeze_core()
            self._unfreeze_mlp()
        
        if 'core-ln' in self.frozen_modules:
            self._freeze_core()
            self._unfreeze_ln()
        
        if 'core-attn-ln' in self.frozen_modules:
            self._freeze_core()
            self._unfreeze_attn()
            self._unfreeze_ln()
        
        # Handle embeddings
        if 'embs' in self.frozen_modules:
            self._freeze_embs()
        else:
            self._unfreeze_embs()

        # Handle layer norms
        if 'ln' in self.frozen_modules:
            self._freeze_ln()
        else:
            self._unfreeze_ln()
        
        # Handle position embeddings
        if 'pos' in self.frozen_modules:
            self._freeze_pos()
        else:
            self._unfreeze_pos()

    def _freeze_embs(self):
        """
        Freeze all parameters of the input and output projections
        """
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.output_proj.parameters():
            param.requires_grad = False
        print('Freezing input and output projections')

    def _unfreeze_embs(self):
        """
        Unfreeze all parameters of the input and output projections
        """
        for param in self.input_proj.parameters():
            param.requires_grad = True
        for param in self.output_proj.parameters():
            param.requires_grad = True
        print('Unfreezing input and output projections')

    def set_input_bias(self, val=0.0):
        self.input_proj.bias.data.fill_(val)

    def forward(self, x, attention_mask=None, output_attentions=False):
        B, L, P = x.shape
        device = x.device
        input_embeds = self.input_proj(x)
        position_ids = torch.arange(input_embeds.shape[1], device=device).unsqueeze(0)
        position_embeds = self.wpe(input_embeds, position_ids)

        hidden_states = input_embeds

        attentions = []

        for layer in self.layers[:self.n_layer]:
            out = layer(hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeds,
                        output_attentions=output_attentions)
            hidden_states = out[0]

            if output_attentions:
                attentions.append(out[1])

        hidden_states = self.norm(hidden_states)
        logits = self.output_proj(hidden_states)

        logits = logits.view(B, L, self.output_dim, self.num_classes)

        if output_attentions:
            return logits, attentions
        return logits
    
class DownstreamLlamaLM(BaseDownstreamLlamaModel):
    def __init__(self, model: CustomLlamaModel,
                 vocab_size=40,
                 output_vocab=None,
                 frozen_modules=None,
                 reinit_modules=None,
                 weight_tying=False,
                 fixed_embed_init=False):
        
        super().__init__(model, frozen_modules=frozen_modules, reinit_modules=reinit_modules)

        self.vocab_size = vocab_size
        self.weight_tying = weight_tying
        self.n_embd = model.n_embd
        self.fixed_embed_init = fixed_embed_init

        if 'embed' in reinit_modules:
            print(f'Reinitializing input and output projections (init)')
            self.input_proj = nn.Embedding(vocab_size, model.n_embd)
            self.output_proj = nn.Linear(model.n_embd, vocab_size, bias=False)
            if fixed_embed_init:
                # match the transformer body's init (normal, std 0.02) instead of
                # the nn.Embedding/nn.Linear constructor defaults (N(0,1) / kaiming)
                print(f'Applying fixed 0.02 init to input and output projections (init)')
                nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        else:
            assert model.vocab_size == vocab_size, f"Model vocabulary size {model.vocab_size} does not match provided vocabulary size {vocab_size}"
            print(f'Using model input and output projections (init)')
            self.input_proj = model.input_proj
            self.output_proj = model.output_proj

        if self.weight_tying:
            self.input_proj.weight = self.output_proj.weight # weight-tying
    
        # Apply freezing/unfreezing based on frozen_modules
        self._freeze_unfreeze_modules()

    def reinit_embeddings(self):
        print(f'Reinitializing input and output projections')
        self.input_proj = nn.Embedding(self.vocab_size, self.n_embd)
        self.output_proj = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        if getattr(self, 'fixed_embed_init', False):
            print(f'Applying fixed 0.02 init to input and output projections')
            nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        self._freeze_unfreeze_modules()
    
    def _freeze_unfreeze_modules(self):
        """
        Apply freezing/unfreezing based on self.frozen_modules
        """
        # Handle core (GPT layers)
        if 'core' in self.frozen_modules:
            self._freeze_core()
        else:
            self._unfreeze_core()
        
        if 'core-attn' in self.frozen_modules:
            self._freeze_core()
            self._unfreeze_attn()
        
        if 'core-mlp' in self.frozen_modules:
            self._freeze_core()
            self._unfreeze_mlp()
        
        if 'core-ln' in self.frozen_modules:
            self._freeze_core()
            self._unfreeze_ln()
        
        if 'core-attn-ln' in self.frozen_modules:
            self._freeze_core()
            self._unfreeze_attn()
            self._unfreeze_ln()
        
        # Handle embeddings
        if 'embs' in self.frozen_modules:
            self._freeze_embs()
        else:
            self._unfreeze_embs()

        # Handle layer norms
        if 'ln' in self.frozen_modules:
            self._freeze_ln()
        else:
            self._unfreeze_ln()
        
        # Handle position embeddings
        if 'pos' in self.frozen_modules:
            self._freeze_pos()
        else:
            self._unfreeze_pos()

    def _freeze_embs(self):
        """
        Freeze all parameters of the input and output projections
        """
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.output_proj.parameters():
            param.requires_grad = False
        print('Freezing input and output projections')

    def _unfreeze_embs(self):
        """
        Unfreeze all parameters of the input and output projections
        """
        for param in self.input_proj.parameters():
            param.requires_grad = True
        for param in self.output_proj.parameters():
            param.requires_grad = True
        print('Unfreezing input and output projections')

    
    def forward(self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        past_key_values = None,
        use_cache: Optional[bool] = False
     ):
        device = input_ids.device
        input_embeds = self.input_proj(input_ids)

        hidden_states = input_embeds

        cache = past_key_values
        if cache is None and use_cache:
            cache = DynamicCache()

        # different versions expose different helpers; this is the most robust pattern
        if use_cache:
            if hasattr(cache, "get_seq_length"):
                past_len = cache.get_seq_length()
            elif hasattr(cache, "seq_length"):
                past_len = cache.seq_length
            else:
                # fallback: try layer 0 if it supports indexing (older hybrids)
                try:
                    past_len = cache[0][0].shape[-2]
                except Exception:
                    past_len = 0
        else:
            past_len = 0

        position_ids = torch.arange(past_len, past_len + hidden_states.shape[1], device=input_ids.device).unsqueeze(0)

        # RoPE tuple (cos, sin)
        position_embeds = self.wpe(hidden_states, position_ids)
        attentions = []

        for i, layer in enumerate(self.layers[:self.n_layer]):
            out = layer(
                hidden_states = hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                position_embeddings = position_embeds,
                past_key_value = cache,
                use_cache = use_cache,
                output_attentions = output_attentions
            )
            hidden_states = out[0]

            if output_attentions:
                attentions.append(out[1])

        hidden_states = self.norm(hidden_states)
        logits = self.output_proj(hidden_states)

        if use_cache:
            if output_attentions:
                return logits, cache, attentions
            else:
                return logits, cache
        else:
            if output_attentions:
                return logits, attentions
            else:
                return logits
    
#############################################################
#  Utility Functions
#############################################################

def get_grad_norm(model):
    return torch.norm(torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]))

def get_grad_norm_attention(model):
    attention_grad_norms = []
    for i, layer in enumerate(model.gpt):
        attention_grad_norm = torch.norm(torch.stack([
            p.grad.norm(2) for p in layer.parameters() if p.grad is not None
        ]))
        attention_grad_norms.append(attention_grad_norm)

    return attention_grad_norms

#############################################################
#  API Compatibility Aliases (Phase 3)
#############################################################

# Alias for backward compatibility with entrypoints
# DownstreamLanguageModel is used by src/openwebtext_ft.py
# It is functionally equivalent to DownstreamLlamaLM
DownstreamLanguageModel = DownstreamLlamaLM
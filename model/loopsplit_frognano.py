import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoreLoopLoRALayer(nn.Module):
    """
    State-based Relaxed Recursion LoRA wrapper for core layers.
    Controls execution via `active_loop`:
      - active_loop == 1: Base weights only (Loop 1 base teacher).
      - active_loop == 2: Base + LoRA_2.
      - active_loop == 3: Base + LoRA_3.
    """
    def __init__(self, base_layer, rank=64, alpha=128):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.active_loop = 1  # Controlled globally by LoopSplitFrogNano
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA for Loop 2
        self.lora_A_2 = nn.Parameter(torch.empty(rank, in_features, dtype=base_layer.weight.dtype))
        self.lora_B_2 = nn.Parameter(torch.zeros(out_features, rank, dtype=base_layer.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A_2, a=math.sqrt(5))
        
        # LoRA for Loop 3
        self.lora_A_3 = nn.Parameter(torch.empty(rank, in_features, dtype=base_layer.weight.dtype))
        self.lora_B_3 = nn.Parameter(torch.zeros(out_features, rank, dtype=base_layer.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A_3, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base_layer(x)
        if self.active_loop == 1:
            return base_out
        elif self.active_loop == 2:
            delta = F.linear(F.linear(x, self.lora_A_2), self.lora_B_2) * self.scaling
            return base_out + delta
        elif self.active_loop == 3:
            delta = F.linear(F.linear(x, self.lora_A_3), self.lora_B_3) * self.scaling
            return base_out + delta
        return base_out


def _text_backbone(hf_model):
    """Resolve the decoder stack (.layers/.embed_tokens/.norm/.rotary_emb) across HF wrappers:
    Qwen3_5ForCausalLM.model, or Qwen3_5ForConditionalGeneration.model.language_model."""
    m = getattr(hf_model, "model", hf_model)
    if not hasattr(m, "layers") and hasattr(m, "language_model"):
        m = m.language_model
    assert hasattr(m, "layers"), "could not locate decoder layers on hf_model"
    return m


class LoopSplitFrogNano(nn.Module):
    """
    LoopSplit FrogNano with State-Routed 3-Pass Recursive Core:
      - Prelude: Layers 0..7 (unlooped, frozen base)
      - Core: Layers 8..23 (looped N=3 times, reasoning engine)
      - Coda: Layers 24..31 (unlooped, frozen base)

    Mathematical Guarantees:
      - Loop 1: Runs 100% pure frozen base weights (exact base teacher).
      - Loops 2 & 3: Activated via `set_active_loop(k)` and modulated by bounded tanh gates.
      - At Step 0: g2 = g3 = 0.0 -> exact bit-parity with Base FrogNano.

    adapter_param_names(): names of all non-base (trainable-able) parameters, for checkpointing.
    """
    def __init__(self, hf_model, core_start=8, core_end=24, lora_rank=64, gate_mode="tanh", boundary_adapter=False,
                 entry_mode="none", entry_init=0.99):
        """
        gate_mode:
          "tanh" : g = tanh(a)                    (Phase A-C, ablation treatment)
          "raw"  : g = a, zero-init, unbounded    (salvage arm; loops 2/3 are exact identity at step 0)
        boundary_adapter: at the entry of loops 2/3 apply
            h <- h + g_k * ( W . RMSNorm(h) + e_k ),  W = I, bias = 0, e_k = 0, g_k = 0 at init
          so layer-23 states can be re-mapped into layer-8's input domain. e_0 is implicit and fixed at 0:
          loop 1 stays the exact base model.
        entry_mode "reinject" (arXiv:2610.00673 recipe): for loops t>1 the core input is
            z_t = w_t * p + (1 - w_t) * h_{t-1},   w_t = sigmoid(a_t),  sigmoid(a_t) = entry_init at start
          where p is the prelude output (the exact loop-1 input). Per-layer output gates are disabled
          (plain layer calls on every loop) and loops 2/3 share LoRA_2. Loop 1 remains the exact base model.
        """
        super().__init__()
        self.backbone = _text_backbone(hf_model)
        assert entry_mode in ("none", "reinject", "concat")
        self.entry_mode = entry_mode
        if entry_mode == "concat":
            # arXiv:2511.07384: u_t = RMSNorm(W [h_{t-1} ; p]).  A dense 2d x d W under Adam drifts by ~lr per entry per
            # step from init regardless of gradient (observed: offdiag rms 0.023 after 30 steps -> CE +1.5). So W is
            # parametrized as  W = [alpha*I | beta*I] + B A  with alpha=entry_init, beta=1, A,B low-rank (B zero-init).
            bb = _text_backbone(hf_model)
            d = bb.config.hidden_size
            dt = bb.embed_tokens.weight.dtype
            r = 64
            self.concat_alpha = nn.Parameter(torch.full((1,), float(entry_init), dtype=torch.float32))  # gain on h
            self.concat_beta = nn.Parameter(torch.ones(1, dtype=torch.float32))                          # gain on p
            self.concat_A = nn.Parameter(torch.empty(r, 2 * d, dtype=dt))
            self.concat_B = nn.Parameter(torch.zeros(d, r, dtype=dt))
            nn.init.kaiming_uniform_(self.concat_A, a=math.sqrt(5))
            self.concat_bias = nn.Parameter(torch.zeros(d, dtype=dt))
            self.concat_norm = nn.RMSNorm(d, eps=getattr(bb.config, "rms_norm_eps", 1e-6), dtype=dt)
            self.concat_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
            self._concat_scale_set = False
        if entry_mode == "reinject":
            a0 = math.log(entry_init / (1.0 - entry_init))
            self.entry_mix = nn.Parameter(torch.full((2,), a0, dtype=torch.float32))  # a_2, a_3
        assert gate_mode in ("tanh", "raw")
        self.model = hf_model
        self.core_start = core_start
        self.core_end = core_end
        self.num_layers = len(self.backbone.layers)
        self.lora_layers = []
        self.gate_mode = gate_mode
        self.boundary_adapter = boundary_adapter
        
        # Per-layer gate parameters for loops 2 and 3 (tanh-bounded or raw)
        num_core = core_end - core_start
        self.a2_mixer = nn.Parameter(torch.zeros(num_core, dtype=torch.float32))
        self.a2_mlp = nn.Parameter(torch.zeros(num_core, dtype=torch.float32))
        
        self.a3_mixer = nn.Parameter(torch.zeros(num_core, dtype=torch.float32))
        self.a3_mlp = nn.Parameter(torch.zeros(num_core, dtype=torch.float32))

        if boundary_adapter:
            hidden = self.backbone.config.hidden_size
            dtype = self.backbone.embed_tokens.weight.dtype
            eps = getattr(self.backbone.config, "rms_norm_eps", 1e-6)
            self.loop_norm = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
            self.loop_proj = nn.Linear(hidden, hidden, bias=True, dtype=dtype)
            with torch.no_grad():
                self.loop_proj.weight.copy_(torch.eye(hidden, dtype=dtype))
                self.loop_proj.bias.zero_()
            self.loop_step_emb = nn.Parameter(torch.zeros(2, hidden, dtype=dtype))   # e_1, e_2 (loops 2, 3)
            self.loop_gate = nn.Parameter(torch.zeros(2, dtype=torch.float32))      # g_2, g_3
        
        # Freeze all base model parameters
        for p in self.model.parameters():
            p.requires_grad = False
            
        # Wrap linear projections in core layers with Relaxed Recursion LoRA
        self._wrap_core_lora(lora_rank=lora_rank)

    def adapter_param_names(self):
        return [n for n, _ in self.named_parameters() if not n.startswith("model.") or "lora_" in n]

    def _wrap_core_lora(self, lora_rank):
        for layer_idx in range(self.core_start, self.core_end):
            layer = self.backbone.layers[layer_idx]
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(attn, proj_name):
                        orig_proj = getattr(attn, proj_name)
                        wrapped = CoreLoopLoRALayer(orig_proj, rank=lora_rank)
                        setattr(attn, proj_name, wrapped)
                        self.lora_layers.append(wrapped)
            elif hasattr(layer, "linear_attn"):
                lin_attn = layer.linear_attn
                for proj_name in ["in_proj_qkv", "out_proj"]:
                    if hasattr(lin_attn, proj_name):
                        orig_proj = getattr(lin_attn, proj_name)
                        wrapped = CoreLoopLoRALayer(orig_proj, rank=lora_rank)
                        setattr(lin_attn, proj_name, wrapped)
                        self.lora_layers.append(wrapped)
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(mlp, proj_name):
                        orig_proj = getattr(mlp, proj_name)
                        wrapped = CoreLoopLoRALayer(orig_proj, rank=lora_rank)
                        setattr(mlp, proj_name, wrapped)
                        self.lora_layers.append(wrapped)

    def set_active_loop(self, loop_idx):
        for l in self.lora_layers:
            l.active_loop = loop_idx

    def _run_core_layer(self, layer, hidden_states, position_embeddings, attention_mask, position_ids, loop_idx, core_idx, fixed_gate=None):
        if loop_idx == 1 or self.entry_mode in ("reinject", "concat"):
            return layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        
        # Bounded gates: optionally overridden during Phase A forced warmup
        if fixed_gate is not None:
            g_mixer = fixed_gate
            g_mlp = fixed_gate
        elif loop_idx == 2:
            g_mixer, g_mlp = self.a2_mixer[core_idx], self.a2_mlp[core_idx]
        elif loop_idx == 3:
            g_mixer, g_mlp = self.a3_mixer[core_idx], self.a3_mlp[core_idx]
        if fixed_gate is None and self.gate_mode == "tanh":
            g_mixer, g_mlp = torch.tanh(g_mixer), torch.tanh(g_mlp)

        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        
        if layer.block_type == "linear_attention":
            mixer_out = layer.linear_attn(
                hidden_states=normed,
                attention_mask=attention_mask,
            )
        elif layer.block_type == "full_attention":
            mixer_out, _ = layer.self_attn(
                hidden_states=normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            
        g_mix_val = g_mixer.to(residual.dtype) if isinstance(g_mixer, torch.Tensor) else g_mixer
        hidden_states = residual + (g_mix_val * mixer_out)
        
        residual = hidden_states
        normed = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(normed)
        g_mlp_val = g_mlp.to(residual.dtype) if isinstance(g_mlp, torch.Tensor) else g_mlp
        hidden_states = residual + (g_mlp_val * mlp_out)
        
        return hidden_states

    def forward(self, input_ids, num_loops=3, return_all_readouts=False, fixed_gate=None, single_pass_lora=False):
        """
        single_pass_lora=True (ablation control): run the core exactly once as the plain base layer stack
        (no gates) but with LoRA_2 active on every wrapped projection. Requires num_loops=1.
        """
        assert not (single_pass_lora and num_loops != 1), "single_pass_lora requires num_loops=1"
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)
        text_position_ids = position_ids[0]
        extra_position_ids = position_ids[1:]
        
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask, create_recurrent_attention_mask
        mask_kwargs = {
            "config": self.backbone.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        
        hidden_states = inputs_embeds
        position_embeddings = self.backbone.rotary_emb(hidden_states, extra_position_ids)
        
        # 1. Prelude (Layers 0..7)
        for i in range(0, self.core_start):
            layer = self.backbone.layers[i]
            mask = causal_mask_mapping[self.backbone.config.layer_types[i]]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=text_position_ids,
            )
        prelude_out = hidden_states
            
        readouts = {}
        
        # 2. Core Loops (Layers 8..23)
        for loop_idx in range(1, num_loops + 1):
            if single_pass_lora or (self.entry_mode in ("reinject", "concat") and loop_idx >= 2):
                self.set_active_loop(2)   # shared LoRA_2 on every non-base pass
            else:
                self.set_active_loop(loop_idx)  # Explicitly route LoRA branches!
            if self.entry_mode == "reinject" and loop_idx >= 2:
                w = torch.sigmoid(self.entry_mix[loop_idx - 2]).to(hidden_states.dtype)
                hidden_states = w * prelude_out + (1 - w) * hidden_states
            elif self.entry_mode == "concat" and loop_idx >= 2:
                if not self._concat_scale_set:
                    with torch.no_grad():
                        self.concat_scale.fill_(prelude_out.float().pow(2).mean().sqrt().item())
                    self._concat_scale_set = True
                cat = torch.cat([hidden_states, prelude_out], dim=-1)
                u = (self.concat_alpha.to(hidden_states.dtype) * hidden_states
                     + self.concat_beta.to(hidden_states.dtype) * prelude_out
                     + F.linear(F.linear(cat, self.concat_A), self.concat_B) + self.concat_bias)
                hidden_states = self.concat_norm(u) * self.concat_scale.to(u.dtype)
            if self.boundary_adapter and loop_idx >= 2:
                k = loop_idx - 2
                mapped = self.loop_proj(self.loop_norm(hidden_states)) + self.loop_step_emb[k]
                hidden_states = hidden_states + self.loop_gate[k].to(hidden_states.dtype) * mapped
            for i in range(self.core_start, self.core_end):
                layer = self.backbone.layers[i]
                core_idx = i - self.core_start
                mask = causal_mask_mapping[self.backbone.config.layer_types[i]]
                hidden_states = self._run_core_layer(
                    layer,
                    hidden_states,
                    position_embeddings,
                    mask,
                    text_position_ids,
                    loop_idx,
                    core_idx,
                    fixed_gate=fixed_gate,
                )
                
            if return_all_readouts:
                coda_h = hidden_states
                for j in range(self.core_end, self.num_layers):
                    coda_mask = causal_mask_mapping[self.backbone.config.layer_types[j]]
                    coda_h = self.backbone.layers[j](
                        hidden_states=coda_h,
                        position_embeddings=position_embeddings,
                        attention_mask=coda_mask,
                        position_ids=text_position_ids,
                    )
                normed_final = self.backbone.norm(coda_h)
                readouts[f"logits_loop_{loop_idx}"] = self.model.lm_head(normed_final)

        # 3. Coda (Layers 24..31)
        for i in range(self.core_end, self.num_layers):
            layer = self.backbone.layers[i]
            mask = causal_mask_mapping[self.backbone.config.layer_types[i]]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=text_position_ids,
            )
            
        final_normed = self.backbone.norm(hidden_states)
        final_logits = self.model.lm_head(final_normed)
        
        # Reset to base pass
        self.set_active_loop(1)
        
        if return_all_readouts:
            return final_logits, readouts
        return final_logits

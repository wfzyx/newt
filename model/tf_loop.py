"""
Training-free loop wrapper over a frozen HF Qwen3.5-family checkpoint (FrogNano / Qwen3.5).

Follows arXiv:2605.23872 (Training-Free Looped Transformers): a contiguous window [a, b] of decoder
layers is re-applied K times at inference, with no weight change. The window operator is
g = L_b o ... o L_a and its residual field F(x) = g(x) - x, so one pass is a forward-Euler step with h=1.

Strategies (x0 = window input):
  naive   : x <- g(x), K times                                  (what our Stage 0 / LoopSplit loop-1->2 did)
  euler   : x <- x + (1/K) F(x), K times                        (damped sub-steps; the paper's main recipe)
  anchor  : beta * g(x0) + (1 - beta) * euler_K(x0)             (paper's RK tableau: interpolate toward base)
  uniform : x_{k+1} = g(mean(x_0..x_k))                         (arXiv:2602.14759 "Uniform" interpolation)
  noise   : g(x0) + (1 - beta) * eps,  |eps| per token = |euler_K(x0) - g(x0)| per token
            (control from arXiv:2602.14759: the loop's deviation from base replaced by matched-norm Gaussian noise)

Modes:
  block   : apply the strategy to the whole window operator g
  layer   : apply the strategy to each layer in the window in turn (L_a^K, then L_{a+1}^K, ...)

K=1 (any strategy/mode) or beta=1 reproduces the base model exactly.
"""
import torch

from model.loopsplit_frognano import _text_backbone


class TFLoop:
    _gens = {}

    @classmethod
    def _gen(cls, device):
        key = str(device)
        if key not in cls._gens:
            cls._gens[key] = torch.Generator(device=device).manual_seed(0)
        return cls._gens[key]

    def __init__(self, hf_model):
        self.hf = hf_model
        self.bb = _text_backbone(hf_model)
        self.num_layers = len(self.bb.layers)
        self.lm_head = hf_model.lm_head

    # ------------------------------------------------------------------ plumbing
    def _prep(self, input_ids):
        bsz, T = input_ids.shape
        x = self.bb.embed_tokens(input_ids)
        pos = torch.arange(T, device=input_ids.device).view(1, 1, -1).expand(4, bsz, -1)
        text_pos, extra_pos = pos[0], pos[1:]
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask, create_recurrent_attention_mask
        kw = {"config": self.bb.config, "inputs_embeds": x, "attention_mask": None,
              "past_key_values": None, "position_ids": text_pos}
        masks = {"full_attention": create_causal_mask(**kw),
                 "linear_attention": create_recurrent_attention_mask(**kw)}
        pe = self.bb.rotary_emb(x, extra_pos)
        return x, text_pos, pe, masks

    def _layer(self, i, x, text_pos, pe, masks):
        L = self.bb.layers[i]
        return L(hidden_states=x, position_embeddings=pe,
                 attention_mask=masks[self.bb.config.layer_types[i]], position_ids=text_pos)

    def _window(self, a, b, x, ctx):
        for i in range(a, b + 1):
            x = self._layer(i, x, *ctx)
        return x

    # ------------------------------------------------------------------ strategies
    @staticmethod
    def _iterate(g, x0, K, strategy, beta):
        if K <= 1:
            return g(x0)
        if strategy == "naive":
            x = x0
            for _ in range(K):
                x = g(x)
            return x
        if strategy in ("euler", "anchor", "noise"):
            x = x0
            gx0 = None
            for k in range(K):
                gx = g(x)
                if k == 0:
                    gx0 = gx
                x = x + (gx - x) / K
            if strategy == "anchor":
                return beta * gx0 + (1.0 - beta) * x
            if strategy == "noise":
                d = (x - gx0).float()
                eps = torch.randn(d.shape, device=d.device, dtype=torch.float32, generator=TFLoop._gen(d.device))
                eps = eps * (d.norm(dim=-1, keepdim=True) / eps.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                return gx0 + (1.0 - beta) * eps.to(gx0.dtype)
            return x
        if strategy == "uniform":
            hist = [x0]
            x = x0
            for _ in range(K):
                x = g(torch.stack(hist, 0).mean(0))
                hist.append(x)
            return x
        raise ValueError(strategy)

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, input_ids, a, b, K=2, strategy="euler", mode="block", beta=0.5):
        assert 0 <= a <= b < self.num_layers
        x, text_pos, pe, masks = self._prep(input_ids)
        ctx = (text_pos, pe, masks)
        for i in range(0, a):
            x = self._layer(i, x, *ctx)
        if mode == "block":
            x = self._iterate(lambda z: self._window(a, b, z, ctx), x, K, strategy, beta)
        elif mode == "layer":
            for i in range(a, b + 1):
                x = self._iterate(lambda z, i=i: self._layer(i, z, *ctx), x, K, strategy, beta)
        else:
            raise ValueError(mode)
        for i in range(b + 1, self.num_layers):
            x = self._layer(i, x, *ctx)
        return self.lm_head(self.bb.norm(x))

    def make_forward(self, **cfg):
        return lambda ids: self.forward(ids, **cfg)

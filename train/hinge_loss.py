import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeTrustRegionLoopSplitLoss(nn.Module):
    """
    Advisor-audited Hinge Trust-Region Multi-Loop Loss:
      - Loop 1 is frozen base forward (exact base teacher).
      - Multi-loop CE: w2 * CE(loop2) + w3 * CE(loop3).
      - Annealed w2: ramps linearly from 0.3 down to 0.1 over the first 2,000 steps.
      - Per-token Hinge Trust-Region KL in true nats:
          kl_tok = sum_v p_base * (log p_base - log p_L3)
          raw_kl = mean(kl_tok) (per-token nats)
          L_anchor = lambda_kl * relu(raw_kl - tau)
          where lambda_kl = 0.05, tau = 0.05 nats.
          Stops penalizing when the model is within 0.05 nats of base.
    """
    def __init__(self, w2_start=0.3, w2_end=0.1, w3=1.0, lambda_kl=0.05, tau=0.05, anneal_steps=2000):
        super().__init__()
        self.w2_start = w2_start
        self.w2_end = w2_end
        self.w3 = w3
        self.lambda_kl = lambda_kl
        self.tau = tau
        self.anneal_steps = anneal_steps

    def get_w2(self, step_offset):
        if step_offset >= self.anneal_steps:
            return self.w2_end
        progress = max(0.0, float(step_offset) / float(self.anneal_steps))
        return self.w2_start - (self.w2_start - self.w2_end) * progress

    def forward(self, readouts, target_ids, step_offset=0, loss_mask=None):
        shift_targets = target_ids[:, 1:].contiguous().view(-1)
        w2 = self.get_w2(step_offset)
        
        if loss_mask is not None:
            mask_flat = loss_mask[:, 1:].contiguous().view(-1).float()
        else:
            mask_flat = None
            
        # 1. Base Readout (Loop 1)
        logits_1 = readouts["logits_loop_1"][:, :-1, :].contiguous()
        
        # 2. Loop 2 CE
        logits_2 = readouts["logits_loop_2"][:, :-1, :].contiguous()
        ce_2 = F.cross_entropy(logits_2.view(-1, logits_2.size(-1)), shift_targets, reduction="none")
        
        # 3. Loop 3 CE
        logits_3 = readouts["logits_loop_3"][:, :-1, :].contiguous()
        ce_3 = F.cross_entropy(logits_3.view(-1, logits_3.size(-1)), shift_targets, reduction="none")
        
        if mask_flat is not None:
            ce_2 = (ce_2 * mask_flat).sum() / (mask_flat.sum() + 1e-6)
            ce_3 = (ce_3 * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        else:
            ce_2 = ce_2.mean()
            ce_3 = ce_3.mean()
            
        # 4. Exact Per-Token KL in nats (FP32 precision for numerical stability)
        log_p_3 = F.log_softmax(logits_3.float(), dim=-1)
        log_p_1 = F.log_softmax(logits_1.detach().float(), dim=-1)
        # kl_tok shape: [B, T-1] -> [B*(T-1)]
        kl_tok = (log_p_1.exp() * (log_p_1 - log_p_3)).sum(dim=-1).view(-1)
        
        if mask_flat is not None:
            raw_kl = (kl_tok * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        else:
            raw_kl = kl_tok.mean()
            
        hinge_kl = F.relu(raw_kl - self.tau)
        kl_penalty = self.lambda_kl * hinge_kl
        
        total_loss = (w2 * ce_2) + (self.w3 * ce_3) + kl_penalty
        
        with torch.no_grad():
            ce_1 = F.cross_entropy(logits_1.view(-1, logits_1.size(-1)), shift_targets).item()
        
        return {
            "loss": total_loss,
            "w2": w2,
            "ce_loop1_base": ce_1,
            "ce_loop2": ce_2.item(),
            "ce_loop3": ce_3.item(),
            "raw_kl": raw_kl.item(),
            "hinge_kl": hinge_kl.item()
        }

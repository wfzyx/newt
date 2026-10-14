import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularizedLoopSplitLoss(nn.Module):
    """
    Overhauled Regularized Multi-Loop Loss:
      - Loop 1 is frozen base forward (exact base teacher).
      - Multi-loop CE: w2 * CE(loop2) + w3 * CE(loop3).
      - Base KL Anchor: beta_kl * KL(p_base || p_loop3) to prevent loop drift/repetition.
      - Removed base-hard token weighting and monotone hinge loss.
    """
    def __init__(self, w2=0.3, w3=1.0, beta_kl=0.05):
        super().__init__()
        self.w2 = w2
        self.w3 = w3
        self.beta_kl = beta_kl

    def forward(self, readouts, target_ids, loss_mask=None):
        shift_targets = target_ids[:, 1:].contiguous().view(-1)
        
        # 1. Base Readout (Loop 1) - used as KL anchor
        logits_1 = readouts["logits_loop_1"][:, :-1, :].contiguous()
        p_base = F.softmax(logits_1.detach() / 1.0, dim=-1)
        
        # 2. Loop 2
        logits_2 = readouts["logits_loop_2"][:, :-1, :].contiguous()
        ce_2 = F.cross_entropy(logits_2.view(-1, logits_2.size(-1)), shift_targets, reduction="none")
        
        # 3. Loop 3
        logits_3 = readouts["logits_loop_3"][:, :-1, :].contiguous()
        ce_3 = F.cross_entropy(logits_3.view(-1, logits_3.size(-1)), shift_targets, reduction="none")
        
        if loss_mask is not None:
            mask_flat = loss_mask[:, 1:].contiguous().view(-1).float()
            ce_2 = (ce_2 * mask_flat).sum() / (mask_flat.sum() + 1e-6)
            ce_3 = (ce_3 * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        else:
            ce_2 = ce_2.mean()
            ce_3 = ce_3.mean()
            
        # Base KL Anchor: KL(p_base || p_loop3)
        log_p_3 = F.log_softmax(logits_3, dim=-1)
        kl_div = F.kl_div(log_p_3, p_base, reduction="batchmean")
        
        total_loss = (self.w2 * ce_2) + (self.w3 * ce_3) + (self.beta_kl * kl_div)
        
        # Monitor unweighted loop 1 CE
        ce_1 = F.cross_entropy(logits_1.view(-1, logits_1.size(-1)), shift_targets).item()
        
        return {
            "loss": total_loss,
            "ce_loop1_base": ce_1,
            "ce_loop2": ce_2.item(),
            "ce_loop3": ce_3.item(),
            "kl_anchor": kl_div.item()
        }

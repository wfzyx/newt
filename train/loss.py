import torch
import torch.nn as nn
import torch.nn.functional as F

class LoopSplitLoss(nn.Module):
    """
    Loss function for LoopSplit Recursive Transformer:
      1. Base-Hard Token Weighting:
         u_t = clip(CE_base(t) / mean_batch(CE_base), 0.5, 4.0)
         Forces Loops 2 & 3 to concentrate gradient mass on tokens where 1-pass fails.
      2. Multi-Loop Supervised Cross-Entropy:
         L_ce = w1*CE(loop1) + w2*CE(loop2) + w3*CE(loop3)
      3. Monotone Hinge Loss:
         L_mono = relu(CE_3 - CE_2) + relu(CE_2 - CE_1)
         Penalizes Loop 3 whenever it is worse than Loop 2, or Loop 2 worse than Loop 1.
    """
    def __init__(self, w1=0.1, w2=0.3, w3=1.0, lambda_mono=1.0):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.lambda_mono = lambda_mono

    def forward(self, readouts, target_ids):
        """
        readouts: dict containing 'logits_loop_1', 'logits_loop_2', 'logits_loop_3'
        target_ids: tensor of shape (batch, seq_len)
        """
        shift_targets = target_ids[:, 1:].contiguous().view(-1)
        
        # 1. Base loss (Loop 1) per token
        logits_1 = readouts["logits_loop_1"][:, :-1, :].contiguous()
        ce_per_token_1 = F.cross_entropy(logits_1.view(-1, logits_1.size(-1)), shift_targets, reduction="none")
        
        # Base-Hard Token Weights
        mean_ce_1 = ce_per_token_1.mean().detach() + 1e-6
        u_t = torch.clamp(ce_per_token_1.detach() / mean_ce_1, min=0.5, max=4.0)
        
        # Weighted loop losses
        loss_loop_1 = (ce_per_token_1 * u_t).mean()
        
        # 2. Loop 2
        logits_2 = readouts["logits_loop_2"][:, :-1, :].contiguous()
        ce_per_token_2 = F.cross_entropy(logits_2.view(-1, logits_2.size(-1)), shift_targets, reduction="none")
        loss_loop_2 = (ce_per_token_2 * u_t).mean()
        
        # 3. Loop 3
        logits_3 = readouts["logits_loop_3"][:, :-1, :].contiguous()
        ce_per_token_3 = F.cross_entropy(logits_3.view(-1, logits_3.size(-1)), shift_targets, reduction="none")
        loss_loop_3 = (ce_per_token_3 * u_t).mean()
        
        # Multi-loop total CE
        total_ce = (self.w1 * loss_loop_1) + (self.w2 * loss_loop_2) + (self.w3 * loss_loop_3)
        
        # Monotone hinge: penalize if Loop 3 > Loop 2, or Loop 2 > Loop 1
        mean_raw_1 = ce_per_token_1.mean()
        mean_raw_2 = ce_per_token_2.mean()
        mean_raw_3 = ce_per_token_3.mean()
        
        loss_mono = F.relu(mean_raw_3 - mean_raw_2) + F.relu(mean_raw_2 - mean_raw_1)
        
        total_loss = total_ce + (self.lambda_mono * loss_mono)
        
        return {
            "loss": total_loss,
            "loss_ce": total_ce.item(),
            "loss_mono": loss_mono.item(),
            "ce_loop1": mean_raw_1.item(),
            "ce_loop2": mean_raw_2.item(),
            "ce_loop3": mean_raw_3.item()
        }

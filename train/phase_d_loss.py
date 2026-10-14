import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseDLoss(nn.Module):
    """
    Phase D per-micro-batch loss:

        L = w2 * CE_L2 + w3 * CE_L3 + lambda_kl[stream] * relu(KL(p_L1 || p_L3) - tau[stream]) + lambda_g * sum(a_i^2)

      - CE_L1 logged under no_grad (pure base baseline; loop-1 readout traverses the coda as the exact base model).
      - KL is forward KL against the detached loop-1 distribution, per-token in nats, masked-mean.
      - tau is per-stream: measured at step 5000 (KL_0) so existing drift is not penalized.
      - Gate L2 is applied once per optimizer step by the trainer (not per micro-batch); this module only exposes it.
    """

    def __init__(self, w2: float = 0.25, w3: float = 0.75, lambda_kl=None, tau=None, lambda_g: float = 2e-3):
        super().__init__()
        self.w2 = w2
        self.w3 = w3
        self.lambda_kl = lambda_kl or {1: 0.5, 2: 0.5, 3: 0.1}
        self.tau = tau or {1: 0.0, 2: 0.0, 3: 0.0}
        self.lambda_g = lambda_g

    def set_tau(self, tau: dict) -> None:
        self.tau = dict(tau)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x * mask).sum() / (mask.sum() + 1e-6)

    def forward(self, readouts: dict, input_ids: torch.Tensor, loss_mask: torch.Tensor, stream: int) -> dict:
        targets = input_ids[:, 1:].contiguous().view(-1)
        mask = loss_mask[:, 1:].contiguous().view(-1).float()

        logits_1 = readouts["logits_loop_1"][:, :-1, :].contiguous()
        logits_2 = readouts["logits_loop_2"][:, :-1, :].contiguous()
        logits_3 = readouts["logits_loop_3"][:, :-1, :].contiguous()
        V = logits_3.size(-1)

        ce_2 = self._masked_mean(F.cross_entropy(logits_2.view(-1, V), targets, reduction="none"), mask)
        ce_3 = self._masked_mean(F.cross_entropy(logits_3.view(-1, V), targets, reduction="none"), mask)

        # Forward KL(p1 || p3) per token, fp32
        log_p1 = F.log_softmax(logits_1.detach().float(), dim=-1)
        log_p3 = F.log_softmax(logits_3.float(), dim=-1)
        kl_tok = (log_p1.exp() * (log_p1 - log_p3)).sum(dim=-1).view(-1)
        raw_kl = self._masked_mean(kl_tok, mask)
        hinge = F.relu(raw_kl - float(self.tau.get(stream, 0.0)))
        kl_pen = float(self.lambda_kl[stream]) * hinge

        loss = self.w2 * ce_2 + self.w3 * ce_3 + kl_pen

        with torch.no_grad():
            ce_1 = self._masked_mean(F.cross_entropy(logits_1.view(-1, V), targets, reduction="none"), mask)

        return {
            "loss": loss,
            "ce_loop1_base": ce_1.item(),
            "ce_loop2": ce_2.item(),
            "ce_loop3": ce_3.item(),
            "raw_kl": raw_kl.item(),
            "hinge_kl": hinge.item(),
            "kl_pen": kl_pen.item() if torch.is_tensor(kl_pen) else float(kl_pen),
        }

    def gate_penalty(self, gate_params) -> torch.Tensor:
        """lambda_g * sum_i a_i^2 over all pre-tanh gate logits (per optimizer step)."""
        return self.lambda_g * sum((g.float() ** 2).sum() for g in gate_params)

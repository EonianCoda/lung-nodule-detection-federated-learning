import torch
import torch.nn.functional as F

def dice_log_loss(axis=(2, 3, 4), smooth=1e-6):
    def loss_function(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        intersection = torch.sum(y_true * y_pred, dim=axis)
        intersection = torch.mean(intersection + smooth, dim=1)
        total = torch.sum(y_pred**2, dim=axis) + torch.sum(y_true**2, dim=axis)
        total = torch.mean(total + smooth, dim=1)
        loss = -torch.log(2.0 * intersection) + torch.log(total)
        loss = torch.mean(loss)
        return loss
    return loss_function


def focal_loss(alpha: float = 0.25 , gamma: float = 2, reduction: str = 'mean'):
    def loss_function(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction = 'none')
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
    return loss_function
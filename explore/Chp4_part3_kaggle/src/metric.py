from dataclasses import dataclass
import torch
import torch.nn as nn

# dataloader -> class 작게작게 
class Metric:
    fold: int 
    train_loss_rmse: float 
    valid_loss_rmse: float 
    
def log_rmse(pred_y, labels):
    clipped_preds = torch.clamp(pred_y, 1, float('inf'))
    rmse = torch.sqrt(nn.MSELoss()(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# torch.clamp => pred_y, min값은 1, max값은 inf로 고정 -> 근데 왜 고정을 해주는거지 max값은? 
# torch.sqrt => Returns a new tensor with the square-root of the elements of input.
# rmse.item() => loss 갖고 있는 스칼라 값 가져오는 역할 

import torch
import torch.nn.functional as F

def KL_divergence(mean:torch.Tensor, logvar: torch.Tensor):
    """
    Docstring for KL_divergence
    
    mean: [Batch, dimention_latent, 1]
    variance: [Batch, dimention_latent, 1]

    """
    return torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar), dim=1))

def L2_loss(output: torch.Tensor, target: torch.Tensor):
    return F.mse_loss(output, target)


if __name__ == "__main__":
    mean = torch.rand(4, 100, 1).to(torch.device('cuda'))
    var = torch.rand(4, 100, 1).to(torch.device('cuda'))

    print(KL_divergence(mean, var)) 
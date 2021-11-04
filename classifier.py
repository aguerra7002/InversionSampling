# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    
    def __init__(self, x_dim, h_dim1, h_dim2, out_dim):
        super(MLPClassifier, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, out_dim)
    
    def forward(self, x, return_latent=False):
        h1 = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope=0.1)
        out = F.log_softmax(self.fc3(h2))
        if return_latent:
            return h1, h2, out
        else:
            return out
        
        
    def invert_leaky_relu(self, vec, slope=0.1):
        new_vec = vec.clone()
        new_vec[vec < 0] /= slope
        return new_vec
    
        
    def loss_function(self, data, target):
        
        data_flat = data.reshape((data.shape[0], 784)) #+ torch.randn((data.shape[0], 784)) / 0.02
        h1, h2, preds = self.forward(data_flat, return_latent=True)
        t0 = F.nll_loss(preds, target)

        dist1 = (self.invert_leaky_relu(h2) - self.fc2.bias) @ torch.linalg.pinv(self.fc2.weight).T
#         mu_vec1 = torch.mean(h1, 0)
#         cov_mat1 = torch.var(h1, 0)
#         norm1 = torch.distributions.Normal(mu_vec1, cov_mat1)
#         t1 = -norm1.log_prob(dist1).mean()
        #t1 = torch.trace((dist1 - h1) @ (dist1 - h1).T) / dist1.shape[0] # MSE
        t1 = F.mse_loss(dist1, h1)

        dist2 = (self.invert_leaky_relu(h1) - self.fc1.bias) @ torch.linalg.pinv(self.fc1.weight).T
#         mu_vec2 = torch.mean(data_flat, 0)
#         cov_mat2 = torch.var(data_flat + torch.randn(data_flat.shape).detach() / 20, 0)
#         norm2 = torch.distributions.Normal(mu_vec2, cov_mat2)
#         t2 = -norm2.log_prob(dist2).mean()
        #t2 = torch.trace((dist2 - data_flat) @ (dist2 - data_flat).T) / dist2.shape[0] # MSE
        t2 = F.mse_loss(dist2, data_flat)
        
        return t0 + t1 + t2
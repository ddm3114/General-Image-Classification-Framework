import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        # Concatenate the two sets of embeddings
        z = torch.cat((z_i, z_j), dim=0)

        # Calculate the cosine similarity matrix
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        # Scale the similarity matrix by the temperature
        sim_matrix = sim_matrix / self.temperature

        # Create labels for the positive pairs
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat((labels, labels), dim=0)
        labels[:batch_size] = labels[:batch_size]+batch_size-1

        # Create a mask to remove self-similarities
        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix[~mask].view(batch_size * 2, -1)
        # Calculate the cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


criterion = ContrastiveLoss(temperature=0.5)


def get_loss(loss_name,**args):
    if loss_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'l1':
        criterion = nn.L1Loss()
    elif loss_name == 'ContrastLoss':
        if 'temperature' in args:
            temperature = args['temperature']
        else:
            temperature = 0.5
        criterion = ContrastiveLoss(temperature)
    else:
        raise NotImplementedError
    return criterion


if __name__ == '__main__':
    batch_size = 4
    embedding_dim = 128
    temperature = 0.5
    z_i = torch.randn(batch_size, embedding_dim, requires_grad=True)
    z_j = z_i

    criterion = get_loss("ContrastLoss")
    loss = criterion(z_i, z_j)
    print('Loss:', loss.item())

    # Backward pass
    # loss.backward()
    # print('Gradient for z_i:', z_i.grad)
    # print('Gradient for z_j:', z_j.grad)
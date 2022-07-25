import torch, torch.nn as nn

class WaveNet(nn.Module):
    def __init__(self, n_in, n_out, branch_dims=[64, 32]):
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        
        shared_layers = [
            nn.BatchNorm1d(self.n_in), 
            nn.Linear(self.n_in, 256),
            nn.ReLU()
        ]
        
        for i in range(4):
            shared_layers.append(nn.Linear(256, 256))
            shared_layers.append(nn.ReLU())
            
        shared_layers.append(nn.Linear(256, branch_dims[0]))
        shared_layers.append(nn.ReLU())
        
        branches = []
        for _ in range(self.n_out):
            args = []
            for a, b in zip(branch_dims[:-1], branch_dims[1:]):
                args.append(nn.Linear(a, b))
                args.append(nn.ReLU())
                
            args.append(nn.Linear(branch_dims[-1], 1))
            branches.append(nn.Sequential(*args))
            
        self.shared = nn.Sequential(*shared_layers)
        self.branches = nn.ModuleList(branches)
        
        self.shared.apply(_xavier_init)
        for branch in self.branches:
            branch.apply(_xavier_init)
            
        self.to(torch.double)
        
    def forward(self, X):
        Z = self.shared(X)
        
        out = torch.zeros((X.shape[0], self.n_out))
        for j, branch in enumerate(self.branches):
            out[:, j] = branch(Z).squeeze()
            
        return out
    
def _xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
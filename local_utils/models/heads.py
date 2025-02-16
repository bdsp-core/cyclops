import torch.nn as nn

# create a regression head for the datset
class RegressionHead(nn.Sequential):
    def __init__(self, emb_size,dropout=0.3):
        super().__init__()
        self.reghead = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ELU(),
            nn.Linear(emb_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.reghead(x)
        return out
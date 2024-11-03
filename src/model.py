import torch as pt


class Model(pt.nn.Module):

    def __init__(self, in_features=1024, out_features=2):

        super().__init__()
        self.linear = pt.nn.Linear(in_features=in_features, out_features=out_features)
        self.softmax = pt.nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.linear(x)
        x = self.softmax(x)
        return x


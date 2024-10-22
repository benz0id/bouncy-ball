import torch


class Model(torch.nn.Module):
    """
    A simple neural network with fully connected layers
    """

    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 input_shape: int,
                 output_shape: int):

        super().__init__()
        layers = []

        # Construct variable number of layers.
        for i in range(num_layers):
            # Specify input and output layer sizes.
            in_dim = hidden_dim
            out_dim = hidden_dim
            if i == 0:
                in_dim = input_shape
            if i == num_layers - 1:
                out_dim = output_shape

            layers.append(torch.nn.Linear(in_dim, out_dim))

            # Add norm layers.
            if i == num_layers - 1:
                pass
            else:
                layers.append(torch.nn.ReLU())

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.layers(X)

"""
This file contains a very simple PyTorch module to use for enhancement.

To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module.

Authors
 * Peter Plantinga 2021
"""
import torch


class CustomModel(torch.nn.Module):
    """Complex spec mapping for BWE.

    Arguments
    ---------
    input_size : int
        Size of the expected input in the 3rd dimension.
    rnn_size : int
        Number of neurons to use in rnn (for each direction -> and <-).
    projection : int
        Number of neurons in projection layer.
    layers : int
        Number of RNN layers to use.
    """

    def __init__(self, input_size=101, output_size=100, rnn_size=256, projection=128, layers=4):
        # input shape:[B, T, input_size, 2]
        # output shape:[B, T, output_size, 2]
        # not mask and just mapping, like CRN
        # narrow band spec ---> wide band spec
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for i in range(layers):
            self.layers.append(
                torch.nn.LSTM(
                    input_size=input_size * 2 if i == 0 else projection,
                    hidden_size=rnn_size,
                    bidirectional=True,
                )
            )

            # Projection layer reduces size, except last layer, which
            # goes back to input size to create the mask
            linear_size = output_size * 2 if i == layers - 1 else projection
            self.layers.append(
                torch.nn.Linear(
                    in_features=rnn_size * 2, out_features=linear_size,
                )
            )

    def forward(self, x):
        x = x.flip(dims=[2])
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        x = x.transpose(0, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1, 2)
        return x

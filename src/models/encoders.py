import torch
from torch import nn
import abc

class Base_Encoder(abc.ABC):
    """
    Class to add methods for common modality specific methods, which is not part of nn Module
    """

    @abc.abstractmethod
    def get_output_size(self):
        pass


#from https://github.com/jameschapman19/cca_zoo
class EncoderMLP(nn.Module,Base_Encoder):
    #extension of Encoder from cca that contains batchnorm layer
    def __init__(
        self,
        latent_dims: int,
        feature_size: int,
        layer_sizes: tuple = None,
        activation=nn.ReLU, #LeakyReLU, GELU or nn.Tanh()
        dropout=0,
        batchnorm: bool=False,
        variational: bool = False,
    ):
        super(EncoderMLP, self).__init__()
        self.latent_dims = latent_dims 
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (feature_size,) + layer_sizes + (latent_dims,)
        layers = []
        # other layers
        for l_id in range(len(layer_sizes) - 2):
            layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    activation(),
                    nn.BatchNorm1d(layer_sizes[l_id+1], affine=True) if batchnorm else nn.Identity(),
                    nn.Dropout(p=dropout) if dropout!=0 else nn.Identity(),
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x

    def get_output_size(self):
        return self.latent_dims
    
class DecoderMLP(nn.Module, Base_Encoder):
    def __init__(
        self,
        latent_dims: int,
        feature_size: int,
        layer_sizes: tuple = None,
        activation=nn.ReLU,
        dropout=0,
        batchnorm: bool = False
    ):
        super(DecoderMLP, self).__init__()
        self.latent_dims = latent_dims
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (latent_dims,) + layer_sizes
        layers = []
        for l_id in range(len(layer_sizes) - 1):
            layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    activation(),
                    nn.BatchNorm1d(layer_sizes[l_id]) if batchnorm else nn.Identity(),
                    nn.Dropout(p=dropout) if dropout!=0 else nn.Identity(),  
                )
            )
        self.layers = torch.nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.layers(x)
        return x

    def get_output_size(self):
        return self.latent_dims



class EncoderRNN(nn.Module,Base_Encoder):
    def __init__(
        self,
        latent_dims: int,
        feature_size: int,
        layer_size: int = 128,
        dropout: float =0,
        num_layers: int = 1,
        bidirectional: bool = False,
        unit_type: str="gru",
        variational: bool = False,
        batchnorm: bool = False, 
    ):
        super(EncoderRNN, self).__init__()
        self.unit_type = unit_type.lower()
        self.latent_dims = latent_dims 
        self.feature_size = feature_size 
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batchnorm = batchnorm

        if self.unit_type == "gru":
            rnn_type_class = torch.nn.GRU
        elif self.unit_type == "lstm":
            rnn_type_class = torch.nn.LSTM
        elif self.unit_type == "rnn":
            rnn_type_class = torch.nn.RNN
        else:
            pass #raise error

        self.rnn = rnn_type_class(
                input_size=self.feature_size,
                hidden_size=self.layer_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)

        self.fc = torch.nn.Sequential(
            nn.BatchNorm1d(self.layer_size) if self.batchnorm else nn.Identity(),
            torch.nn.Linear(self.layer_size, self.latent_dims)
        )


    def forward(self, x):
        rnn_out, (h_n, c_n) = self.rnn(x) 
        rnn_out = rnn_out[:, -1] # only consider output of last time step-- what about attention-aggregation
        z = self.fc(rnn_out)
        return z

    def get_output_size(self):
        return self.latent_dims

import torch, copy
from .encoders import EncoderMLP, EncoderRNN

def create_encoder_model(input_dims, emb_dims: int, model_type: str = "mlp", n_layers: int = 2, batchnorm=False,dropout=0 ,**args):
    model_type = model_type.lower()
    args = copy.deepcopy(args)

    args["latent_dims"] = emb_dims
    args["feature_size"] = input_dims 
    args["batchnorm"] = batchnorm
    args["dropout"] = dropout
    if model_type == 'mlp':
        #map arguments to arguments required for Encoder class
        if "layer_size" in args:
            args["layer_sizes"] = tuple([args["layer_size"] for i in range(n_layers)])
            del args["layer_size"] #because Encoder do not accept it
        if "layer_sizes" in args:
            args["layer_sizes"] = tuple(args["layer_sizes"])
        else: #default setting of architecture
            args["layer_sizes"] = tuple([128 for i in range(n_layers)])

        return EncoderMLP(**args)
    
    elif model_type =="rnn" or model_type== "gru" or model_type== "lstm":
        args["num_layers"] = n_layers
        args["unit_type"] = model_type
        return EncoderRNN(**args)
        
    else:
        raise ValueError(f'Invalid value for model_type: {self.model_type}. Valid values: ["mlp","cnn","rnn","gru","lstm"]')
import numpy as np
import torch
from typing import List, Union, Dict

class Lambda(torch.nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func
    def forward(self, x): return self.func(x)

def get_loss_by_name(name, **loss_args):
    #https://pytorch.org/docs/stable/nn.html#loss-functions
    name = name.strip().lower().replace("_","")
    if ("cross" in name and "entr" in name) or name=="ce":
        return torch.nn.CrossEntropyLoss(reduction="mean", **loss_args)
    elif ("bin" in name and "entr" in name) or name=="bce":
        return torch.nn.KLDivLoss(reduction="mean")
    elif name == "kl" or name=="divergence": 
        return torch.nn.KLDivLoss(reduction="mean")
    elif name == "mse" or name =="l2":
        return torch.nn.MSELoss(reduction='mean')
    elif name == "mae" or name =="l1":
        return torch.nn.L1Loss(reduction='mean')

def detach_all(z):
    if isinstance(z, dict):
        z_ = {}
        for k, v in z.items():
            z_[k] = detach_all(v)
        z = z_
    elif isinstance(z, list):
        z = [z_.detach().cpu().numpy() for z_ in z]
    else:
        z = z.detach().cpu().numpy()
    return z

def collate_all_list(z, z_):
    if isinstance(z_, dict):
        for k, v in z_.items():
            collate_all_list(z[k], v)
    elif isinstance(z_,list):
        for i, z_i in enumerate(z_):
            z[i].append( z_i )
    else:
        z.append(z_) 

def object_to_list(z):
    if isinstance(z, dict):
        for k, v in z.items():
            z[k] = object_to_list(v)
        return z
    elif isinstance(z,list):
        return [ [z_i] for z_i in z]
    else:
        return [z]

def stack_all(z_list):
    if isinstance(z_list, dict):
        for k, v in z_list.items():
            z_list[k] = stack_all(v)
    elif isinstance(z_list[0], list):
        for i, v in enumerate(z_list):
            z_list[i] = stack_all(v)
    elif isinstance(z_list, list):
        z_list = np.concatenate(z_list, axis = 0)
    else:
        print(type(z_list))
        pass
    return z_list

def get_dic_emb_dims(encoders: Dict[str,torch.nn.Module], emb_dims: Union[int, List[int], Dict[str,int]]=0) -> dict:
    return_emb_dims = {}
    for i,  view_name in enumerate(encoders):
        if hasattr(encoders[view_name], "get_output_size"):
            return_emb_dims[view_name] = encoders[view_name].get_output_size()
        else:
            if type(emb_dims) == int:
                return_emb_dims[view_name] = emb_dims
            elif type(emb_dims) == list:
                return_emb_dims[view_name] = emb_dims[i]
            elif type(emb_dims) == dict:
                return_emb_dims[view_name] = emb_dims[view_name]
            else: 
                raise Exception("if the encoders do not have the method 'get_output_size', please indicate it on the init of this class, as emb_dims=list or int")
    return return_emb_dims 

def map_encoders(encoders: Union[List[torch.nn.Module],Dict[str,torch.nn.Module]], view_names: List[str] = []) -> dict:
    if type(encoders) == list: 
        if len(view_names) == 0:
            view_names = [f"S{i}" for i in range(1, 1+len(encoders))]     
        if len(view_names) !=  len(encoders):
            raise Exception("view names is not from the same size as encoder list")
        encoders = {v_name: encoders[i] for i,v_name in enumerate(view_names)}
    elif type(encoders) != dict:
        raise Exception("Encoders must be a dict or list")
    return encoders
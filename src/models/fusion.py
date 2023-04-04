from torch import nn
from typing import List, Union, Dict

from .base_fusion import MVFusion, MVFusionMultiLoss, SVPool
from .fusion_module import FusionModule

class InputFusion(MVFusion):
    def __init__(self, 
                 predictive_model, 
                 fusion_module: dict = {}, 
                 loss_args: dict = {}, 
                 view_names: List[str] = [],
                 input_dim_to_stack: Union[List[int], Dict[str,int]] = 0,
                 ):
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "concat", "adaptive":False, "emb_dims": input_dim_to_stack }
            fusion_module = FusionModule(**fusion_module)
        fake_view_encoders = [nn.Identity() for _ in range(fusion_module.N_views)] 
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(InputFusion, self).__init__(fake_view_encoders, fusion_module, predictive_model,
            loss_args=loss_args, view_names=view_names)

class DecisionFusion(MVFusion):
    def __init__(self, 
                 view_encoders,  
                 fusion_module: dict = {},  
                 loss_args: dict ={}, 
                 view_names: List[str] = [], 
                 n_outputs: int = 0,
                 ):
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "sum", "adaptive":False, "emb_dims":[n_outputs for _ in range(len(view_encoders))]}
            fusion_module = FusionModule(**fusion_module)
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(DecisionFusion, self).__init__(view_encoders, fusion_module, nn.Identity(),
            loss_args=loss_args, view_names=view_names)
        self.n_outputs = n_outputs

class FeatureFusion(MVFusion):
    def __init__(self, 
                 view_encoders,  
                 fusion_module: nn.Module, 
                 predictive_model: nn.Module,
                 loss_args: dict ={}, 
                 view_names: List[str] = [], 
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(FeatureFusion, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_args=loss_args, view_names=view_names)

        #normalize features -- helps stability between multiple models 
        for v_name in self.views_encoder:
            if type(self.views_encoder[v_name]) != type(nn.Identity()) : 
                get_func = self.views_encoder[v_name].get_output_size
                self.views_encoder[v_name] = nn.Sequential(self.views_encoder[v_name], 
                                      nn.BatchNorm1d(self.emb_dims[v_name], affine=False))
                self.views_encoder[v_name].get_output_size = get_func

class FeatureFusionMultiLoss(MVFusionMultiLoss): #same asFeatureFusion
    def __init__(self, 
                 view_encoders,  
                 fusion_module: nn.Module, 
                 predictive_model: nn.Module,
                 loss_args: dict ={}, 
                 view_names: List[str] = [], 
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(FeatureFusionMultiLoss, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_args=loss_args, view_names=view_names)

        #normalize features -- helps stability
        for v_name in self.views_encoder:
            if type(self.views_encoder[v_name]) != type(nn.Identity()) : 
                get_func = self.views_encoder[v_name].get_output_size
                self.views_encoder[v_name] = nn.Sequential(self.views_encoder[v_name], 
                                      nn.BatchNorm1d(self.emb_dims[v_name], affine=False))
                self.views_encoder[v_name].get_output_size = get_func


class SingleViewPool(SVPool):
    pass
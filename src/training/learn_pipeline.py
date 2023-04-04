import shutil, os, sys, gc, time
from typing import List, Union, Dict
import numpy as np
from pathlib import Path

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl

from src.models.fusion import FeatureFusion, FeatureFusionMultiLoss
from src.models.fusion import InputFusion, DecisionFusion, SingleViewPool
from src.models.models import create_encoder_model
from src.models.fusion_module import FusionModule
from src.models.utils import get_dic_emb_dims

from src.datasets.utils import _to_loader


def prepare_loggers(data_name, method_name, run_id, folder_c, tags_ml, run_id_mlflow,monitor_name, **early_stop_args):
    save_dir_tsboard = f'{folder_c}/tensorboard_logs/'
    save_dir_chkpt = f'{folder_c}/checkpoint_logs/'
    save_dir_mlflow = f'{folder_c}/mlruns/'
    exp_folder_name = f'{data_name}/{method_name}'
    
    for v in Path(f'{save_dir_chkpt}/{exp_folder_name}/').glob(f'r={run_id:02d}*'):
        v.unlink()
    if os.path.exists(f'{save_dir_tsboard}/{exp_folder_name}/version_{run_id:02d}'):
        shutil.rmtree(f'{save_dir_tsboard}/{exp_folder_name}/version_{run_id:02d}')
    early_stop_callback = EarlyStopping(monitor=monitor_name, **early_stop_args)
    tensorlogg = TensorBoardLogger(name="", save_dir=f'{save_dir_tsboard}/{exp_folder_name}/')
    checkpoint_callback = ModelCheckpoint(monitor=monitor_name, mode=early_stop_args["mode"], every_n_epochs=1, save_top_k=1, 
        dirpath=f'{save_dir_chkpt}/{exp_folder_name}/', filename=f'r={run_id:02d}-'+'{epoch}-{step}-{val_objective:.2f}') 
    tags_ml = dict(tags_ml,**{"data_name":data_name,"method_name":method_name})
    if run_id_mlflow == "ind":
        mlf_logger = MLFlowLogger(experiment_name=exp_folder_name, run_name = f"version-{run_id:02d}",
                              tags = tags_ml, tracking_uri=f"file:{save_dir_mlflow}")
    else:
        mlf_logger = MLFlowLogger(experiment_name=data_name, run_name = method_name,
                              run_id= run_id_mlflow,
                              tags = tags_ml, tracking_uri=f"file:{save_dir_mlflow}")
    return {"callbacks": [early_stop_callback,checkpoint_callback], "loggers":[mlf_logger,tensorlogg]}

def log_additional_mlflow(mlflow_model, trainer, model, architecture):
    mlflow_model.experiment.log_artifact(mlflow_model.run_id, trainer.checkpoint_callback.best_model_path, artifact_path="models")
    mlflow_model.experiment.log_text(mlflow_model.run_id, str(model), "models/model_summary.txt")
    mlflow_model.experiment.log_dict(mlflow_model.run_id, model.count_parameters(), "models/model_parameters.yaml")
    mlflow_model.experiment.log_param(mlflow_model.run_id, "type_fusion", model.where_fusion)
    mlflow_model.experiment.log_param(mlflow_model.run_id, "emb_dims", model.emb_dims)
    mlflow_model.experiment.log_param(mlflow_model.run_id, "feature_pool", model.feature_pool)
    if model.where_fusion =="feature":
        mlflow_model.experiment.log_param(mlflow_model.run_id, "joint_dim", model.fusion_module.get_info_dims()["joint_dim"] )
    
def InputFusion_train(train_data: dict, val_data = None,
                data_name="", method_name="", run_id=0, output_dir_folder="", run_id_mlflow=None,
                training={}, architecture= {}, **kwargs):
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]

    start_time_pre = time.time()
    folder_c = output_dir_folder+"/run-saves"
    
    if "weight" in loss_args: 
        n_labels = np.max(train_data["target"]) +1
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    else:
        n_labels = 1 
        
    #MODEL DEFINITION
    feats_dims = [v.shape[-1] for v in train_data["views"]]
    args_model = {"input_dim_to_stack": feats_dims, "loss_args": loss_args}

    if "dropout" not in architecture["predictive_model"]:
            architecture["predictive_model"]["dropout"] = 0.2 
    if "dropout" not in architecture["encoders"]:
        architecture["encoders"]["dropout"] = 0.4 
    encoder_model = create_encoder_model(np.sum(feats_dims), emb_dim, **architecture["encoders"])
    predictive_model = create_encoder_model(emb_dim, n_labels, model_type="mlp", **architecture["predictive_model"])
    full_model = torch.nn.Sequential(encoder_model, predictive_model)
    #FUSION DEFINITION
    model = InputFusion(predictive_model=full_model, view_names=train_data["view_names"], **args_model) 

    #DATA DEFITNION
    if type(val_data) != type(None):
        val_dataloader = _to_loader(val_data, batch_size=batch_size, train=False)
        monitor_name = "val_objective"
    else:
        monitor_name = "train_objective"
    train_dataloader = _to_loader(train_data, batch_size=batch_size)
    
    extra_objects = prepare_loggers(data_name, method_name, run_id, folder_c, model.hparams_initial, run_id_mlflow, monitor_name, **early_stop_args)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, 
                         callbacks=extra_objects["callbacks"],logger=extra_objects["loggers"])
    trainer.fit(model, train_dataloader, val_dataloaders=(val_dataloader if type(val_data) != type(None) else None))

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    log_additional_mlflow(trainer.loggers[0], trainer, model, architecture)
    return model, trainer

def assign_multifusion_name(training = {}, method = {}):
    method_name = "MuFu"
    if method["feature"]:
        method_name += f"_Feat_{method['agg_args']['mode']}"
    else:
        method_name += f"_Deci_{method['agg_args']['mode']}"

    if "adaptive" in method["agg_args"]:
        if method["agg_args"]["adaptive"]:
            method_name += "_adapt"
    if "features" in method["agg_args"]:
        if method["agg_args"]["features"]:
            method_name += "F"     

    if "multi" in training["loss_args"]: 
        if training["loss_args"]["multi"]:
            method_name += "_MuLoss"
    return method_name


def MultiFusion_train(train_data: dict, val_data = None, 
                      data_name="", run_id=0, output_dir_folder="", method_name="", run_id_mlflow=None,
                     training = {}, method = {}, architecture={}, **kwargs):
    if method_name == "":
        method_name = assign_multifusion_name(training, method)
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"] 

    start_time_pre = time.time()
    folder_c = output_dir_folder+"/run-saves"   

    if "weight" in loss_args:
        n_labels = np.max(train_data["target"]) +1
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    else:
        n_labels = 1
        
    #MODEL DEFINITION -- ENCODER
    views_encoder  = {}
    for i, view_n in enumerate(train_data["view_names"]): 
        views_encoder[view_n] = create_encoder_model(train_data["views"][i].shape[-1], emb_dim, **architecture["encoders"][view_n]) 
    #MODEL DEFINITION -- Fusion-Part
    if "multi" in loss_args:
        multi = loss_args.pop("multi")
    else:
        multi = False
    args_model = {"loss_args": loss_args}      
    if method["feature"]:
        method["agg_args"]["emb_dims"] = get_dic_emb_dims(views_encoder)
        fusion_module = FusionModule(**method["agg_args"])
        input_dim_task_mapp = fusion_module.get_info_dims()["joint_dim"] 

        predictive_model = create_encoder_model(input_dim_task_mapp, n_labels, model_type="mlp", **architecture["predictive_model"])
        if multi:
            model = FeatureFusionMultiLoss(views_encoder, fusion_module, predictive_model,view_names=list(views_encoder.keys()), **args_model) 
        else:
            model = FeatureFusion(views_encoder, fusion_module, predictive_model,view_names=list(views_encoder.keys()), **args_model) 

    else:#decision
        method["agg_args"]["emb_dims"] = [n_labels for _ in range(len(views_encoder))] 
        fusion_module = FusionModule(**method["agg_args"])

        predictive_models = {}
        for view_n in views_encoder: 
            pred_ = create_encoder_model(emb_dim, n_labels, model_type="mlp", **architecture["predictive_model"] )
            predictive_models[view_n] = torch.nn.Sequential(views_encoder[view_n], pred_)
        model = DecisionFusion(view_encoders=predictive_models, fusion_module=fusion_module,view_names=list(views_encoder.keys()),**args_model)                       
    loss_args["multi"] = multi
    #DATA DEFINITION -- 
    if type(val_data) != type(None):
        val_dataloader = _to_loader(val_data, batch_size=batch_size, train=False)
        monitor_name = "val_objective"
    else:
        monitor_name = "train_objective"
    train_dataloader = _to_loader(train_data, batch_size=batch_size)
    
    #FIT
    extra_objects = prepare_loggers(data_name, method_name, run_id, folder_c, model.hparams_initial, run_id_mlflow, monitor_name, **early_stop_args)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, 
                         callbacks=extra_objects["callbacks"],logger=extra_objects["loggers"])
    trainer.fit(model, train_dataloader, val_dataloaders=(val_dataloader if type(val_data) != type(None) else None))
    
    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    log_additional_mlflow(trainer.loggers[0], trainer, model, architecture)
    return model, trainer


def PoolEnsemble_train(train_data: dict, val_data = None, 
                      data_name="", run_id=0, output_dir_folder="", method_name="MuFu", run_id_mlflow=None,
                     training = {}, architecture={}, **kwargs):
    start_time_pre = time.time()
    folder_c = output_dir_folder+"/run-saves"
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"] 
    if "weight" in loss_args:
        n_labels = np.max(train_data["target"]) +1
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    else:
        n_labels = 1 
        
    #MODEL DEFINITION -- ENCODER
    views_encoder  = {}
    for i, view_n in enumerate(train_data["view_names"]): 
        views_encoder[view_n] = create_encoder_model(train_data["views"][i].shape[-1], emb_dim, **architecture["encoders"][view_n]) 
    predictive_model = create_encoder_model(emb_dim, n_labels, model_type="mlp", **architecture["predictive_model"])
    args_model =  {"loss_args":loss_args, "view_names":train_data["view_names"]}
    model = SingleViewPool(views_encoder, predictive_model, **args_model)               

    #DATA DEFINITION -- 
    if type(val_data) != type(None):
        val_dataloader = _to_loader(val_data, batch_size=batch_size, train=False)
        monitor_name = "val_objective"
    else:
        monitor_name = "train_objective"
    train_dataloader = _to_loader(train_data, batch_size=batch_size)
    
    #FIT
    extra_objects = prepare_loggers(data_name, method_name, run_id, folder_c, model.hparams_initial, run_id_mlflow, monitor_name, **early_stop_args)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, 
                         callbacks=extra_objects["callbacks"],logger=extra_objects["loggers"])
    trainer.fit(model, train_dataloader, val_dataloaders=(val_dataloader if type(val_data) != type(None) else None))
    
    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    log_additional_mlflow(trainer.loggers[0], trainer, model, architecture)
    return model, trainer
import numpy as np
import torch
from typing import List, Union, Dict

def check_discrete(v):
    v = np.squeeze(v)
    if v % int(v) == 0:
        return True
    else:
        return False

class DataViews_torch(torch.utils.data.Dataset):
    def __init__(self, views: Union[List[np.ndarray],Dict[str,np.ndarray]],  #you can feed with dictionary, but it create list
                    target: list =[], 
                    view_names: List[str] = [],
                    view_first: bool = True,
                    view_names_data: List[List[str]] = [], 
                    return_list: bool = False, 
                    **kwargs):
        super(DataViews_torch,self).__init__()
        self.view_names = list(view_names) 
        self.view_names_data = list(view_names_data)
        self.target = list(target) 
        self.view_first = view_first
        self.return_list = return_list
        if (not self.view_first) and len(self.view_names_data) == 0:
            raise Exception("view_first=True only work by giving view_names_data as additional argument")

        if type(views) == dict:
            self.view_names = list(views.keys())
            self.views = [views[v_name] for v_name in self.view_names]
        else:
            self.views = views #already a list [view for view in views] #to copy them
        self.views = list(self.views) #it work with lists
        
        if len(self.view_names) == 0:
            self.view_names = [f"S{i}" for i in range(1,1+len(self.views))]
        
        if len(self.target) != 0:
            self.supervised_tag = True
            self.classifi = check_discrete(self.target[0])
        else:
            self.supervised_tag=False

    def __len__(self):
        if self.view_first:
            return len(self.views[0])
        else:
            return len(self.views)

    def __getitem__(self, index):
        #return a dictionary
        #key: "views" return a dictionary of {view-name: array or tensor}
            # if views are given as a list, you also should return a key: "view_names" with the names of the views
        #key: "index" return the index of the requested item
        #key: "target" return the target as array or tensor
        if self.supervised_tag:
            if self.classifi: #perhaps inside the model (if loss is CE)
                target = self.transform_type(self.target[index], "int")
            else:
                target = self.transform_type(self.target[index] , "float32")
        else:
            target = -1
        if self.view_first:
            views = {view_n: self.transform_type(view[index],"float32") for view, view_n in zip(self.views,self.view_names)}
        else: #it require "view_names_data" array
            views = {view_n: self.transform_type(view, "float32") for view, view_n in zip(self.views[index],self.view_names_data[index])}
        return {"views": views if not self.return_list else list(views.values()), "index": index, "target": target}
                #"view_names":self.view_names} #however, model still work with list, by giving view_names as input
    
    def transform_type(self,x, dt):
        if type(x)==np.ndarray:
            return x.astype(dt)
        else: #torch
            return x.to(dt)

    def get_torch_dataloader(self, batch_size=32, train=True, **kwargs):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            drop_last=False, #train,
            num_workers=0,
            shuffle=train,
            **kwargs
        )
import numpy as np


class NormalizeDataClass(object):
    def __init__(self, v_min = -1, v_max= 1, form="zscore"):
        super(NormalizeDataClass).__init__()
        self.mean = []
        self.std = []
        self.max = []
        self.min = []

        self.v_min = v_min
        self.v_max = v_max
        self.form = form.lower()

    def fit(self, data: np.ndarray):
        axis_to_norm = tuple(range(data.ndim - 1))    
        self.mean = np.nanmean(data, axis = axis_to_norm,  keepdims=True)
        self.std = np.nanstd(data, axis = axis_to_norm,  keepdims=True)
        self.max = np.nanmax(data, axis = axis_to_norm,  keepdims=True) 
        self.min = np.nanmin(data, axis = axis_to_norm,  keepdims=True)

    def transform(self, data:np.ndarray):
        if self.form == "zscore":
            return (data - self.mean)/self.std
        elif self.form == "minmax": 
            return self.v_min + (self.v_max - self.v_min)* (data - self.min)/(self.max - self.min)
        elif self.form == 'max': 
            return data/self.max
        elif self.form == "minmax-01":
            return (data - self.min)/(self.max - self.min)

        elif self.form == "zscore-own": 
            axis_to_norm = tuple(range(1,data.ndim-1))  
            self.own_mean = np.nanmean(data, axis=axis_to_norm, keepdims=True)
            self.own_std = np.nanstd(data, axis=axis_to_norm, keepdims=True)
            return (data - self.own_mean)/self.own_std
        elif self.form == "minmax-own": 
            axis_to_norm = tuple(range(1,data.ndim-1))  
            self.own_max = np.nanmax(data, axis=axis_to_norm, keepdims=True)
            self.own_min = np.nanmin(data, axis=axis_to_norm, keepdims=True)
            return self.v_min + (self.v_max - self.v_min)* (data - self.own_min)/(self.own_max - self.own_min)

    def __call__(self, data:np.ndarray): 
        return self.transform(data)

def build_fillna_func(values):
    def fill_nan(arr):
        new_array = []
        for col in range(arr.shape[1]):
            new_array.append( np.nan_to_num(arr[:, col], nan= values[col]) )
        return np.vstack(new_array).T
    return fill_nan

def preprocess_views(data, train=True, funcs=[], fillnan=False, flatten=False, form="zscore") -> list:
    #return function to norm test data and function to fill nans
    if train:
        norm_func_list = {}
        for view_name in data.get_view_names():
            aux_data = data.get_view_data(view_name)["views"]
            norm_func_list[view_name] = NormalizeDataClass(form=form) #
            norm_func_list[view_name].fit(aux_data)
    else:
        norm_func_list = funcs[0]
    print(f"Normalize views")
    data.apply_views(norm_func_list) 
    
    return_ = [norm_func_list]
    if fillnan:
        print("Fill nans")
        #data.apply_views(fillna_func_list) 
        #return_.append(fillna_func_list)
        data.apply_views(lambda x: np.nan_to_num(x, nan= 0)) #since data already centered
        
    if flatten:
        print("Flatten views")
        data.flatten_views()    
    return return_
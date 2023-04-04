import copy, gc, sys, os , pickle
import numpy as np
import pandas as pd
import xarray as xray
from pathlib import Path
from typing import List, Union, Dict

from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

def check_discrete(v):
    v = np.squeeze(v)
    if v % int(v) == 0:
        return True
    else:
        return False

class DataViews(object):
    """a structure to handle the views in variable sence for the dataset,
    an instance could have variable number of views than other.
    calculate correlation between two arrays.
    
    Example: one item, one data example, could contain several views.
    n-views: number of views
    n-examples: number of examples
    
    Attributes
    ----------
        views_data : dictionary 
            with the data {key:name}: {view name:array of data in that view}
        views_data_ident2indx : dictionary of dictionaries
            with the data {view name: dictionary {index:identifier} }
        inverted_ident : dictionary 
            with {key:indx}: {identifier:list of views name (index) that contain that identifier}
        view_names : list of string
            a list with the view names
        views_cardinality : dictionary 
            with {key:name}: {view name: n-examples that contain that view}
        train_mask_identifiers : dictionary
            with boolean mask for each identifier if it is train or not
        identifiers_target: dictionary
            with the target corresponding ot each index example
        target_names : list of strings
            list with string of target names, indicated in the order
        supervised_tag : boolean
            if the data is supervised or not (contain target data)
    """ 
    def __init__(self, views_to_add: Union[list, dict] = [], identifiers:List[int] = [], view_names: List[str] =[] , target: List[int] =[]):
        #init only work with full-view data

        """initialization of attributes. You also could given the views to add in the init function to create the structure already, without using add_view method

        Parameters
        ----------
            views_to_add : list or dict of numpy array, torch,tensor or any
                the views to add
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the views_to_add
            view_names : list of string 
                the name of the views being added
            target: list of int
                the target values if available (e.g. supervised data)
        """

        ## if views to add given, it create the instance with views already saved
        self.views_data = {}
        self.views_data_ident2indx = {} 
        self.inverted_ident = {} 
        self.view_names = []
        self.views_cardinality = {} 
        self.train_mask_identifiers = {}
        self.identifiers_target = {}
        self.target_names = ["unsupervised"]
        self.supervised_tag = False
        
        if len(views_to_add) != 0:
            if len(identifiers) == 0:
                identifiers = np.arange(len(views_to_add[0]))
            if len(view_names) == 0 and type(views_to_add) != dict:
                view_names = ["S"+str(v) for v in np.arange(len(views_to_add))]
            elif len(view_names) == 0 and type(views_to_add) == dict:
                view_names = list(views_to_add.keys())

            for v in range(len(views_to_add)):
                if type(views_to_add) == list or type(views_to_add) == np.ndarray:
                    self.add_view(views_to_add[v], identifiers, view_names[v])
                if type(views_to_add) == dict:
                    self.add_view(views_to_add[view_names[v]], identifiers, view_names[v])
            if len(target) != 0:
                self.supervised_tag = True
                self.add_target(target, identifiers)
        
    def add_target(self, target_to_add: Union[list,np.ndarray], identifiers: List[int], target_names: List[str] = [], update: bool =True):
        """add a target for the corresponding identifiers indicated, it also works by updating target

        Parameters
        ----------
            target_to_add : list, np.array or any structure that could be indexed as ARRAY[i]
                the target values to add 
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the target_to_add
            target_names : list of str 
                the target names if available (e.g. categorical targets)
            update: bool
                whether update the target when identifiers match an already saved value
        """
        for i, ident in enumerate(identifiers): 
            v = target_to_add[i]
            if type(v) == list: #or type(v) == np.ndarray:
                v = np.asarray( v)
            else:
                v = int(v) if check_discrete(v) else v #just for save memory in storage
                v = np.asarray([v])
            if ident not in self.identifiers_target:
                self.identifiers_target[ident] = v
            else:
                if (self.identifiers_target[ident] != target_to_add[i]) and update:
                    print("There are target that are being updated, if you dont want this behavior, set update=False")
                    self.identifiers_target[ident] = v
        if len(target_names) == 0:
            self.target_names = [f"T{i}" for i in range(len(self.identifiers_target[identifiers[-1]]))]
        else:
            self.target_names = target_names
        self.supervised_tag = True
        
    def get_target(self, identifiers: List[int] = [], matrix:bool=False) -> dict:
        """ get all the target associated to each identifiers

        Parameters
        ----------
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the view_to_add
        Returns
        -------
            dictionary of values
                "target": list of target for the corresponding identifiers
                "names": if the target is categorical and has available name, it will be returned also
        """

        if self.supervised_tag:
            if len(identifiers) == 0:
                target_dic = self.identifiers_target
            else:
                target_dic = {ident: self.identifiers_target[ident] for ident in identifiers}
            if matrix:
                target_dic = np.asarray(list(target_dic.values()))
            return {"target":target_dic, "target_names":self.target_names, "identifiers": identifiers}
        else:
            print("The dataset do not contain target")
            
    def flatten_views(self):
        """ flatten each view into a 2D-array (matrix)
        """
        for name in self.view_names:
            data = self.views_data[name]
            self.views_data[name] = data.reshape(data.shape[0], -1)

    def add_view(self, view_to_add, identifiers: List[int], name: str):
        """add a view array based on identifiers of list and name of the view. The identifiers is used to match the view with others views.

        Parameters
        ----------
            view_to_add : numpy array
                the array of the view to add (no restriction in dimensions or shape)
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the view_to_add
            name : string 
                the name of the view being added
        """
        if name in self.view_names:
            print("The view is already saved, try update")
            return
        self.view_names.append(name)
        self.views_data[name] = np.asarray(view_to_add, dtype="float32")
        self.views_cardinality[name] = len(identifiers)

        #update inverted identifiers of items
        self.views_data_ident2indx[name] = {}
        for indx, ident in enumerate(identifiers):
            if ident not in self.inverted_ident:
                self.inverted_ident[ident] = [len(self.view_names)-1]
                self.train_mask_identifiers[ident] = True
            else:
                self.inverted_ident[ident].append(len(self.view_names)-1)
            self.views_data_ident2indx[name][ident] = indx
    
    def remove_view(self, name: str):
        """remove a view array for the dictionary classes based on the name of the view

        Parameters
        ----------
            name : string with the name of the view
        """
        del self.views_data[name]
        del self.views_data_ident2indx[name]
        #del self.views_data_ident_list[name]
        del self.views_cardinality[name]
        self.view_names.remove(name)
        for ident in self.inverted_ident.keys():
            if name in self.inverted_ident[ident]:
                self.inverted_ident[ident].remove(np.where(self.view_names == name)[0][0])
                if len(self.inverted_ident[ident]) == 0:
                    del self.inverted_ident[ident]
                    if self.supervised_tag:
                        del self.identifiers_target[ident]
        gc.collect() 
        
    def get_n_total(self):
        return len(self.inverted_ident)

    def __len__(self) -> int:
        return len(self.views_data)

    def __getitem__(self, i: int):
        """
        Parameters
        ----------
            i : int value that correspond to the example to get (with all the views available)

        Returns
        -------
            dictionary with three values
                data : numpy array of the example indicated on 'i' arg
                views : a list of strings with the views available for that example
                train? : a mask indicated if the example is used for train or not    
        """
        return self.get_item_ident(i)

    def get_item_ident(self, identifier: int):
        if identifier not in self.inverted_ident:
            raise Exception("identifier requested not available for the data in the DataViews class")
        viewsname_based_ident = self.inverted_ident[identifier]
        if np.isnan(viewsname_based_ident).sum() > 0 : 
            viewsname_based_ident = np.asarray(viewsname_based_ident, dtype="int")[~np.isnan(viewsname_based_ident)].tolist()
        views_available = self.get_view_names(viewsname_based_ident)
        S_data = [self.views_data[view][self.views_data_ident2indx[view][identifier]] for view in views_available]
        return_info = {"views": S_data, 'view_names':views_available, 'train?': self.train_mask_identifiers[identifier], "identifier": identifier}
        
        if self.supervised_tag:
            if identifier in self.identifiers_target:
                return_info["target"] = self.identifiers_target[identifier]
            else:
                return_info["target"] = np.asarray([-1]) 
        return return_info

    def filter_dict_by_views(self, views_data_dict: Dict[str, list], view_names_to_return: List[str]):
        return [views_data_dict[view_n] for view_n in view_names_to_return if view_n in views_data_dict ]
        

    def get_item_selected_views(self, identifier: int, view_names: List[str]):
        """same as getitem method but masking on selected views
        Parameters
        ----------
            view_names : list of views to obtain the example 
            identifier : the identifier of the data to obtain
            
        Returns
        -------
            same as getitem method            
        """
        return_info = self[identifier]
        view_n_data_dict = dict(zip(return_info["view_names"], return_info["views"]))
        return_info["views"] = self.filter_dict_by_views(view_n_data_dict, view_names)
        return_info["view_names"] = view_names
        return return_info

    def get_view_data(self, name: str):
        """get the numpy array of the view.
        The views inside the structure are not necesarrily in the same order, i.e. the examples contained are not related in the same axis. If you want to obtain same examples for each view run the generate_full_view_data method.

        Parameters
        ----------
            name : string with the name of the view

        Returns
        -------
            numpy array of the view indicated in 'name' param
            
        """
        return {"views":self.views_data[name], "identifiers": list(self.views_data_ident2indx[name].keys()) , "view_names": [name]}

    def get_views_card(self, view_names: List[str]=[]):
        """get views cardinality or n-examples in each view.

        Parameters
        ----------
            view_names : list of string with the name of the views to calculate cardinality

        Returns
        -------
            a dictionary with the name of the indicated view in key and the cardinality in value
            
        """
        if len(view_names) == 0:
            return self.views_cardinality
        else:
            return {view_n: self.views_cardinality[view_n] for view_n in view_names}

    def get_all_identifiers(self) -> list:
        """get the identifiers of all views on the structure
     
        Returns
        -------
            list of identifiers
            
        """
        #set by train and test also
        return list(self.inverted_ident.keys())

    def get_view_names(self, indexs: List[int] = []) -> List[str]:
        """get the view names

        Parameters
        ----------
            indexs : if the index of which the view names will be returned
        
        Returns
        -------
            all the view names used in the structure
            
        """
        if len(indexs) == 0:
            return self.view_names
        else:
            return np.asarray(self.view_names)[indexs].tolist()

    def get_view_shapes(self, view_names: List[str]=[]):
        if len(view_names) == 0:
            view_names = self.view_names
        return {name: self.views_data[name].shape[1:] for name in view_names}
    def get_n_per_views(self, view_names: List[str]=[]):
        if len(view_names) == 0:
            view_names = self.view_names
        return {name: self.views_data[name].shape[0] for name in view_names}

    def generate_all_view_data(self, train: bool =True):
        """obtain all the data loaded in this structure with all the views available, 
        each view could be of variable length and variable representation and each 
        example could contain a variable number of views, therefore the return of 
        the method is based on each example. return = n-samples x views x data dimension on each view

        Parameters
        ----------
            train : if the examples to obtain should be training examples or not

        Returns
        -------
            dictionary with three values
                data : a list of numpy arrays with the examples, len(data) == n-examples, each example contain a list with the views available for that example
                    n-samples x n-views x data dimension on each view
                views : a list of strings with the views available for that example
                identifiers : a list of identifiers of the data returned
        """
        print("The views available are ",self.view_names)
        S_data, view_names, Y_data = [], [], []
        for ident in self.get_all_identifiers():
            data_ = self[ident]
            S_ident, V_ident, T_mask = data_["views"], data_["view_names"], data_["train?"]
            if train == T_mask:
                S_data.append(S_ident)
                view_names.append(V_ident)
                if self.supervised_tag:
                    Y_data.append(data_["target"])
        if self.supervised_tag:
            Y_data = np.asarray(Y_data)
            if len(Y_data.shape) == 1:
                Y_data = np.expand_dims(Y_data, axis =-1)
        return {"views":S_data, "view_names_data":view_names, "view_names":self.view_names, "target":Y_data,
                "identifiers":self.get_all_identifiers()}

    def generate_full_view_data(self, view_names: List[str]=[], stack: bool = False, views_first: bool=True, train: bool =True,  N:int =-1):
        """obtain a full-view dataset, i.e. get all the examples that contain all the views on the data structure.
        
        The return of the method is based on each example. return = n-samples x views x data dimension on each view

        Parameters
        ----------
            view_names : a list of strings for each of the views to generate the full view data. An example will be selected if all the views are available for that example. If not specified, all data views will be returned. Repeated views are ignored.
            stack: ??
            views_first: if the return array should contain the view as first dimension, as a list of views for the dataset
            train : if the examples to obtain should be training examples or not
            N: the total ammount of examples to sample, if -1, retrieve all.

        Returns
        -------
            dictionary with three values
                data : a list of numpy arrays with the examples, if view_first: len(data) = n-views (and n-views x n-examples x data dimension on each view), if not view_first: len(data) = n-examples (and n-examples x n-views x data dimension on each view)
                identifiers : a list of identifiers of the data returned
            
        """        
        
        if len(view_names) == 0:
            view_names = self.get_view_names()
        view_names = list(dict.fromkeys(view_names))
        print(f"You select {len(view_names)} views from the {len(self.view_names)} available, you could use get_view_names() to check which are available, the selected views are {view_names}")

        #search by the data with the least cardinality (view with less data)
        view_less_data =  sorted(self.get_views_card(view_names).items(), key=lambda x: x[1], reverse=True)[0][0]

        S_data, Idts_data, Y_data = [] , [], []
        if views_first:
            S_data = [[] for _ in range(len(view_names))]
            print("first dimension of list will be the views, instead of the standard of n-samples")
        for ident in self.views_data_ident2indx[view_less_data].keys(): #just check the other views that has the query view
            data_ = self[ident]
            V_ident, T_mask = data_["view_names"], data_["train?"]
            if train == T_mask:
                if all(view in V_ident for view in view_names): #check if data contain all the views
                    Idts_data.append(ident)
                    if self.supervised_tag:
                        Y_data.append(data_["target"])
                    
                    if views_first:
                        data_ = self.get_item_selected_views(ident, view_names)["views"]
                        for v in range(len(view_names)):
                            S_data[v].append(data_[v])
                    else:
                        S_data.append(self.get_item_selected_views(ident, view_names)["views"])

        Idts_data = np.asarray(Idts_data)
        S_data = [np.asarray(S_data[v]) for v in range(len(view_names))] #transform each view into a fixed-size matrix
        if self.supervised_tag:
            Y_data = np.asarray(Y_data)
            if len(Y_data.shape) == 1:
                Y_data = np.expand_dims(Y_data, axis =-1)

        if N != -1:
            indx_sampled = np.random.choice( np.arange(len(Idts_data)), size=N, replace=False)
            Idts_data = Idts_data[indx_sampled]
            S_data = [S_data[v][indx_sampled] for v in range(len(view_names))]
            if self.supervised_tag:
                Y_data = Y_data[indx_sampled]
        
        if stack:
            print('stack set up, it is mandatory to be used only with same data dimension on all views')
            return {"views": np.concatenate(S_data, axis=-1), "identifiers": Idts_data, "target":Y_data, "view_names":view_names}
        else:
            
            return {"views": S_data, "identifiers": Idts_data, "target":Y_data, "view_names":view_names} 
        
    def set_test_mask(self, identifiers: List[int], reset = False):
        """set a binary mask to indicate the test examples

        Parameters
        ----------
            identifiers : list of identifiers that correspond to the test examples

        """
        if reset:
            for ident in self.train_mask_identifiers.keys():
                if ident in identifiers:
                    self.train_mask_identifiers[ident] = False
                else:
                    self.train_mask_identifiers[ident] = True
        else:
            for v in identifiers:
                self.train_mask_identifiers[v] = False
            
    def view_train_test(self):
        N_train = np.sum(list(self.train_mask_identifiers.values()))
        N_test = len(self.train_mask_identifiers.values()) - N_train
        print(f'in total there is {N_train} train and {N_test} test, corresponding {N_train/(N_train+N_test)}/{N_test/(N_train+N_test)}')
        
    def apply_views(self, func):
        #what about parallel?
        for view_name in self.view_names:
            if type(func) == dict:
                self.views_data[view_name] = func[view_name](self.views_data[view_name])  
            else:
                self.views_data[view_name] = func(self.views_data[view_name])  

    def _to_xray(self):
        data_vars = {}
        for view_n in self.get_view_names():
            data_vars[view_n] = xray.DataArray(data=self.views_data[view_n], 
                                  dims=["identifier"] +[f"{view_n}-D{str(i+1)}" for i in range (len(self.views_data[view_n].shape)-1)], 
                                 coords={"identifier": list(self.views_data_ident2indx[view_n].keys()),
                                        #"dims": 
                                        }, )  
        data_vars["train_mask"] = xray.DataArray(data=np.asarray(list(self.train_mask_identifiers.values()))*1,  #*1 to map to int
                                        dims=["identifier"], 
                                         coords={"identifier": list(self.train_mask_identifiers.keys()) })
        if self.supervised_tag:
            data_vars["target"] = xray.DataArray(data =np.asarray(list(self.identifiers_target.values())),
                dims=["identifier","dim_target"] , coords ={"identifier": list(self.identifiers_target.keys())} ) 

        #dummy way?
        ohe_views = MultiLabelBinarizer().fit_transform(self.inverted_ident.values())
        data_vars["inverted_ident"] = xray.DataArray(data=ohe_views, dims=["identifier", "view_available"], 
                   coords={"identifier": list(self.inverted_ident.keys()) })
        return xray.Dataset(data_vars =  data_vars,
                        attrs = {"view_names": self.view_names, 
                                 "target_names": self.target_names,
                                "supervised_tag": self.supervised_tag*1,
                                },
                        )

    def save(self, name_path, xarray = True, ind_views = False):
        """save data in name_path

        Parameters
        ----------
            name_path : path to a file to save the model, without extension, since extension is '.pkl.
            ind_views : if you want to save the individual views as csv files 
        """
        path_ = Path(name_path)
        name_path_, _, file_name_ = name_path.rpartition("/") 
        path_ = Path(name_path_)
        path_.mkdir(parents=True, exist_ok=True)
        if xarray and (not ind_views): 
            xarray_data = self._to_xray()
            path_ = path_ / (file_name_+".nc" if "nc" != file_name_.split(".")[-1] else file_name_)
            xarray_data.to_netcdf(path_, engine="h5netcdf") 
        elif (not xarray) and ind_views:  #only work with 2D array , for furtther dimension xarray!
            path_ = Path(name_path_ +"/"+ file_name_)
            path_.mkdir(parents=True, exist_ok=True)
            for view_name in self.get_view_names():
                view_data_aux = self.get_view_data(view_name)
                df_tosave = pd.DataFrame(view_data_aux["views"])
                df_tosave.index = view_data_aux["identifiers"]
                df_tosave.to_csv(f"{str(path_)}/{view_name}.csv", index=True)
        else: #when xarray and ind views are both on or off (XOR)
            path_ = path_ / (file_name_+".pkl" if "pkl" != file_name_.split(".")[-1] else file_name_)
            with path_.open(mode = "wb") as file:
                pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def load(self, name_path):
        """load with pickle 

        Parameters
        ----------
            name_path : path to a file to save the model, without extension, since extension is '.pkl.
        """
        return load_structure(name_path)


def load_structure(name_path: str):
    ext = name_path.split(".")[-1]
    if ("pkl" == ext): #or (not os.path.isfile(f"{name_path}.nc")):
        sys.modules['repo.views_structure'] = sys.modules[__name__]
        sys.modules["repo"] = "."
        sys.modules['views_structure'] = sys.modules[__name__]

        with open(name_path,'rb') as file:
            return pickle.load(file)
    else:  #default
        if "nc" != ext:
            name_path= name_path+'.nc'
        data  = xray.open_dataset(name_path, engine="h5netcdf")
        return xray_to_dataviews(data)

def xray_to_dataviews(xray_data: xray.Dataset):
    all_possible_index = xray_data.coords["identifier"].values
    
    dataviews = DataViews()    
    dataviews.view_names = xray_data.attrs["view_names"]
    dataviews.target_names = xray_data.attrs["target_names"]
    dataviews.supervised_tag = xray_data.attrs["supervised_tag"].astype(bool)

    #ask for a block of memory together is faster than individually
    aux_xray = xray_data["inverted_ident"]*np.arange(1,1+len(dataviews.view_names))
    if (aux_xray == 0).sum() != 0: #contain nans
        aux_xray = aux_xray.where(aux_xray != 0 ) - 1
        dataviews.inverted_ident = dict(zip(all_possible_index, aux_xray.values.tolist()))
    else:
        aux_xray = aux_xray-1
        dataviews.inverted_ident = dict(zip(all_possible_index, aux_xray.values.astype(int).tolist()))
    dataviews.train_mask_identifiers = dict(zip(all_possible_index, xray_data["train_mask"].values.astype(bool)))
    if dataviews.supervised_tag:
        dataviews.identifiers_target = dict(zip(all_possible_index, xray_data["target"].values))

    for view_n in dataviews.view_names:
        #check nans cause of missingness
        data_variable = xray_data[view_n].dropna("identifier", how = "all") #variable array for each view

        dataviews.views_data[view_n] = data_variable.values
        dataviews.views_cardinality[view_n] = len(data_variable["identifier"])
        dataviews.views_data_ident2indx[view_n] = dict(zip(data_variable["identifier"].values,np.arange(len(data_variable["identifier"]))))
    return dataviews
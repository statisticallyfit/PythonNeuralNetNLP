# All Imports
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import PreTrainedModel
#from transformers import XLNetTokenizer, XLNetForTokenClassification
#from transformers import BertTokenizer, BertForTokenClassification

import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple



# Type aliases
Name = str
Type = type
Object = object

# Type alias
ParameterName = str



import collections

# Create named tuple class with names "Names" and "Objects"
Info = collections.namedtuple("Info", ["Names", "Types", "Objects"], verbose=False, rename = False)



# Children ------------------------------------------------------------------------

def getChildInfo(model: nn.Module) -> Info: # -> Tuple[Dict[Name, Type], List[Object]]:
    """
    Uses the model's named children list to return the names of the named children along with the types of the named children objects.
    """
    # Get the names and types first
    names: List[Name] = [name for (name, _) in model.named_children()]
    types: List[Type] = [type(obj) for (_, obj) in model.named_children()]
    objs: List[Object] = [obj for (_, obj) in model.named_children()]

    return Info(Names = names, Types = types, Objects  = objs)



def printChildInfo(model: nn.Module):
    (names, types, objects) = getChildInfo(model)

    assert len(names) == len(types) == len(objects), "Lengths not equal!"

    NUM_CHILDREN: int = len(names)

    for i in range(NUM_CHILDREN):
        print(f"Child {i} \n\t | Name = {names[i]} \n\t | Type = {names[i]}")


# Parameters ------------------------------------------------------------------------

def printParamSizes(model: PreTrainedModel):
    #print(f"Number of parameters = {model.num_parameters()}")
    print(f"Length of parameter list = {len(list(model.parameters()))}")
    print(f"Number of modules = {len(list(model.named_modules()))}")




# Creating my named tuple to hold parameter information.
ParameterInfo = collections.namedtuple("ParameterInfo", ["Names", "Sizes", "Tensors"], verbose=False, rename = False)


def getParamInfo(model: nn.Module) -> ParameterInfo:
    # Getting names of all model's parameters
    names: List[ParameterName] = [name for (name, paramTensor) in model.named_parameters()]
    sizes: List[Size] = [paramTensor.size() for (name, paramTensor) in model.named_parameters()]
    tensors: List[Parameter] = [paramTensor for (name, paramTensor) in model.named_parameters()]

    return ParameterInfo(Names = names, Sizes = sizes, Tensors = tensors)


def printParamInfo(model: nn.Module):
    (names, sizes, tensors) = getParamInfo(model)

    assert len(names) == len(sizes) == len(tensors), "Param info lengths not equal!"

    # Print info
    NUM_PARAMS: int = len(names)

    for i in range(NUM_PARAMS):
        print(f"Parameter {i} \n\t | Name = {names[i]} \n\t | Size = {sizes[i]}")


# Modules --------------------------------------------------------------------------



def getModuleInfo(model: nn.Module) -> Info: # -> Tuple[Dict[ModuleName, ModuleType], List[ModuleObject]]:

    names: List[Name] = [name for (name, _) in model.named_modules()]
    types: List[Type] = [type(obj) for (_, obj) in model.named_modules()]
    objs: List[Object] = [obj for (_, obj) in model.named_modules()]

    return Info(Names = names, Types = types, Objects  = objs)


def printModuleInfo(model: nn.Module):
    (names, types, objects) = getModuleInfo(model)

    assert len(names) == len(types) == len(objects), "Lengths not equal!"

    NUM_MODULES: int = len(objects)

    for i in range(NUM_MODULES):
        print(f"Module {i} \n\t | Name = {names[i]} \n\t | Type = {types[i]} ")

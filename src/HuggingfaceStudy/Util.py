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

import string # for string.punctuation

# Type aliases
Name = str
Type = type
Object = object

# Type alias
ParameterName = str



import collections

# Create named tuple class with names "Names" and "Objects"
Info = collections.namedtuple("Info",
                              ["Names", "Types", "Objects"], verbose=False, rename = False)

#OuterInfo = collections.namedtuple("OuterInfo", ["OuterName", "InnerInfo"], verbose=False, rename = False)


# Children ------------------------------------------------------------------------

def getChildInfo(model: nn.Module) -> Info: # -> Tuple[Dict[Name, Type], List[Object]]:
    """
    Uses the model's named children list to return the names of the named children along with the types of the named  children objects inside a MODULE

    `named_children()` gives a short list with many types of children.  Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.
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
        print(f"Child {i} | Name = {names[i]} | Type = {names[i]}")


# Parameters ------------------------------------------------------------------------

def printLengthInfo(model: PreTrainedModel):
    print(f"Number of parameters = {model.num_parameters()}")
    print(f"Length of parameter list = {len(list(model.parameters()))}")
    print(f"Number of modules = {len(list(model.named_modules()))}")




# Creating my named tuple to hold parameter information.
ParameterInfo = collections.namedtuple("ParameterInfo", ["Names", "Sizes", "Tensors"], verbose=False, rename = False)


def getParamInfo(model: nn.Module) -> ParameterInfo:
    """
    Get parameters of the modules in the given model.

    `named_parameters()` Returns an iterator over module parameters, yielding both the name of the parameter as well as  the parameter itself.
    """
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
        print(f"Parameter {i} | Name = {names[i]} | Size = {sizes[i]}")


# Modules --------------------------------------------------------------------------



def getModuleInfo(model: nn.Module) -> Info: # -> Tuple[Dict[ModuleName, ModuleType], List[ModuleObject]]:

    """
    Get names and modules of all the modules in the MODEL.

    `named_modules()`:
    Returns an iterator over all modules in the network, yielding both the name of the module as well as the module itself.
    """
    names: List[Name] = [name for (name, _) in model.named_modules()]
    types: List[Type] = [type(obj) for (_, obj) in model.named_modules()]
    objs: List[Object] = [obj for (_, obj) in model.named_modules()]

    # Getting name of the upper module (the rest of the ones inside are its inner modules)
    #[outerName] = str(types[0]).split(".")[-1].split(" ")
    #cleanedOuterName: str = ''.join(letter for letter in outerName if letter not in string.punctuation)

    #return Info(Names = names[1:], Types = types[1:], Objects  = objs[1:])
    return Info(Names =names, Types =types, Objects = objs)



def printModuleInfo(model: nn.Module):
    (names, types, objects) = getModuleInfo(model)

    assert len(names) == len(types) == len(objects), "Lengths not equal!"

    NUM_MODULES: int = len(objects)

    for i in range(NUM_MODULES):
        print(f"Module {i} | Name = {names[i]} | Type = {types[i]} ")

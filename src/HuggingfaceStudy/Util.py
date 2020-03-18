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



# Children ------------------------------------------------------------------------

def getChildInfo(model: nn.Module) -> Tuple[Dict[Name, Type], List[Object]]:
    """
    Uses the model's named children list to return the names of the named children along with the types of the named children objects.
    """
    # Get the names and types first
    namesAndTypes: Dict[str, Type] = [(name, type(childObj)) for (name, childObj) in model.named_children()]
    # Get the objects first (lengthy to the eye, so zipping them separately below)
    objects: List[Object] = [childObj for (_, childObj) in model.named_children()]

    # Then pass in a tuple:
    return (namesAndTypes, objects)



def printChildInfo(model: nn.Module):
    (namesAndTypes, objects) = getChildInfo(model)

    assert len(namesAndTypes) == len(objects), "Lengths not equal!"

    NUM_NAMED_CHILDREN: int = len(namesAndTypes)

    for i in range(NUM_NAMED_CHILDREN):
        print(f"Child {i} \n\t | Name = {namesAndTypes[i][0]} \n\t | Type = {namesAndTypes[i][1]}")


# Parameters ------------------------------------------------------------------------

def printParamSizes(model: PreTrainedModel):
    #print(f"Number of parameters = {model.num_parameters()}")
    print(f"Length of parameter list = {len(list(model.parameters()))}")
    print(f"Number of modules = {len(list(model.named_modules()))}")



def getParamInfo(model: nn.Module) -> Dict[ParameterName, Size] :
    params: List[Tuple[ParameterName, Parameter]] = list(model.named_parameters())

    # Getting names of all model's parameters
    paramNameAndSizes: Dict[ParameterName, Size] = [(name, tensor.size()) for (name, tensor) in params]

    return paramNameAndSizes


def printParamInfo(model: nn.Module):
    paramInfo: Dict[ParameterName, Size] = getParamInfo(model)

    # Print info
    NUM_PARAMS: int = len(paramInfo)

    for i in range(NUM_PARAMS):
        print(f"Parameter {i} \n\t | Name = {paramInfo[i][0]} \n\t | Size = {paramInfo[i][1]}")


# Modules --------------------------------------------------------------------------

import collections

# Create named tuple class with names "Names" and "Objects"
ModuleInfo = collections.namedtuple("ModuleInfo", ["Names", "Types", "Objects"] , verbose=False, rename = False)


def getModuleInfo(model: nn.Module) -> ModuleInfo: # -> Tuple[Dict[ModuleName, ModuleType], List[ModuleObject]]:

    names: List[Name] = [name for (name, _) in model.named_modules()]
    types: List[Name] = [type(obj) for (_, obj) in model.named_modules()]
    objs: List[Name] = [obj for (_, obj) in model.named_modules()]


    return ModuleInfo(Names = names, Types = types, Objects  = objs)


def printModuleInfo(model: nn.Module):
    (names, types, objects) = getModuleInfo(model)

    assert len(names) == len(types) == len(objects), "Lengths not equal!"

    NUM_MODULES: int = len(objects)

    for i in range(NUM_MODULES):
        print(f"Module {i} \n\t | Name = {names[i]} \n\t | Type = {types[i]} ")

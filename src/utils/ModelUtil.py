# All Imports
import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple, Set

import numpy as np

import string # for string.punctuation

from pydoc import locate # for conversion of string back to type

# Type aliases
Name = str
Type = type
Object = object

# Type alias
ParameterName = str



import collections

# Create named tuple class with names "Names" and "Objects"
Info = collections.namedtuple("Info",
                              ["Names", "Types", "Objects"], #verbose=False, 
                              rename = False)


#OuterInfo = collections.namedtuple("OuterInfo", ["OuterName", "InnerInfo"], verbose=False, rename = False)






# Unique Things -----------------------------------------------------------------------------


# Converts
# "<class 'transformers.modeling_bert.BertForTokenClassification'>"
# INTO
# 'transformers.modeling_bert.BertForTokenClassification'
def cleanName(uncleanName: Name) -> Name:
    return uncleanName.split("'")[1]

def cleanTypes(typesList: List[Type]) -> List[Type]:
    #strNames: List[Name] = [str(aType) for aType in typesList]
    listOfCleanerNames: List[Name] = [cleanName(str(aType)) for aType in typesList]

    # Cast them back to type
    return [locate(cleanName) for cleanName in listOfCleanerNames]


# Converts
# transformers.modeling_bert.BertForTokenClassification'
# INTO
# BertForTokenClassification
def simplifyName(uncleanName: Name) -> Name:
    last: str = uncleanName.split(".")[-1]
    punctuationWithoutUnderscore: str = string.punctuation.replace("_", "")
    return ''.join(letter for letter in last if letter not in punctuationWithoutUnderscore)


def simplifyTypes(typesList: List[Type]) -> List[Name]:
    #strNames: List[Name] = [str(aType) for aType in typesList]
        #list(np.unique([str(aType) for aType in typesList]))
    listOfSimplerNames: List[Name] = [simplifyName(str(aType)) for aType in typesList]

    # Cast them back to type
    return listOfSimplerNames # [locate(simpleName) for simpleName in listOfSimplerNames]




def getUniqueChildren(model: nn.Module) -> Dict[Name, Type]:
    (_, types, _) = getChildInfo(model)

    listOfUniquesInDict: List[Dict[Name, Type]] = np.unique(
        dict(zip(simplifyTypes(types), cleanTypes(types)))
    )
    [uniqueChildrenDict] = listOfUniquesInDict

    return uniqueChildrenDict

def getUniqueModules(model: nn.Module) -> Dict[Name, Type]:
    (_, types, _) = getModuleInfo(model)

    listOfUniquesInDict: List[Dict[Name, Type]] = np.unique(
        dict(zip(simplifyTypes(types), cleanTypes(types)))
    )
    [uniqueChildrenDict] = listOfUniquesInDict

    return uniqueChildrenDict





# -----------------------------------------------------------------------------------



def allModulesByType(shortTypeName: Name, model: nn.Module) -> List[Name]:
    """
    Given a ashort type name, like `BertLayer` or `BertSelfAttention`, this function returns a list of all the module names which have that type.
    """
    (names, types, _) = getModuleInfo(model)
    # Get the long module name from the short  type name
    shortToLongNames: Dict[Name, Type] = getUniqueModules(model)
    LongTypeName: Type = shortToLongNames[shortTypeName]

    return [names[i] for i in range(len(types)) if types[i] == LongTypeName]

def allModuleObjsByType(shortTypeName: Name, model: nn.Module) -> Dict[Name, Object]:
    """
    Given the model, return a dictionary of PyTorch Embedding string names mapped to the Embedding object itself (since they are short)
    """
    (names, types, objs) = getModuleInfo(model)
    #PytorchEmbeddingType = torch.nn.modules.sparse.Embedding
    # Get the long module name from the short  type name
    shortToLongNames: Dict[Name, Type] = getUniqueModules(model)
    LongTypeName: Type = shortToLongNames[shortTypeName]
    # Here showing what KINDS of Embeddings, need full name not just short name ( since layernorm weight is different that attention query weight)
    return dict([(names[i], objs[i]) for i in range(len(types)) if types[i] == LongTypeName])


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

    # Cleaning the types so they print without the 'class' prefix
    cleanerTypes: List[Name] = [cleanName(str(aType)) for aType in types]

    NUM_CHILDREN: int = len(names)

    for i in range(NUM_CHILDREN):
        print(f"Child {i} | Name = {names[i]} | Type = {cleanerTypes[i]}")


# Parameters ------------------------------------------------------------------------

def printLengthInfo(model: nn.Module):
    print(f"Number of parameters = {model.num_parameters()}")
    print(f"Length of parameter list = {len(list(model.parameters()))}")
    print(f"Number of modules = {len(list(model.named_modules()))}")




# Creating my named tuple to hold parameter information.
ParameterInfo = collections.namedtuple("ParameterInfo", ["Names", "Sizes", "Tensors"], #verbose=False, 
rename = False)


def getParamInfo(model: nn.Module) -> ParameterInfo:
    """
    Get parameters of the modules in the given model.

    `named_parameters()` Returns an iterator over module parameters, yielding both the name of the parameter as well as  the parameter itself.
    """
    # Getting names of all model's parameters
    names: List[ParameterName] = [name for (name, paramTensor) in model.named_parameters()]
    sizes: List[Size] = [list(zip(paramTensor.names, paramTensor.size())) for (name, paramTensor) in model.named_parameters()]
    tensors: List[Parameter] = [paramTensor for (name, paramTensor) in model.named_parameters()]

    return ParameterInfo(Names = names, Sizes = sizes, Tensors = tensors)



def printParamInfo(model: nn.Module):
    (names, sizes, tensors) = getParamInfo(model)

    assert len(names) == len(sizes) == len(tensors), "Param info lengths not equal!"

    # Print info
    NUM_PARAMS: int = len(names)

    for i in range(NUM_PARAMS):
        print(f"Parameter {i} | Name = {names[i]} | Size = {sizes[i]}")






def briefParams(model: nn.Module) -> List[Tuple[Name, Size, Dict[Name, int]]]:
    """
    Gets the (name, paramsize, param tensor name) as tuple and returns list of these for all parameters in the model.
    Purpose: to get brief view into sizes, named dimensions, and names of parameters.
    """
    tupleList: List[(Name, List[(Name, int)])] = [(name, tuple(zip(paramTensor.names, paramTensor.size())))
                                                  for (name, paramTensor) in model.named_parameters()]
    # return [(name, tuple(paramTensor.size()), paramTensor.names) for (name, paramTensor) in model.named_parameters()]

    d = dict()
    for name, dimTuple in tupleList:
        d[name] = dimTuple

    return d
    # Converting tuple list into dict list with NAME and DIM as keys
    #dictList = []
    #for name, dimTuple in tupleList:
    #     dictList.append({'PARAM': name, 'DIM': dimTuple})

    #return dictList


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

    # Cleaning the types so they print without the 'class' prefix
    cleanerTypes: List[Name] = [cleanName(str(aType)) for aType in types]

    NUM_MODULES: int = len(objects)

    for i in range(NUM_MODULES):
        print(f"Module {i} | Name = {names[i]} | Type = {cleanerTypes[i]} ")


#
#
#
# -------------------------------------------------------------------------------------------------------------------------

def isEqualStructure(module1: nn.Module, module2: nn.Module) -> bool:
    """
    Tests that the modules are equal in terms of types and names of objects and param sizes and names.
    (So the actual numbers in the parameter weights / biases can be different)
    """
    def paramNames(model: nn.Module) -> List[Name]:
        return [paramName for (paramName, paramTensor) in model.named_parameters()]

    def dimNames(model: nn.Module) -> List[Tuple[Name, Name]]:
        return [paramTensor.names for (paramName, paramTensor) in model.named_parameters()]

    def paramSizes(model: nn.Module) -> List[Tuple[int, int]]:
        return [tuple(paramTensor.size()) for (paramName, paramTensor) in model.named_parameters()]

    def moduleNames(model: nn.Module) -> List[Name]:
        return [modName for (modName, modObj) in model.named_modules()]

    def moduleTypes(model: nn.Module) -> List[Type]:
        return [type(modObj) for (modName, modObj) in model.named_modules()]


    return paramNames(module1) == paramNames(module2) \
           and dimNames(module1) == dimNames(module2) \
           and paramSizes(module1) == paramSizes(module2) \
           and moduleNames(module1) == moduleNames(module2) \
           and moduleTypes(module1) == moduleTypes(module2)
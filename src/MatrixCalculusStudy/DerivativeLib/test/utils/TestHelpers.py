
import numpy as np
from numpy import ndarray


import itertools
from functools import reduce


from sympy import det, Determinant, Trace, Transpose, Inverse, Function, Lambda, HadamardProduct, Matrix, MatrixExpr, Expr, Symbol, derive_by_array, MatrixSymbol, Identity,  Derivative, symbols, diff

from sympy import srepr , simplify

from sympy import tensorcontraction, tensorproduct, preorder_traversal
from sympy.functions.elementary.piecewise import Undefined
from sympy.physics.quantum.tensorproduct import TensorProduct

from sympy.abc import x, i, j, a, b, c

from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul

from sympy.core.numbers import NegativeOne, Number

from sympy.core.assumptions import ManagedProperties

# TORCH
import torch
import torch.tensor as tensor

# TYPES
from typing import *

MatrixType = ManagedProperties

Tensor = torch.Tensor
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor



# PATH SETTING
import sys
import os

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

UTIL_DISPLAY_PATH: str = PATH + "/src/utils/GeneralUtil/"

MATDIFF_PATH: str = PATH + "/src/MatrixCalculusStudy/DerivativeLib"


sys.path.append(PATH)
sys.path.append(UTIL_DISPLAY_PATH)
sys.path.append(MATDIFF_PATH)



from src.utils.GeneralUtil import *
from src.MatrixCalculusStudy.DerivativeLib.main.Simplifications import *

# ------------------------------------------------



def testSimplifyAlgo(algo, expr, check: MatrixExpr, byType: MatrixType = None):
    params = [byType, expr] if not (byType == None) else [expr]
    res = algo(*params)

    showGroup([
        expr, res, check
    ])
    #assert expr.doit() == res.doit()
    try:
        assert equal(expr, res)
    except Exception:
        hasMatPow = lambda e: "MatPow" in srepr(res)

        print("ASSERTION ERROR: equal(expr, res) did not work")
        print("Had MatPow: ", hasMatPow(res))

    try:
        assert res.doit() == check.doit()
    except Exception:
        print("ASSERTION ERROR: res.doit() == check.doit() --- maybe MatPow ? ")



def testDerivAlgo(algo, expr: MatrixExpr, byVar: MatrixExpr, groupedCheck: MatrixExpr, onlineCheck: MatrixExpr):

    res = algo(expr, byVar)

    print("Expr:")
    display(freeze(expr))
    
    print("\nResult:")
    display(res)

    print("\nResult (polarized):")
    display(polarize(Transpose, res))

    print("\nCheck (grouped version):")
    display(freeze(groupedCheck))

    # Online from matrixcalculus.org
    print("\nCheck (online)")
    display(freeze(onlineCheck))

    print("\nCheck (online + polarized):")
    display(polarize(Transpose, onlineCheck))

    if expr.is_Trace:
        diffCheck = diff(expr, byVar)

        print("\nResult (sympy diff()):")
        display(diffCheck)
        
        print("\nResult (sympy diff() + polarized)")
        display(polarize(Transpose, diffCheck))


    assert equal(res, groupedCheck)
    assert equal(groupedCheck, onlineCheck)
    
    if expr.is_Trace:
        assert equal(res, diffCheck)
        assert equal(groupedCheck, diffCheck)



def testSimplifyAlgo_GroupCombineAdds(expr, check: MatrixExpr, byType: MatrixType):
    res = group(byType = byType, expr = expr, combineAdds = True)

    showGroup([
        expr, res, check
    ])

    assert equal(expr, res)
    assert res.doit() == check.doit()
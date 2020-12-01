## SOURCE for this code = https://github.com/mshvartsman/symbolic-mat-diff/blob/master/symbdiff/simplifications.py

import itertools 

from sympy import Trace, Transpose, Inverse, Function, Derivative, MatMul, preorder_traversal, MatAdd, Add
from collections import OrderedDict


### CONDITIONS AND REPLACEMENTS ----------------------------------


def transpose_traces_cond(dX):
    def cond(x):
        return x.is_Trace and x.arg.is_MatMul and x.has(dX.T)
    return cond


def transpose_traces_repl(dX):
    return lambda x: Trace(x.arg.T)


def trace_sum_distribute_cond(dX):
    return lambda x: x.is_Trace and x.arg.is_MatAdd

# NOTE: this probably assumes each of the matrix 'args' are Square Matrices (else putting them as arg to Trace would NOT be allowed)
def trace_sum_distribute_repl(dX):
    return lambda x: Add(*[Trace(A) for A in x.arg.args])


def matmul_distribute_cond(dX):
    return lambda x: x.is_MatMul and x.has(MatAdd)


def matmul_distribute_repl(dX):
    def repl(x):
        pre, post = [], []
        sawAdd = False
        for arg in x.args:
            if arg.is_MatAdd:
                sawAdd = True
                add = arg
                continue
            if not sawAdd:
                pre.append(arg)
            else:
                post.append(arg)
        # ugly hack here because I can't figure out how to not end up
        # with nested parens that break other things
        addends = [[*addend.args] if addend.is_MatMul else [addend] for addend in add.args]
        return MatAdd(*[MatMul(*[*pre, *addend, *post]) for addend in addends])
    return repl


def inverse_transpose_cond(dX):
    return lambda x: x.is_Transpose and x.arg.is_Inverse and x.arg.is_Symmetric


def inverse_transpose_repl(dX):
    return lambda x: x.arg




def _cyclic_permute(expr):
    
    #if expr.is_Trace and expr.arg.is_MatMul:
    if expr.is_MatMul:
        prods = expr.args 
        newprods = [prods[-1], *prods[:-1]]
        
        return MatMul(*newprods)
    
def _cyclic_permute_constructor(expr):
    '''
    Expects a matmul argument inside either Inverse, Trace, Transpose, and it will permute the matmul arg elements. 
    '''
    
    if (expr.is_Trace or expr.is_Inverse or expr.is_Transpose) and expr.arg.is_MatMul: # need .arg before checking matmul
        prods = expr.arg.args 
        newprods = [prods[-1], *prods[:-1]]

        # Now must painstakingly control for each constructor
        # TODO is there a better way to do this? To get the constructor dynamically at runtime and then wrap the result inside that variable constructor at runtime?
        # TODO think so, see here: https://hyp.is/j9FwwDM-EeuhzIcSn7kZhg/docs.sympy.org/latest/tutorial/manipulation.html
        if expr.is_Trace:
            return Trace(MatMul(*newprods))
        elif expr.is_Inverse:
            return Inverse(MatMul(*newprods))
        elif expr.is_Transpose:
            return Transpose(MatMul(*newprods))
        ## TODO do I need this?
        #elif expr.is_Function:
        #    return Function(MatMul(*newprods), commutative=True)
        ## TODO do I need this?
        #elif expr.is_Derivative:
        #    return Derivative(MatMul(*newprods))
    

def cyclic_permute_dX_cond(dX):
    def cond(x):
        '''
        Expects a matmul on which it can verify if it contains dX and it dX is not last. 
        '''
        if x.is_MatMul: #  if not trace / inverse ... etc can avoid using .arg before .args
            return x.has(dX) and x.args[-1] != dX 
        # Else assume x is_Derivative, is_Inverse, is_Trace, is_Transpose, is_Dummy, is_Function , ... anything that is a constructor and thus requires calling .arg before .args
        #return x.has(dX) and x.arg.args[-1] != dX 

    return cond


def cyclic_permute_constructor_dX_cond(dX):
    def cond(x):
        '''
        Expects a matmul inside a constructor
        '''
        if (x.is_Trace or x.is_Inverse or x.is_Transpose) and x.arg.is_MatMul:
            return x.has(dX) and x.arg.args[-1] != dX 
    
    return cond 


def cyclic_permute_dX_repl(dX):
    def repl(x):
        '''
        Expects a matmul argument, not surrounded in any constructor like Trace
        '''
        newx = x
        nperm = 0

        while newx.args[-1] != dX:
            newx = _cyclic_permute(newx)
            nperm = nperm + 1
            if nperm > len(newx.args):
                raise RuntimeError("Cyclic permutation failed to move dX to end!")
        return newx
    
    return repl


def cyclic_permute_constructor_dX_repl(dX):
    def repl(x):
        '''
        Expects a constructor with matmul inside
        '''
        newx = x
        nperm = 0

        while newx.arg.args[-1] != dX:
            #newx = _cyclic_permute(newx)
            newx = _cyclic_permute_constructor(newx)
            nperm = nperm + 1
            if nperm > len(newx.arg.args):
                raise RuntimeError("Cyclic permutation failed to move dX to end!")
        return newx
    
    return repl



### RULE MANAGEMENT ----------------------------------------------

def _conditional_replace(expr, condition, replacement):
    for x in preorder_traversal(expr):
        try:
            if condition(x):
                expr = expr.xreplace({x: replacement(x)})
        except AttributeError:  # scalar ops like Add won't have is_Trace
            # TODO remove this print statement after fixing the cyclic permute function
            print("got attribute error")
            pass
    return expr
    



def simplify_matdiff(expr, dX):
    for cond, repl in rules.items():
        expr = _conditional_replace(expr, cond(dX), repl(dX))
    return expr







### RULES -------------------------------------------------------


# rules applied in order, this should hopefully work to simplify
rules = OrderedDict([(matmul_distribute_cond, matmul_distribute_repl),
                    (trace_sum_distribute_cond, trace_sum_distribute_repl),
                    (transpose_traces_cond, transpose_traces_repl),
                    (inverse_transpose_cond, inverse_transpose_repl),
                    (cyclic_permute_dX_cond, cyclic_permute_dX_repl),
                    (cyclic_permute_constructor_dX_cond, cyclic_permute_constructor_dX_repl)])


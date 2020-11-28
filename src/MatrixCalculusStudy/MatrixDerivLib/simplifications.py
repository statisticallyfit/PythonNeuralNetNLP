## SOURCE for this code = https://github.com/mshvartsman/symbolic-mat-diff/blob/master/symbdiff/simplifications.py


from sympy import Trace, MatMul, preorder_traversal, MatAdd, Add
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
    # TODO changing 
    # if expr.is_Trace and expr.arg.is_MatMul:
    if expr.arg.is_MatMul:
        prods = expr.arg.args
        newprods = [prods[-1], *prods[:-1]]
        #return Trace(MatMul(*newprods))
        return MatMul(*newprods)
    else:
        print(expr)
        raise RuntimeError("Only know how to cyclic permute products inside traces!")


def cyclic_permute_dX_cond(dX):
    def cond(x):
        #return x.is_Trace and x.has(dX) and x.arg.args[-1] != dX
        # TODO changing
        # TODO changed x.has(dX) into dX.has(x)
        return dX.has(x) and x.arg.args[-1] != dX # TODO need to remove the x.arg.args[-1] and say instead x.args[-1] because the extra .arg accessor means x is assumed Trace() obj. 
    return cond


def cyclic_permute_dX_repl(dX):
    # TODO trying to fix this since it doesn't return the right result: cyclic permute doesn't act on correct thing. 
    def repl(x):
        
    '''def repl(x):
        newx = x
        nperm = 0
        while newx.arg.args[-1] != dX:
            newx = _cyclic_permute(newx)
            nperm = nperm + 1
            if nperm > len(newx.arg.args):
                raise RuntimeError("Cyclic permutation failed to move dX to end!")
        return newx
    '''
    return repl




### RULE MANAGEMENT ----------------------------------------------

def _conditional_replace(expr, condition, replacement):
    for x in preorder_traversal(expr):
        try:
            if condition(x):
                expr = expr.xreplace({x: replacement(x)})
        except AttributeError:  # scalar ops like Add won't have is_Trace
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
                    (cyclic_permute_dX_cond, cyclic_permute_dX_repl)])


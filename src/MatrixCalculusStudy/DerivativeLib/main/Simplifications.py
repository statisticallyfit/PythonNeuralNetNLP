

import numpy as np
from numpy import ndarray

from typing import *
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
from sympy.matrices.expressions.matpow import MatPow 
from sympy.core.mul import Mul 
from sympy.core.add import Add
from sympy.core.power import Pow

from sympy.core.singleton import Singleton 
from sympy.core.numbers import NegativeOne, Number, Integer, One, IntegerConstant

from sympy.core.assumptions import ManagedProperties

from sympy import UnevaluatedExpr , parse_expr 

# Types
import inspect # for isclass() function 
import collections  # for namedtuple


# Path settings
import sys
import os

PATH: str = '/'

UTIL_DISPLAY_PATH: str = PATH + "/src/utils/GeneralUtil/"

MATDIFF_PATH: str = PATH + "/src/MatrixCalculusStudy/DifferentialLib"


sys.path.append(PATH)
sys.path.append(UTIL_DISPLAY_PATH)
sys.path.append(MATDIFF_PATH)




from src.utils.GeneralUtil import *

# For displaying
from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax')



# -------------------------------------------------------------

# Need to create an instance of this each time you have a matpow and want to store the exponent in the same list as transpose and inverse constructors, so that this looks like any ordinary transpose / inverse constructor. 

# TODO used to have name "MatPow" but changed it to have same name as in code, PowHolder --- should I c hange back to MatPow so the PowHolder will be "read" in output as a "MatPow"??
PowHolder = collections.namedtuple("PowHolder", ["expo"])


#MatrixType = ManagedProperties
# ConstrType is either Transpose / Inverse (of type ManagedProperties) OR is a MatPow (which is of type collections.namedtuple)
MatrixType = ManagedProperties

# TODO: whenever I create a new named tuple I would have to include it in this union ... any other way to get it dynamically included after creation?
ConstrType = Union[MatrixType, PowHolder]



INV_TRANS_LIST: List[MatrixType] = [Transpose, Inverse] # this always will include just these two constructors, necessary since only the matrix rules of inverting matrix args when warpping applies to these.


# Dictionary to map between sympy constructors and my own named tuple constructors: 
# TODO need to add here the pairs every time you create a named tuple to match the sympy constructor. 
NAMED_TUP_TO_CONSTR: Dict[ConstrType, ConstrType] = {MatPow : PowHolder, Pow : PowHolder}


# Named tuple list of items (to be able to compare at the type-level)
NAMED_TUPLE_CONSTR_LIST: List[ConstrType] = list(set(NAMED_TUP_TO_CONSTR.values()))
#[PowHolder] 

# Need to include list of constructors that you dig out of.
# TODO: need Trace / Derivative / Function ...??
CONSTR_LIST: List[ConstrType] = INV_TRANS_LIST + list(set(itertools.chain(*NAMED_TUP_TO_CONSTR.items())))
#[Transpose, Inverse, MatPow, PowHolder]



OP_ADD_MUL_LIST : List[MatrixType] = [MatMul, Mul, MatAdd, Add]
OP_POW_LIST: List[ConstrType] = [MatPow, Pow, PowHolder]
SYM_LIST: List[MatrixType] = [MatrixSymbol, Symbol]
NUM_LIST: List[MatrixType] = [NegativeOne, Number, Integer, IntegerConstant, Singleton] # NOTE: SIngleton to encompoass types like sympy.core.numbers.One 
NON_POW_LIST: List[MatrixType] = OP_ADD_MUL_LIST + SYM_LIST + NUM_LIST + INV_TRANS_LIST 

# TODO to add Function, Derivative, and Trace ? any others?
ALL_TYPES_LIST = INV_TRANS_LIST + OP_ADD_MUL_LIST + OP_POW_LIST + SYM_LIST + NUM_LIST 


def typeName(instOrType: ConstrType) -> str:

    # If the arg is a CLASS, must return its name
    if inspect.isclass(instOrType):
        return instOrType.__name__ 

    # Else it is an INSTANCE and I want to show its arguments too:
    return "{}".format(instOrType) 


names = lambda ts: list(map(lambda t: typeName(t), ts))
namess = lambda tts: list(map(lambda lst: names(lst), tts))




hasType = lambda Type, expr : Type.__name__ in srepr(expr)
#hasTranspose = lambda e : "Transpose" in srepr(e)

isMul = lambda e: isMulC(e.func)
isMulC = lambda t: t in [MatMul, Mul]
isAdd = lambda e: isAddC(e.func)
isAddC = lambda t: t in [MatAdd, Add]
isPow = lambda e: isPowC(e.func)
isPowC = lambda t : anyTypeEqual([t], OP_POW_LIST) or anyTypeInstance([t], OP_POW_LIST)


isNum = lambda e: isNumC(e.func)

# Expects t (constructor / type / class)
isNumC = lambda t: isOfType(t, NUM_LIST)

isSymOrNum = lambda e : isSymOrNumC(e.func)
isSymOrNumC = lambda t : (t in SYM_LIST) or isNumC(t) 

isSym = lambda m: len(m.free_symbols) in [0,1]


isOfType = lambda t, ts: anyTypeInstance([t], ts) or anyTypeSub([t], ts) or anyTypeEqual([t], ts)



# GOAL: test equality using INSTANCE_type-equality (like MatPow(expo = 5) == PowHolder) AND type-to-type equality (like MatAdd == MatAdd)
isEq = lambda c, t: anyTypeEqual([c], [t]) or anyTypeInstance([c], [t])
# OLD BAD VERSION
# isEq = lambda c, t: True if isEqMatPow(c, t) else (anyTypeEqual([c], [t]) or anyTypeInstance([c], [t]) )


isEqLists = lambda cs, ts : anyTypeEqual(cs, ts) or anyTypeInstance(cs, ts)
# OLD BAD VERSION : 
# isEqLists = lambda cs, ts: any(list(itertools.chain(*list(map(lambda t: list(map(lambda c: isEqMatPow(c, t), cs)), ts ))))) if (any(map(lambda c: isPowC(c), cs)) or any(map(lambda t: isPowC(t), ts))) else (anyTypeEqual(cs, ts) or anyTypeInstance(cs, ts))

isIn = lambda c, ts: any(map(lambda t : isEq(c, t), ts))


# GOAL: testing simplest form of object equality (between classes or objects)
# Using `equalityOverClassAndNamedtup` to be able to compare MatPow with PowHolder (should be same TYPE)
anyTypeEqual = lambda testTypes, searchTypes : any( list(itertools.chain(*map(lambda tester: list(map(lambda searcher: (tester == searcher) or equalityOverClassAndNamedtup(tester, searcher), searchTypes)), testTypes)) ) )
#any([True for tpe in testTypes if tpe in searchTypes])


# GOAL: test just if the given items in `testTypes` are instances of the items in `searchTypes`:
anyTypeInstance = lambda testTypes, searchTypes : any(list(itertools.chain(*map(lambda tester : list(map(lambda searcher: equalityOverClassAndInstance(tester, searcher) , searchTypes)), testTypes ))))

# OLD BAD VERSION 
# # anyTypeInstance = lambda testTypes, searchTypes : any( list(itertools.chain(*map(lambda tester: list(map(lambda searcher: isEqMatPow(tester, searcher) if not inspect.isclass(searcher) else isinstance(tester, searcher), searchTypes)), testTypes)) ) )

# GOAL: testing, given two classes, if one is subtype of the other (all combos from the lists)
anyTypeSub = lambda testTypes, searchTypes : any(list(itertools.chain(*map(lambda tester : list(map(lambda searcher: inspect.isclass(tester) and inspect.isclass(searcher) and issubclass(tester, searcher), searchTypes)), testTypes ))))

# OLD BAD VERSION
#anyTypeSub = lambda testTypes, searchTypes : any( list(itertools.chain(*map(lambda tester: list(map(lambda searcher: isEqMatPow(tester, searcher) if not (inspect.isclass(tester) and inspect.isclass(searcher)) else issubclass(tester, searcher), searchTypes)), testTypes)) ) )

# NOTE: len == 0 of free syms when Number else for MatSym len == 1 so put 0 case just for safety.
#onlySymComponents_AddMul = lambda expr: (expr.func in [MatAdd, MatMul]) and all(map(lambda expr: len(expr.free_symbols) in [0, 1], expr.args))

isSimpleArgs = lambda e: all(map(lambda a: len(a.free_symbols) in [0, 1], e.args))

isInnerExpr = lambda e: (e.func in OP_ADD_MUL_LIST) and isSimpleArgs(e)

# GOAL: make equality between named tuple and original sympy type
# Example 1: MatPow, PowHolder ---> TRUE 
# Example 2: PowHolder, MatPow ---> TRUE (symmetric)
# Example 3: MatAdd, PowHolder ---> FALSE
# Example 3: PowHolder, PowHolder(expo = 4) ---> FALSE (because I want this function to JUST test equality at type-level. This example would pass for the `equalityOverClassAndInstance` function)
def equalityOverClassAndNamedtup(x, y) -> bool: 
    return ((x, y) in NAMED_TUP_TO_CONSTR.items()) or ((y, x) in NAMED_TUP_TO_CONSTR.items())

# GOAL: test whether the args are equal in an instance-related way: 
# Example 1: MatPow(expo = 111) , PowHolder ---> TRUE (since first is instance of type PowHolder. NOTE: IS a symmetric definition)
# Example 2: PowHolder , PowHolder ---> False (since both are classes not instances)
# Example 3: MatPow(expo = 1), MatPow(expo = 2) ---> False (since both are instances, not of each other, but of another class)
# Example 4: MatPow(expo = 2), MatPow(expo = 2) ---> TRUE
def equalityOverClassAndInstance(x, y) -> bool: 
    if equalityOverClassAndNamedtup(x, y):
        return False # since the 'cross-instance' func tests somehing else, don't want to confuse the two jobs. 

    # NOW can continue the main logic of the function:
    # TODO STAR left off here CHECK if this is right (want just instance-instance and instance-class equality tester here)
    def analyzeEquality(x, y) -> bool:
        instOfNamedTup: bool = isinstance(y, x) 
        value = NAMED_TUP_TO_CONSTR.get(x, False) 
        instOfConstr: bool = isinstance(y, value) if value != False else False 

        return instOfNamedTup or instOfConstr # isnstance of PowHolder or instance of MatPow

    if inspect.isclass(x):
        return analyzeEquality(x, y) # class x needs to be second arg

    elif inspect.isclass(y):
        return analyzeEquality(y, x) # class y needs to be second arg
    else: #OLD: else if NIETHER x, y are classes then cannot use the isintance method, doesn't make sense since neither x,y is a class
        # If neither args are classes, then they are instances so test equality of the fields: 
        return x == y # case 4: MatPow(expo = 1) == MatPow(expo = 2) ??

'''
def addToOpPows(ts: List[ConstrType]) -> List[ConstrType]:
    # Must filter to not allow non-pow types to get added to the pow list (but must do this while not using isPow because that would cause recursion)
    onlyPowTs: List[ConstrType] = list(filter(lambda t: 
        not (any(map(lambda np: isEq(t, np), NON_POW_LIST))), 
    ts))

    return list(set(OP_POW_LIST + onlyPowTs))
'''

# Need separate definition from anyTypeInst ... etc because otherwise will get recursion error. 
'''
def isEqMatPow(c: ConstrType, t: ConstrType) -> bool:
    opPowList: List[ConstrType] = addToOpPows([c, t])
    return ((c in opPowList) and (t in opPowList))
'''


def getConstr(expr: MatrixExpr) -> ConstrType:
    i: int  = findWhere(ALL_TYPES_LIST, type(expr))[0]
    # Getting the name from the list
    return ALL_TYPES_LIST[i]









def pickOut(WrapType: MatrixType, expr: MatrixExpr):
    '''
    GOAL: extract the superficial-level-inner argument inside the nesting if it contains the WrapType on the outer, else add the WrapType over it so it will cancel out later on. 


    ---> NOTE: ok to add extra transpose layer since functions later on will apply / cancel out transpose accordingly
    ---> NOTE: inverse does not cancel out, so need to remove it so later functions act as if it weren't there, so they can add it back later
        TODO: Check this
    ---> NOTE: power cancels out, so same action as transpose (add negative power)


    *** EXAMPLE: Transpose case (transpose on outer)
    INPUT: pickOut(Transpose, ((J + X*A)^4)^T)
    RESULT: (J + X*A)^4

    *** EXAMPLE: Transpose case (no outer transpose or none at all)
    INPUT: pickOut(Transpose, ((J + X*A)^T)^8 )
    RESULT: (((J + X*A)^T)^8)^T

    (???) EXAMPLE: Inverse case (inverse on outer)
    INPUT: pickOut(Inverse, (A + B)^-1 )
    RESULT: A + B

    (???) EXAMPLE: Inverse case (inverse not on outer or not there at all)
    INPUT: pickOut(Inverse, A)
    RESULT: A^-1

    (???) INPUT: pickOut(Inverse, ((A + B)^-1)^4 )
    RESULT: ((A + B)^-1)^4

    *** EXAMPLE: Power case (power on outer side)
    INPUT: pickOut( PowHolder(expo = 5), (J + X*A)^4 )
    RESULT: J + X*A

    *** EXAMPLE: Power case (no power on outer side, while is there nested)
    INPUT: pickOut( PowHolder(expo = 4), (((J + X*A)^4)^T)^5 )
    RESULT: ((((J + X*A)^4)^T)^5)^ (-4)

    *** EXAMPLE: Power case (no power on outer side)
    INPUT: pickOut(PowHolder(expo = 4), J + X*A)
    RESULT: (J + X*A)^ (-4)
    '''

    # NOTE: this function acts only on Inverse, Transpose, or Power kind of constructor. 
    if not isIn(WrapType, CONSTR_LIST):
        return expr 
    if isPowC(WrapType) and (not typeName(WrapType) == 'PowHolder'):
        return expr 


    # TODO FIX POWER CASE: should I return here to the negative power?
    #if (not hasType(WrapType, expr)) and isPowC(WrapType):
        #return expr
        # NOTE: it is ok for pickOut() to return original arg when arg doesn't have WrapType in it because WrapType is passed separately as a constructor, this is not like in other simplification functions wher eyou need to recurse through nested expressions. 
    #elif not hasType(WrapType, expr):
        #return WrapType(expr) # TODO check here if this passes, hacked to separate the above mat-pow case from the rest of the non-matpow cases. 

    # Inverting the power with the inner WrapType (just at superficial level, not recursing here):
    if isPow(expr) and isEq( getConstr(expr.args[0]) , WrapType ):
        (base, expo) = expr.args 
        return WrapType(MatPow(base.arg, expo))

    elif isEq(getConstr(expr), WrapType):
        return expr.arg # can use .arg now since the MatPow (.args) case is out of the way, above

    #return WrapType(expr) #TODO check

    # Transpose case:
    if isEq(WrapType, Transpose) and isEq(getConstr(expr), Transpose):








def composeTwoMatrixOps(f, g):
    # NOTE: assumes PowHolder field name is 'expo'
    if isPowC(f) and isPowC(g):
        return lambda *a, **kw: MatPow(MatPow(*a,  g.expo), f.expo)
    elif isPowC(f):
        return lambda *a, **kw: MatPow(g(*a, **kw), f.expo)
    elif isPowC(g):
        return lambda *a, **kw: f(MatPow(*a, g.expo))
    return lambda *a, **kw: f(g(*a, **kw))

def composeSinglePowMatrixOp(g):
    # NOTE: assumes PowHolder field name is 'expo'
    if isPowC(g):
        return lambda *a, **kw: MatPow(*a,  g.expo)
    # else if not a pow / powholder / matpow ...
    return lambda *a, **kw: g(*a, **kw)

def composeMatrixOps(*fs):
    # NOTE: when the only fs is a single PowHolder then TypeError 'PowHolder' object is not callable gets returned so must separate the cases for when getting this single argument in the list
    if len(fs) == 1 and isPowC(*fs):
        # NOTE: using * on the tuple in order to collapse the tuple to pass as single arg, else error occurs: TypeError 'tuple' object not callable
        return composeSinglePowMatrixOp(*fs)
        
    return reduce(composeTwoMatrixOps, fs)


def applyTypesToExpr( pairTypeExpr: Tuple[List[MatrixType], MatrixExpr]) -> MatrixExpr:
    '''Ignores any types that are MatrixSymbol or Symbol or instance of Number because those give error when applied to an expr.'''

    (typeList, expr) = pairTypeExpr

    typeList = list(filter(lambda tpe: not isSymOrNumC(tpe), typeList))

    if typeList == []:
        return expr
    return composeMatrixOps(*typeList)(expr)





def stackTypesBy(byType: ConstrType, types: List[ConstrType]) -> List[ConstrType]:
    '''Given expr type (like Transpose) and given expr list, this function pulls all the signal types to the end of the
    list, leaving the non-signal-types as the front'''

    # Get number of signal types in the list
    signalTypes: List[ConstrType] = list(filter(lambda t : isEq(t, byType), types))
    # Even if we pass in byType = MatPow, and the types list contains PowHolder type instance then we must replicate the PowHolder not the MatPow (signal type)

    # Get the list without the signal types
    nonSignalTypes: List[ConstrType] = list(filter(lambda t : not isEq(t, byType), types))

    #countTypes: int = len(signalTypes)


    # Create the signal types that go at the end
    #endTypes: List[ConstrType] = [byType] * countTypes

    return signalTypes + nonSignalTypes # + endTypes




def chunkTypesBy(byTypes: List[MatrixType], types: List[ConstrType]) -> List[List[ConstrType]]:
    '''Separates the `types` in expr list of list of types by the types in `byConstrs` and keeps other types separate too, just as how they appear in the original list'''

    # Not strictly necessary here, just useful if you want to see shorter version of names below
    #getSimpleTypeName = lambda t : str(t).split("'")[1].split(".")[-1]
    byConstrs: List[MatrixType] = list(set(byTypes))



    # Step 1: coding up in pairs for easy identification: need types from CONSTR_LIST to be tagged as the same number (0) and all other types to be tagged as (1). 
    codeConstrPairs: List[Tuple[int, ConstrType]] = list(map(lambda C : (0, C) if isIn(C, byConstrs) else (1, C), types))

    # Step 2: getting the groups
    chunkedPairs: List[List[Tuple[int, ConstrType]]] = [list(group) for key, group in itertools.groupby(codeConstrPairs, operator.itemgetter(0))]

    # Step 3: getting only the types in the chunked lists
    chunkedConstrs: List[List[ConstrType]] = list(map(lambda lst : list(map(lambda pair : pair[1], lst)), chunkedPairs))

    return chunkedConstrs




def chunkExprsBy(byTypes: List[MatrixType], expr: MatrixExpr) -> Tuple[List[List[ConstrType]], List[List[MatrixExpr]]]:
    '''Given an expression, returns the types and expressions as tuples, listed in preorder traversal, and separated by which expressions are grouped by Transpose or Inverse constructors (in their layering)'''

    byConstrs: List[MatrixType] = list(set(byTypes))

    ps: List[MatrixExpr] = list(preorder_traversal(expr)) # elements broken down
    # NOTE: if p is a matpow then must store its exponent in the named tuple constructor. 
    # NOTE: isPow also checks that `p` is an INSTANCE of PowHolder, so it is capable of extracting it then too (though don't think this function `chunkExprsBy` has that kind of argument, since that extraction happens later down the pipeline)
    cs: List[ConstrType] = list(map(lambda p: PowHolder(expo = p.args[1]) if isPow(p) else type(p), ps)) # types / constructors


    # Check first: does the expr have the types in byConstrs? If not, then return it out, nothing to do here:
    #if not (Transpose in cs):
    #BAD tests for AND all types: if not (set(byConstrs).intersection(cs) == set(byConstrs)):
    # GOOD, tests for OR all types (since should be able to chunk [T, T, T] then [I, I] and [I, T, I, T] so need the OR relationship):
    if not anyTypeEqual(testTypes = CONSTR_LIST, searchTypes = cs):
        return ([cs], [ps])

    csChunked: List[List[ConstrType]] = chunkTypesBy(byTypes = byConstrs, types = cs)


    # Get the lengths of each chunk
    chunkLens: List[int] = list(map(lambda cLst : len(cLst), csChunked))


    # Use lengths to segregate the preorder traversal exprs also, then later to apply the transformations
    psChunked: List[List[MatrixExpr]] = []
    rest: List[MatrixExpr] = ps

    for size in chunkLens:
        (fst, rest) = (rest[:size], rest[size: ])
        psChunked.append( fst )

    return (csChunked, psChunked)







### WARNING: not maintained anymore 

def group(WrapType: MatrixType, expr: MatrixExpr, combineAdds:bool = False) -> MatrixExpr:
    '''Combines transposes when they are the outermost operations.

    Brings the individual transpose operations out over the entire group (factors out the transpose and puts it as the outer operation).
    NOTE: This happens only when the transpose ops are at the same level. If they are nested, that "bringing out" task is left to the rippletranspose function.

    combineAdds:bool  = if True then the function will group all the addition components under expr transpose, like in matmul, otherwise it will group only the individual components of the MatAdd under transpose.'''


    def revAlgo(WrapType: MatrixType, expr: MatrixExpr):
        '''Converts B.T * A.T --> (A * B).T'''

        if hasType(WrapType, expr) and isMul(expr):

            revs = list(map(lambda a : pickOut(WrapType, a), reversed(expr.args)))

            #return Transpose(MatMul(*revs))
            return WrapType(MatMul(*revs))

        return expr


    def algo_Group_MatMul_or_MatSym(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

        ps = list(preorder_traversal(expr))
        #ms = list(filter(lambda p : p.is_MatMul, ps))
        ms = list(filter(lambda p : isMul(p), ps))
        ts = list(map(lambda m : revAlgo(WrapType, m), ms))

        ns = []
        for j in range(0, len(ms)):
            m = ms[j]
            if expr.has(m):
                #newPair = [(m, ts[j])] #revTransposeAlgo(m))]
                #ds[expr] = (ds.get(expr) + newPair) if expr in ds.keys() else newPair
                ns.append( (m, ts[j]) )

        # Apply the changes in the list of replacements we gathered above
        exprToChange = expr
        for (old, new) in ns:
            exprToChange = exprToChange.xreplace({old : new})

        return exprToChange

    # Group function here -----------------------------------

    #if not expr.is_MatAdd: # TODO must change this to identify matmul or matsym exactly
    if not isAdd(expr):
        return algo_Group_MatMul_or_MatSym(WrapType, expr)


    Constr = expr.func

    # TODO fix this to handle any non-add operation upon function entry
    addendsTransp: List[MatrixExpr] = list(map(lambda a: group(WrapType, a), expr.args))

    if combineAdds:
        innerAddends = list(map(lambda t: pickOut(WrapType, t), addendsTransp))
        #return Transpose(MatAdd(*innerAddends))
        return WrapType (Constr(*innerAddends))

    # Else not combining adds, just grouping transposes in addends individually
    # TODO fix this to handle any non-add operation upon function entry. May not be a MatAdd, may be Trace for instance.

    #return MatAdd(*addendsTransp)
    return Constr(*addendsTransp)







# TODO: FIX: this doesn't work to ripple out the innter transpose when in matpow: 
#eout = MatAdd(E, Transpose(MatMul(
#    Transpose(R*J*A.T*B), 
#    MatPow(Transpose(J + X*A), 4)
#)))

def rippleOut(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

    '''Brings transposes to the outermost level when in expr nested expression. Leaves the nested expressions in their same structure.
    Because it preserves the nesting structure, it is expr kind of shallow polarize() function'''


    def algo_RippleOut_MatMul_or_MatSym(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:
        '''For each layered (nested) expression where transpose is the inner operation, this function brings transposes to be the outer operations, leaving all other operations in between in the same order.'''

        assert WrapType in CONSTR_LIST

        (csChunked, psChunked) = chunkExprsBy(byTypes = CONSTR_LIST, expr = expr)

        # Order the types properly now for each chunk: make transposes go last in each chunk:
        stackedChunks = list(map(lambda lst : stackTypesBy(byType = WrapType, types = lst), csChunked))

        # Pair up the correct order of transpose types with the expressions
        # BEFORE: (Transpose in tsPs[0]) or (Inverse in tsPs[0])
        typeListExprListPair = list(filter(lambda tsPs : anyTypeEqual(testTypes = CONSTR_LIST, searchTypes = tsPs[0]),
                                           list(zip(stackedChunks, psChunked))))



        #typeListExprPair = list(map(lambda tsPs : (tsPs[0], tsPs[1][0]), typeListExprListPair))

        # Get the first expression only, since it is the most layered, don't use the entire expression list of the
        # tuple's second part. And get the inner argument (lay bare) in preparation for appling the correct order of
        # `byType`s.
        typeListInnerExprPair = list(map(lambda tsPs : (tsPs[0], inner(tsPs[1][0])), typeListExprListPair))

        outs = list(map(lambda tsExprPair : applyTypesToExpr(tsExprPair), typeListInnerExprPair ))

        # Get the original expressions as they were before applying correct transpose
        ins = list(map(lambda tsPs : tsPs[1][0], typeListExprListPair))

        # Filter: get just the matmul-type arguments (meaning not the D^T or E^-1 type arguments) from the result list (assuming there are other MatMul exprs). Could have done this when first filtering the psChunked, but easier to do it now.
        # NOTE: when there are ONLY mat syms and no other matmuls then we must keep them since it means the expression is layered with only expr matsum as the innermost expression, rather than expr matmul.
        #isSymOrNum = lambda expr : expr.is_Symbol or expr.is_Number
        isSimpleSymOrNum = lambda expr: expr.is_Symbol or isNum(expr)
        #isSym = lambda expr  : len(expr.free_symbols) == 1

        # Flattening the chunked ps list for easier evaluation:
        ps: List[MatrixExpr] = list(itertools.chain(*psChunked))
        allSymOrNums = len(ps) == len(list(filter(lambda e: isSimpleSymOrNum(e), ps)) )
        #allSyms = all(map(lambda expr : isSym(expr), ps))

        #if not allSymOrNums:
        outs = list(filter(lambda expr : not isSimpleSymOrNum(expr), outs))

        ins = list(filter(lambda expr : not isSimpleSymOrNum(expr), ins))
        # else just leave the syms as they are.


        # Zip the non-transp-out exprs with the transp-out expressions as list (NOTE: cannot be dictionary since we need to keep the same order of expressions as obtained from the preorder traversal, else substitution order will be messed up)
        outInPairs = list(zip(outs, ins))

        # Now must apply from the beginning to end, each of the expressions. Must replace in the first expr, all of the latter expressions, kind of like folding operation of Matryoshka dolls, to preserve all the end changes: C -> goes into -> B -> goes into -> A
        accFirst = expr #outs[0]

        f = lambda acc, outInPair: acc.xreplace({outInPair[1] : outInPair[0]})

        resultWithTypeOut = foldLeft(f , accFirst,  outInPairs) # outsNotsPairs[1:])

        return resultWithTypeOut



    #if not expr.is_MatAdd: # TODO must change this to identify matmul or matsym exactly
    if not isAdd(expr):
        return algo_RippleOut_MatMul_or_MatSym(WrapType = WrapType, expr = expr)

    Constr: MatrixType = expr.func

    componentsOut: List[MatrixExpr] = list(map(lambda a: rippleOut(WrapType = WrapType, expr = a), expr.args))

    return Constr(*componentsOut)



# TODO this function cannot identify MatPow that is same as Inverse: for instance the obj D.I.T is MatPow of Transpose (not Inverse as expected) while using Transpose(Inverse(D)) is Transpose of Inverse, as expected.
# ---> So must find way for this function to recognize MatPow with NegativeOne and convert those into Inverse.
# ---> Need to find out if the double NegativeOne come from one set of MatPow inverse or not. Then replace accordingly with the Inverse constructor.
# TODO for now just avoid passing in D.I.T and just use the verbose constructor names.
def digger(expr: MatrixExpr) -> List[Tuple[List[MatrixType], MatrixExpr]]:
    '''Gets list of tuples, where each tuple contains expr list of types that surround the inner argument in the given matrix expression.

    EXAMPLE:

    Input: (((B*A*R)^-1)^T)^T

    Output: (ts, inner) where
        ts = [Transpose, Transpose, Inverse]
        inner = MatMul(B, A, R)
    '''

    #(csChunked, psChunked) = chunkExprsBy(byTypes = [Transpose ,Inverse], expr = expr)
    # Using a Constr_list we can add Trace, Derivative etc and any other constructor we wish.
    (csChunked, psChunked) = chunkExprsBy(byTypes = CONSTR_LIST, expr = expr)


    # Pair up the correct order of transpose types with the expressions
    # BEFORE: (Transpose in tsPs[0]) or (Inverse in tsPs[0])
    typeListExprListPair = list(
        filter(lambda tsPs : isEqLists(CONSTR_LIST, tsPs[0]),  
        list(zip(csChunked, psChunked)))
    )

    # Get the first expression only, since it is the most layered, and pair its inner arg with the list of types from that pair.
    typeListInnerPair = list(map(lambda tsPs : (tsPs[0], inner(tsPs[1][0])), typeListExprListPair))

    return typeListInnerPair







def inner(expr: MatrixExpr) -> MatrixExpr:
    '''Gets the innermost expression (past all the .arg) on the first level only'''
    #isMatSym = lambda e : len(e.free_symbols) == 1

    # TODO missing any base case possibilities? Should include here anything that is not Trace / Inverse ... etc or any kind of constructors that houses an inner argument.
    Constr = expr.func
    #types = [MatMul, MatAdd, MatrixSymbol, Symbol, Number]
    otherTypes = list( set(ALL_TYPES_LIST).symmetric_difference(CONSTR_LIST) - set(OP_POW_LIST))
    #isAnySubclass = any(map(lambda t : issubclass(Constr, t), types))

    #if (Constr in otherTypes) or issubclass(Constr, Number):
    if (Constr in otherTypes) or isNumC(Constr): # or (isPow(expr) and isSym(expr)):

        return expr
    
    elif isPow(expr): # and (not isSym(expr)):
        # Recurse through the base of the power expression
        (base, _) = expr.args 
        return inner(base)

    # else keep recursing
    return inner(expr.arg) # need to get arg from Trace or Transpose or Inverse ... among other constructors






def innerTrail(expr: MatrixExpr) -> List[Tuple[List[MatrixType], MatrixExpr]]:
    '''Gets the innermost expr (past all the .arg) on the first level only, and stores also the list of constructors'''

    def doInnerTrail(expr: MatrixExpr, accConstrs: List[MatrixType]) -> Tuple[List[MatrixType], MatrixExpr]:

        # FIRST TASK when entering this function is to record the constructor type 

        Constr: MatrixType = expr.func

        #otherTypes = list( set(ALL_TYPES_LIST).symmetric_difference(CONSTR_LIST) )
        otherTypes = list( set(ALL_TYPES_LIST).symmetric_difference(CONSTR_LIST) - set(OP_POW_LIST))

        # BASE CASE 
        if (Constr in otherTypes) or isNumC(Constr): # or (isPow(expr) and isSym(expr)):

            return (accConstrs, expr)

        # else keep recursing
        elif isPow(expr): # and (not isSym(expr)): 
            # Recurse through the base of the power expression
            (base, exponent) = expr.args 

            return doInnerTrail(expr = base, accConstrs = accConstrs + [PowHolder(expo = exponent)])
        # else is NOT pow nor OTHERTYPE so must be Inverse or Transpose kind of constructor ...
        return doInnerTrail(expr = expr.arg, accConstrs = accConstrs + [Constr])

        ## END doInnerTrail() function

    # OUTER FUNCTION innerTrail()
    return doInnerTrail(expr = expr, accConstrs = [])







def elimPows(powNums: List[int]) -> List[int]:
    if len(powNums) in [0, 1]:
        return powNums
    
    n = len(powNums)
    x = powNums[0]
    y = powNums[n-1] # last elem

    if abs(x) == abs(y): 
        #return [x] + elimPows(powNums[1 : n-1]) + [y]
        # In this case, the ends are canceled out (represents out powers cancel out)
        return elimPows(powNums[1 : n-1])
    
    if abs(x) > abs(y) and x <= 0:
        return [x] + elimPows(powNums[1 : n])
    
    if abs(x) < abs(y) and x <= 0: 
        return elimPows(powNums[0 : n-1]) + [y]
    
    # else no negatives so no powers to cancel, so just return the list
    return powNums 


def factorPows(types: List[ConstrType]) -> List[ConstrType]:
     # Filter to get just the pow types
    matPows: List[ConstrType] = list(filter(lambda t: isEq(t, MatPow), types))
    # Get the non-matpow types
    #nonMatpows: List[MatrixType] = list(filter(lambda t: not isEq(t, MatPow), types))
    # Extract their exponents 
    expos: List[int] = list(map(lambda mp: mp.expo, matPows))
    # Go through this sorted num power list, canceling out opposite-sign pairs (to represent power canceling) otherwise no other simplification occurs
    exposFactored: List[int] = elimPows(sorted(expos))
    # In order to keep the original order of the exponents in the given list, we want to just select the exponents from the given list, using the exponents we have left (from the sorted list)
    expoFactoredPairs = []
    recordExposFactored: List[int] = exposFactored.copy()
    for originalExpo in expos:
        if originalExpo in recordExposFactored: 
            expoFactoredPairs.append( (originalExpo, originalExpo) )
            recordExposFactored.remove(originalExpo)
        else: 
            expoFactoredPairs.append((originalExpo, None))
    # Finish extracting the chosen expoonents
    chosenExpos: List[Tuple[int, int]] = list(filter(lambda tup: tup[1] != None, expoFactoredPairs))

    chosenExpos: List[int] = list(map(lambda tup: tup[0], chosenExpos))

    # Now make them PowHolder again
    matPowsFactored: List[ConstrType] = list(map(lambda e: PowHolder(expo = e), chosenExpos))

    # Make the simplified list of pows go at beginning (aking to transposes that are all on outer layers) with non-matpow types at the inner layers. 
    return matPowsFactored #+ nonMatPows


def factor(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

    typesInners = digger(expr)

    # Check: if empty list returned by digger (say since expr is just MatrixSymbol) then need to exit and return original expression (since there must be nothing to factor).
    if typesInners == []:
        return expr
    # Otherwise error results on next line:


    (types, innerExprs) = list(zip(*typesInners))
    (types, innerExprs) = (list(types), list(innerExprs))

    # Filtering the wrapper types that are `WrapType`s
    noSignalTypes: List[List[ConstrType]] = list(map(lambda typeList:  list(filter(lambda t: not isEq(t, WrapType), typeList)) , types))

    # Pair up all the types, filtered types, and inner expressions for easier querying later:
    triples: List[Tuple[List[MatrixType], List[MatrixType], List[MatrixExpr]]]  = list(zip(types, noSignalTypes, innerExprs))

    # Create new pairs from the filtered and inner Exprs, by attaching expr Transpose at the end if odd num else none.
    newTypesInners = []

    if WrapType in INV_TRANS_LIST: 
        # The inverse / transpose way of factoring out
        newTypesInners: List[Tuple[List[MatrixType], List[MatrixExpr]]] = list(map(lambda triple: ([WrapType] + triple[1], triple[2]) if (triple[0].count(WrapType) % 2 == 1) else (triple[1], triple[2]) , triples))

    # NOTE: using isPowC not isPow because the latter gives error since PowHolder has no attribute 'func'
    elif isPowC(WrapType): # TODO start here to fix factor() to recognize matpows
        
        newTypesInners: List[Tuple[List[ConstrType], List[MatrixExpr]]] = list(map(lambda triple: (factorPows(types = triple[0]) + triple[1], triple[2]) , triples))

    # Create the old exprs from the digger results:
    oldExprs: List[MatrixExpr] = list(map(lambda pair: applyTypesToExpr(pair), typesInners))

    # Create the new expressions with the simplified transposes:
    newExprs: List[MatrixExpr] = list(map(lambda pair: applyTypesToExpr(pair), newTypesInners))

    # Zip the old and new expressions for replacement later on:
    oldNewExprs: List[Dict[MatrixExpr, MatrixExpr]] = list(map(lambda pair: dict([pair]), list(zip(oldExprs, newExprs))))

    # Use fold to accumulate the results by replacing correct, simplified pieces of expressions into the overall expression.
    accFirst = expr
    f = lambda acc, oldNew: acc.xreplace(oldNew)
    result: MatrixExpr = foldLeft(f, accFirst, oldNewExprs)

    return result








def wrapShallow(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:
    '''Wraps expression in Constr Transpose or Inverse at the superficial level, doesn't wrap from the innermost expression outward. Can receive nested exprs or simple exprs: 
    
    Nested expr example: 
        Input: (B*A*R).T * (R*U.T*C.I).T
        Result: (R * U.T * C.I * B * A * R).T
    
    Simple expr example: 
        Input: B.T + A.T + R.T + C.I
        Result: (C.I.T + B + A + R).T
    '''
    # NOTE: A^a * B^b * C^c ... != (ABC)^(a+b+c) because matrix multiplication is not commutative so practically cannot lift out the MatPow types here when they are stuck inside, must leave them alone
    #if isPowC(WrapType):
    #    return expr 

    #assert WrapType in INV_TRANS_LIST

    # TODO: wrapShallow must be able to interpret a MatMul(CONST, expr) kind of input (dig deeper)
    Constr: MatrixType = expr.func
    #assert Constr in [MatAdd, MatMul]



    # Get only the matrixexprs, leaving the numbers / constants aside: 
    nonMatrixArgs = list(filter(lambda a: not isinstance(a, MatrixExpr), expr.args))
    matrixArgs = list(filter(lambda a: isinstance(a, MatrixExpr), expr.args))

    numArgsOfType: int = len(list(filter(lambda a: type(a) == WrapType, matrixArgs )))

    # Building conditions for checking if we need to wrap the expr, or else return it as is.
    mostSymsAreOfType: bool = (numArgsOfType / len(matrixArgs) ) >= 0.5

    # NOTE: len == 0 of free syms when Number else for MatSym len == 1 so put 0 case just for safety.
    #onlySymComponents_AddMul = lambda expr: (expr.func in [MatAdd, MatMul]) and all(map(lambda expr: len(expr.free_symbols) in [0, 1], expr.args))

    # If signal WrapType is a power-kind of type and given expression constructor (Constr) is either Add or Mul kind then we are not assured mathematically that we can bring out the Pow (since it may be like the eout expression, nested deep beyond reach of the Add / Mul)
    mustNotWrapPow: bool =  (isPowC(WrapType) and (Constr in OP_ADD_MUL_LIST) )

    mustWrap: bool = (Constr in OP_ADD_MUL_LIST) and mostSymsAreOfType

    if (not mustWrap) or mustNotWrapPow:
        return expr

    # Else 

    ### WRAPPING ALGO: 

    # Apply the reversing and wrapping alog part: 
    invertedArgs: List[MatrixExpr] = list(map(lambda theArg: pickOut(WrapType, theArg) , matrixArgs)) # TODO BROKEN BEFORE THIS POINT FIX MATPOW. BROKEN HERE because it puts transpose on the -1 factor of matmul expr
    

    #invertedArgs: List[MatrixExpr] = list(reversed(invertedArgs)) if Constr == MatMul else invertedArgs
    # TODO changed to check for Mul also since freeze() changes to Mul from MatMul
    invertedArgs: List[MatrixExpr] = list(reversed(invertedArgs)) if isMulC(Constr) else invertedArgs

    # Next: must wrap the resulting inverted args in the oriignal constructor / wrapper. But there are two kinds: two-arg MatPow and single-arg other kinds (non-pow):
    
    #wrapped: MatrixExpr = WrapType(Constr(*invertedArgs)) if len(invertedArgs) != 1 else WrapType(*invertedArgs) # to avoid double constructor wrapping (like double Matmul wrapping)
    wrapped = None 
    if len(invertedArgs) != 1:
        wrapped = WrapType(Constr(*invertedArgs))
    elif isPowC(WrapType) and WrapType == Constr: # TODO can't return matpow when WrapTYpe is just Pow
        (_, exponent) = expr.args 
        wrapped = MatPow(*(invertedArgs + [exponent]) )
    else: 
        wrapped = WrapType(*invertedArgs)



    if nonMatrixArgs != [] and not (isPowC(Constr)): 
        wrapped: MatrixExpr = Constr(*(nonMatrixArgs + [wrapped]))

    return wrapped




def wrapDeep(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:
    Constr = expr.func

    # NOTE: A^a * B^b * C^c ... != (ABC)^(a+b+c) because matrix multiplication is not commutative so practically cannot lift out the MatPow types here when they are stuck inside, must leave them alone
    #if isPowC(WrapType):
    #    return expr 

    # NOTE: nothing to wrap when expr is a symbol
    if isSym(expr):
        return expr

    elif isInnerExpr(expr):
        wrappedExpr: MatrixExpr = wrapShallow(WrapType = WrapType, expr = expr)
        return wrappedExpr

    elif Constr in (OP_ADD_MUL_LIST + OP_POW_LIST):
    #elif Constr in [MatAdd, MatMul, MatPow]:  #then split the polarizing operation over the arguments since any one of the args can be an inner expr 
        
        # NOTE 1: (verify the case of matmul(const, expr) individually and then go deeper in the expr)
        # NOTE 2: must also factor the result so that the next step of wrapNest works correctly (doesn't contain superfluous transposes)
        wrappedArgs: List[MatrixExpr] = list(map(lambda a: factor(WrapType, wrapDeep(WrapType, a)) if isinstance(a, MatrixExpr) else a, expr.args))
        # if isinstance(a, MatrixExpr) else a

        exprWithPartsWrapped: MatrixExpr = Constr(*wrappedArgs)

        exprOverallWrapped: MatrixExpr = wrapShallow(WrapType = WrapType, expr = exprWithPartsWrapped) # TODO BROKEN HERE

        return exprOverallWrapped

    
    else: # else is Trace, Transpose, or Inverse or any other constructor
        innerExpr = expr.arg

        return Constr( wrapDeep(WrapType = WrapType, expr = innerExpr) )






# Making expr group transpose that goes deep until innermost expression (largest depth) and applies the group algo there
# Drags out even the inner transposes (aggressive grouper)
def _polarize(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:
    '''Given an expression with innermost args nested as components inside another expression, (many nestings), this function tries to drag / pull / force out all the transposes from the groups of expressions of matmuls / matmadds and from individual symbols.

    Tries to create one nesting level that mentions transpose (there can be other nestings with inverse inside for instance but the outermost nesting must be the only one with transpose).

    There must be no layering of transpose in the nested expressions in the result returned by this function -- polarization of transpose to the outer edges.'''


    def algoPolarize(WrapType: MatrixExpr, expr: MatrixExpr) -> MatrixExpr:
        # Need to factor out transposes and ripple them out first before passing to wrap algo because the wrap algo won't reach in and factor or bring out inner transposes, will just overlay on top of them without simplifying (bad, since yields more complicated expression)
        fe = factor(WrapType = WrapType, expr = expr) # no need for ripple out because factor does this implicitly

        wfe = wrapDeep(WrapType = WrapType, expr = fe)

        # Must bring out the extra transposes that are brought out by the wrap algo and then cut out extra ones.
        fwfe = factor(WrapType = WrapType, expr = wfe)

        return fwfe

    def hackpolarizeMatAddMul(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr: 
        #assert expr.is_MatAdd or expr.is_MatMul 
        assert expr.func in OP_ADD_MUL_LIST # since freeze turns MatAdd --> Add and MatMul --> Mul

        polarizedArgs: List[MatrixExpr] = list(map(lambda a: hackpolarizeGroup(WrapType = WrapType, expr = a), expr.args))

        Constr = expr.func 

        return Constr(*polarizedArgs)


    countTopTypes = lambda WrapType, pairs: sum(list(map(lambda pair: True if pair[0][0] == WrapType else False, pairs)) )
    

    def hackpolarizeGroup(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr: 
        # TODO fix if this extra step of factoring and wrapping deep results in more transposes then  polarize again or just leave at the previous factor

        p = algoPolarize(WrapType = WrapType, expr = expr)
        pp = algoPolarize(WrapType = WrapType, expr = p )
        fe = factor(WrapType, expr)

        p_ds = digger(p)
        pp_ds = digger(pp)
        f_ds = digger(fe) # first factoring

        countMap: Dict[MatrixExpr, int] = dict([
            (p, countTopTypes(WrapType, p_ds)),
            (pp, countTopTypes(WrapType, pp_ds)),
            (fe, countTopTypes(WrapType, f_ds)),
        ])

        # Get the first (get first expression corresponding to first minimum, since that is one of the ones that reduces num transpose)
        countMapSorted: List[Tuple[MatrixExpr, int]] = sorted(countMap.items(), key = operator.itemgetter(1)) # sorting by values

        result: MatrixExpr = countMapSorted[0][0]

        return result
        # TODO if this function works make it more efficient by integrating the algoPolarize into the function itself (like put a list of functinos and apply the params to each arg in a map)


    # Compare the two results for the Addition case: 

    if expr.func in OP_ADD_MUL_LIST: 
    #if expr in [MatAdd, Add, MatMul, Mul]: # NOTE: adding here Add and Mul because of the new addition freeze() function which results in Add / Mul instead to tolerate UnevaluateExpr otherwise matmul coeff error results (when there is a coeff * matrix)
        resultComp: MatrixExpr = hackpolarizeMatAddMul(WrapType, expr)
        resultGroup: MatrixExpr = hackpolarizeGroup(WrapType, expr)

        c_ds = digger(resultComp)
        g_ds = digger(resultGroup)

        if countTopTypes(WrapType, g_ds) > countTopTypes(WrapType, c_ds):
            return resultComp
        else: 
            return resultGroup # favor the group result even when num WrapType are equal for each expression
    
    # else apply the group algo
    resultGroup: MatrixExpr = hackpolarizeGroup(WrapType, expr)

    # TODO why doesn't this freeze work here? why must I declare separate lambda? 
    #resultGroup = freeze(resultGroup)# to keep the polarize result and avoid having the expr get evaluated when MatAdd form

    return resultGroup 


polarize = lambda t, e: freeze(_polarize(t, e))




def freeze(matadd: MatAdd) -> MatAdd:
    '''Given the polarize() result passed as argument, when it is a MatAdd, this function returns an expression preserving the effect of polarize() otherwise presence of MatAdd renders polarize efforts invisible since the args of the MatAdd are evaluated when presented in the MatAdd. 
    Uses string parsing and `UnevaluatedExpr` to "freeze" the polarize results. 

    Arguments: 
        `matadd`: the MatAdd result of polarize()
    
    Returns: 
        A `MatrixExpr` (`MatAdd`) expression that was parsed from a string, containing the original symbols in the original expression. 
    '''
    #assert matadd.is_MatAdd  # otherwise the expressions remain and there is no need for this freeze function
    if not (matadd.func in [MatAdd, Add]):
    #if not matadd.is_MatAdd:
        return matadd 

    SE = lambda e: srepr(UnevaluatedExpr(e))

    accFirst = ""

    # TODO fix this so that -2 can be recognized also not just -1
    # TODO need to use isMul and Mul constructor? 
    isMulNeg = lambda e: e.is_MatMul and (e.args[0] in [NegativeOne(-1), Number(-1), Integer(-1)])
    # TODO need Mul constructor or just MatMul?
    pickOutMulNeg = lambda e: MatMul(*e.args[1:]) if isMulNeg(e) else e 

    # NOTE: freeze works like this: whatever argument is already polarized() is passed through into the UnevaluatedExpr so that it gets "frozen". Otherwise the UnevaluatedExpr for a non-polarized arg does not freeze + polarize. 
    g = lambda accStr, nextE : "{} + {}".format(accStr, SE(nextE)) if not(isMulNeg(nextE)) else "{} - {}".format(accStr, SE(pickOutMulNeg(nextE)))

    seJoined: str = foldLeft(g, accFirst, matadd.args)

    # Create the symbols dict
    symbolsDict: Dict[str, MatrixSymbol] = dict(list(map(lambda s: (str(s), s), matadd.free_symbols)) )

    return parse_expr(seJoined, local_dict = symbolsDict)






# ------------------------------------------------------------------
# UNUSED:

def splitOnce(theArgs: List[MatrixSymbol], signalVar: MatrixSymbol, n: int) -> Tuple[List[MatrixSymbol], List[MatrixSymbol]]:

    assert n <= len(theArgs) and abs(n) == n

    cumArgs = []
    countSignal: int = 0

    for i in range(0, len(theArgs)):
        arg = theArgs[i]

        if arg == signalVar:
            countSignal += 1

            if countSignal == n:
                return (cumArgs, list(theArgs[i + 1: ]) )


        cumArgs.append(arg)

    return ([], [])


# NOTE: the symbols can be E.T or E or E.I (so are technically MatrixExpr's not MatrixSymbols)


ArgPair = Tuple[List[MatrixSymbol], List[MatrixSymbol]]

def splitArgs(givenArgs: List[MatrixExpr], signalVar: MatrixSymbol) -> List[Tuple[MatrixType, ArgPair]]:

    '''Splits even at the signalVar if it is wrapped in theArgs with a constructor, and returns a dict specifying under which constructor it was wrapped in. 
    
    For instance if we split A * E.T * B at signalVar = E then result is {Transpose : ([A], [B]) } because the signalVar was wrapped in a transpose. 
    
    Constructor == Transpose ---> dict key is Transpose
    Constructor == Inverse ---> dict key is Inverse
    Constructor == MatrixSymbol --> dict key is MatrixSymbol (or None?)
    '''
    theArgs = list(givenArgs)

    return [ ( theArgs[i].func  ,(theArgs[0:i], theArgs[i+1 : ]) ) for i in range(0, len(theArgs)) if theArgs[i].has(signalVar)] 

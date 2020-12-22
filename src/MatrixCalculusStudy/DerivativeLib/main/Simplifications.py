

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

from sympy.core.numbers import NegativeOne, Number

from sympy.core.assumptions import ManagedProperties



# Types
MatrixType = ManagedProperties


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


# Need to include list of constructors that you dig out of.
# TODO: need Trace / Derivative / Function ...??
CONSTR_LIST: List[MatrixType] = [Transpose, Inverse]


# TODO to add Function, Derivative, and Trace ? any others?
ALL_TYPES_LIST = [Transpose, Inverse, MatMul, MatAdd, MatrixSymbol, Symbol, Number]



hasConstr = lambda Constr, expr : Constr.__name__ in srepr(expr)
#hasTranspose = lambda e : "Transpose" in srepr(e)

anyTypeIn = lambda testTypes, searchTypes : any([True for tpe in testTypes if tpe in searchTypes])



isSym = lambda m: len(m.free_symbols) in [0,1]

# NOTE: len == 0 of free syms when Number else for MatSym len == 1 so put 0 case just for safety.
#onlySymComponents_AddMul = lambda expr: (expr.func in [MatAdd, MatMul]) and all(map(lambda expr: len(expr.free_symbols) in [0, 1], expr.args))

isSimpleArgs = lambda e: all(map(lambda a: len(a.free_symbols) in [0, 1], e.args))
isInnerExpr = lambda e: (e.func in [MatAdd, MatMul]) and isSimpleArgs(e)



def pickOut(Constr: MatrixType, expr: MatrixExpr):
    #if not hasTranspose(expr):
    if not hasConstr(Constr, expr):
        #return Transpose(MatMul(*expr.args))
        #return Transpose(expr)
        return Constr(expr)
    elif hasConstr(Constr, expr) and expr.func == Constr:
    #elif hasTranspose(expr) and expr.is_Transpose:
        return expr.arg

    return Constr(expr) #TODO check





def applyTypesToExpr( pairTypeExpr: Tuple[List[MatrixType], MatrixExpr]) -> MatrixExpr:
    '''Ignores any types that are MatrixSymbol or Symbol or instance of Number because those give error when applied to an expr.'''

    (typeList, expr) = pairTypeExpr

    isSymOrNum = lambda tpe : tpe in [MatrixSymbol, Symbol] or issubclass(tpe, Number)

    typeList = list(filter(lambda tpe: not isSymOrNum(tpe), typeList))

    if typeList == []:
        return expr
    return compose(*typeList)(expr)





def stackTypesBy(byType: MatrixType, types: List[MatrixType]) -> List[MatrixType]:
    '''Given expr type (like Transpose) and given expr list, this function pulls all the signal types to the end of the
    list, leaving the non-signal-types as the front'''

    # Get number of signal types in the list
    countTypes: int = len(list(filter(lambda t : t == byType, types)))

    # Create the signal types that go at the end
    endTypes: List[MatrixType] = [byType] * countTypes

    # Get the list without the signal types
    nonSignalTypes: List[MatrixType] = list(filter(lambda t : t != byType, types))

    return endTypes + nonSignalTypes # + endTypes





def chunkTypesBy(byTypes: List[MatrixType], types: List[MatrixType]) -> List[List[MatrixType]]:
    '''Separates the `types` in expr list of list of types by the types in `byConstrs` and keeps other types separate too, just as how they appear in the original list'''

    # Not strictly necessary here, just useful if you want to see shorter version of names below
    #getSimpleTypeName = lambda t : str(t).split("'")[1].split(".")[-1]
    byConstrs: List[MatrixType] = list(set(byTypes))

    # Step 1: coding up in pairs for easy identification: need Inverse and Transpose tagged as same kind
    codeConstrPairs: List[Tuple[int, MatrixType]] = list(map(lambda c : (0, c) if (c in byConstrs) else (1, c), types))

    # Step 2: getting the groups
    chunkedPairs: List[List[Tuple[int, MatrixType]]] = [list(group) for key, group in itertools.groupby(codeConstrPairs, operator.itemgetter(0))]

    # Step 3: getting only the types in the chunked lists
    chunkedConstrs: List[List[MatrixType]] = list(map(lambda lst : list(map(lambda pair : pair[1], lst)), chunkedPairs))

    return chunkedConstrs




def chunkExprsBy(byTypes: List[MatrixType], expr: MatrixExpr) -> Tuple[List[List[MatrixType]], List[List[MatrixExpr]]]:
    '''Given an expression, returns the types and expressions as tuples, listed in preorder traversal, and separated by which expressions are grouped by Transpose or Inverse constructors (in their layering)'''

    byConstrs: List[MatrixType] = list(set(byTypes))

    ps: List[MatrixExpr] = list(preorder_traversal(expr)) # elements broken down
    cs: List[MatrixType] = list(map(lambda p: type(p), ps)) # types / constructors


    # Check first: does the expr have the types in byConstrs? If not, then return it out, nothing to do here:
    #if not (Transpose in cs):
    #BAD tests for AND all types: if not (set(byConstrs).intersection(cs) == set(byConstrs)):
    # GOOD, tests for OR all types (since should be able to chunk [T, T, T] then [I, I] and [I, T, I, T] so need the OR relationship):
    if not anyTypeIn(testTypes = CONSTR_LIST, searchTypes = cs):
        return ([cs], [ps])

    csChunked: List[List[MatrixType]] = chunkTypesBy(byTypes = byConstrs, types = cs)


    # Get the lengths of each chunk
    chunkLens: List[int] = list(map(lambda cLst : len(cLst), csChunked))


    # Use lengths to segregate the preorder traversal exprs also, then later to apply the transformations
    psChunked: List[List[MatrixExpr]] = []
    rest: List[MatrixExpr] = ps

    for size in chunkLens:
        (fst, rest) = (rest[:size], rest[size: ])
        psChunked.append( fst )

    return (csChunked, psChunked)








def group(byType: MatrixType, expr: MatrixExpr, combineAdds:bool = False) -> MatrixExpr:
    '''Combines transposes when they are the outermost operations.

    Brings the individual transpose operations out over the entire group (factors out the transpose and puts it as the outer operation).
    NOTE: This happens only when the transpose ops are at the same level. If they are nested, that "bringing out" task is left to the rippletranspose function.

    combineAdds:bool  = if True then the function will group all the addition components under expr transpose, like in matmul, otherwise it will group only the individual components of the MatAdd under transpose.'''


    def revAlgo(Constr: MatrixType, expr: MatrixExpr):
        '''Converts B.T * A.T --> (A * B).T'''

        if hasConstr(Constr, expr) and expr.is_MatMul:

            revs = list(map(lambda a : pickOut(Constr, a), reversed(expr.args)))

            #return Transpose(MatMul(*revs))
            return Constr(MatMul(*revs))

        return expr


    def algo_Group_MatMul_or_MatSym(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

        ps = list(preorder_traversal(expr))
        ms = list(filter(lambda p : p.is_MatMul, ps))
        ts = list(map(lambda m : revAlgo(byType, m), ms))

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

    if not expr.is_MatAdd: # TODO must change this to identify matmul or matsym exactly
        return algo_Group_MatMul_or_MatSym(byType, expr)


    Constr = expr.func

    # TODO fix this to handle any non-add operation upon function entry
    addendsTransp: List[MatrixExpr] = list(map(lambda a: group(byType, a), expr.args))

    if combineAdds:
        innerAddends = list(map(lambda t: pickOut(byType, t), addendsTransp))
        #return Transpose(MatAdd(*innerAddends))
        return byType (Constr(*innerAddends))

    # Else not combining adds, just grouping transposes in addends individually
    # TODO fix this to handle any non-add operation upon function entry. May not be a MatAdd, may be Trace for instance.

    #return MatAdd(*addendsTransp)
    return Constr(*addendsTransp)








def rippleOut(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

    '''Brings transposes to the outermost level when in expr nested expression. Leaves the nested expressions in their same structure.
    Because it preserves the nesting structure, it is expr kind of shallow polarize() function'''


    def algo_RippleOut_MatMul_or_MatSym(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr:
        '''For each layered (nested) expression where transpose is the inner operation, this function brings transposes to be the outer operations, leaving all other operations in between in the same order.'''

        assert byType in CONSTR_LIST

        (csChunked, psChunked) = chunkExprsBy(byTypes = CONSTR_LIST, expr = expr)

        # Order the types properly now for each chunk: make transposes go last in each chunk:
        stackedChunks = list(map(lambda lst : stackTypesBy(byType = byType, types = lst), csChunked))

        # Pair up the correct order of transpose types with the expressions
        # BEFORE: (Transpose in tsPs[0]) or (Inverse in tsPs[0])
        typeListExprListPair = list(filter(lambda tsPs : anyTypeIn(testTypes = CONSTR_LIST, searchTypes = tsPs[0]),
                                           list(zip(stackedChunks, psChunked))))



        #typeListExprPair = list(map(lambda tsPs : (tsPs[0], tsPs[1][0]), typeListExprListPair))

        # Get the first expression only, since it is the most layered, don't use the entire expression list of the
        # tuple's second part. And get the inner argument (lay bare) in preparation for apply the correct order of
        # `byType`s.
        typeListInnerExprPair = list(map(lambda tsPs : (tsPs[0], inner(tsPs[1][0])), typeListExprListPair))

        outs = list(map(lambda tsExprPair : applyTypesToExpr(tsExprPair), typeListInnerExprPair ))

        # Get the original expressions as they were before applying correct transpose
        ins = list(map(lambda tsPs : tsPs[1][0], typeListExprListPair))

        # Filter: get just the matmul-type arguments (meaning not the D^T or E^-1 type arguments) from the result list (assuming there are other MatMul exprs). Could have done this when first filtering the psChunked, but easier to do it now.
        # NOTE: when there are ONLY mat syms and no other matmuls then we must keep them since it means the expression is layered with only expr matsum as the innermost expression, rather than expr matmul.
        isSymOrNum = lambda expr : expr.is_Symbol or expr.is_Number
        #isSym = lambda expr  : len(expr.free_symbols) == 1

        # Flattening the chunked ps list for easier evaluation:
        ps: List[MatrixExpr] = list(itertools.chain(*psChunked))
        allSymOrNums = len(ps) == len(list(filter(lambda e: isSymOrNum(e), ps)) )
        #allSyms = all(map(lambda expr : isSym(expr), ps))

        #if not allSymOrNums:
        outs = list(filter(lambda expr : not isSymOrNum(expr), outs))

        ins = list(filter(lambda expr : not isSymOrNum(expr), ins))
        # else just leave the syms as they are.


        # Zip the non-transp-out exprs with the transp-out expressions as list (NOTE: cannot be dictionary since we need to keep the same order of expressions as obtained from the preorder traversal, else substitution order will be messed up)
        outInPairs = list(zip(outs, ins))

        # Now must apply from the beginning to end, each of the expressions. Must replace in the first expr, all of the latter expressions, kind of like folding operation of Matryoshka dolls, to preserve all the end changes: C -> goes into -> B -> goes into -> A
        accFirst = expr #outs[0]

        f = lambda acc, outInPair: acc.xreplace({outInPair[1] : outInPair[0]})

        resultWithTypeOut = foldLeft(f , accFirst,  outInPairs) # outsNotsPairs[1:])

        return resultWithTypeOut



    if not expr.is_MatAdd: # TODO must change this to identify matmul or matsym exactly
        return algo_RippleOut_MatMul_or_MatSym(byType = byType, expr = expr)

    Constr: MatrixType = expr.func

    componentsOut: List[MatrixExpr] = list(map(lambda a: rippleOut(byType = byType, expr = a), expr.args))

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
    typeListExprListPair = list(filter(lambda tsPs : anyTypeIn(testTypes = CONSTR_LIST, searchTypes = tsPs[0]),
                                       list(zip(csChunked, psChunked))))

    # Get the first expression only, since it is the most layered, and pair its inner arg with the list of types from that pair.
    typeListInnerPair = list(map(lambda tsPs : (tsPs[0], inner(tsPs[1][0])), typeListExprListPair))

    return typeListInnerPair







def inner(expr: MatrixExpr) -> MatrixExpr:
    '''Gets the innermost expression (past all the .arg) on the first level only'''
    #isMatSym = lambda e : len(e.free_symbols) == 1

    # TODO missing any base case possibilities? Should include here anything that is not Trace / Inverse ... etc or any kind of constructors that houses an inner argument.
    Constr = expr.func
    #types = [MatMul, MatAdd, MatrixSymbol, Symbol, Number]
    otherTypes = list( set(ALL_TYPES_LIST).symmetric_difference(CONSTR_LIST) )
    #isAnySubclass = any(map(lambda t : issubclass(Constr, t), types))

    if (Constr in otherTypes) or issubclass(Constr, Number):

        return expr

    # else keep recursing
    return inner(expr.arg) # need to get arg from Trace or Transpose or Inverse ... among other constructors






def innerTrail(expr: MatrixExpr) -> List[Tuple[List[MatrixType], MatrixExpr]]:
    '''Gets the innermost expr (past all the .arg) on the first level only, and stores also the list of constructors'''

    def doInnerTrail(expr: MatrixExpr, accConstrs: List[MatrixType]) -> Tuple[List[MatrixType], MatrixExpr]:

        Constr = expr.func
        #types = [MatMul, MatAdd, MatrixSymbol, Symbol, Number]
        otherTypes = list( set(ALL_TYPES_LIST).symmetric_difference(CONSTR_LIST) )

        #isAnySubclass = any(map(lambda t : issubclass(Constr, t), otherTypes))

        #if (Constr in otherTypes) or isAnySubclass:
        if (Constr in otherTypes) or issubclass(Constr, Number):

            return (accConstrs, expr)

        # else keep recursing
        return doInnerTrail(expr = expr.arg, accConstrs = accConstrs + [Constr])

    return doInnerTrail(expr = expr, accConstrs = [])




def factor(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

    typesInners = digger(expr)

    # Check: if empty list returned by digger (say since expr is just MatrixSymbol) then need to exit and return original expression (since there must be nothing to factor).
    if typesInners == []:
        return expr
    # Otherwise error results on next line:


    (types, innerExprs) = list(zip(*typesInners))
    (types, innerExprs) = (list(types), list(innerExprs))

    # Filtering the wrapper types that are `byType`s
    noSignalTypes: List[MatrixType] = list(map(lambda typeList:  list(filter(lambda t: t != byType, typeList)) , types))

    # Pair up all the types, filtered types, and inner expressions for easier querying later:
    triples: List[Tuple[List[MatrixType], List[MatrixType], List[MatrixExpr]]]  = list(zip(types, noSignalTypes, innerExprs))

    # Create new pairs from the filtered and inner Exprs, by attaching expr Transpose at the end if odd num else none.
    newTypesInners: List[Tuple[List[MatrixType], List[MatrixExpr]]] = list(map(lambda triple: ([byType] + triple[1], triple[2]) if (triple[0].count(byType) % 2 == 1) else (triple[1], triple[2]) , triples))

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













def wrapShallow(WrapType: MatrixType, innerExpr: MatrixExpr) -> MatrixExpr:
    '''The wrapping algo for taking expr set of simple arguments and enveloping them in the given `WrapType`.'''

    Constr: MatrixType = innerExpr.func
    #assert Constr in [MatAdd, MatMul]

    numArgsOfType: int = len(list(filter(lambda a: type(a) == WrapType, innerExpr.args )))

    # Building conditions for checking if we need to wrap the expr, or else return it as is.
    mostSymsAreOfType: bool = (numArgsOfType / len(innerExpr.args) ) >= 0.5

    # NOTE: len == 0 of free syms when Number else for MatSym len == 1 so put 0 case just for safety.
    #onlySymComponents_AddMul = lambda expr: (expr.func in [MatAdd, MatMul]) and all(map(lambda expr: len(expr.free_symbols) in [0, 1], expr.args))

    mustWrap: bool = (Constr in [MatAdd, MatMul]) and mostSymsAreOfType #and onlySymComponents_AddMul(innerExpr)

    if not mustWrap:
        return innerExpr

    # Else do the wrapping algorithm:
    invertedArgs: List[MatrixExpr] = list(map(lambda a: pickOut(Constr = WrapType, expr = a), innerExpr.args))


    invertedArgs: List[MatrixExpr] = list(reversed(invertedArgs)) if Constr == MatMul else invertedArgs

    wrapped: MatrixExpr = WrapType(Constr(*invertedArgs))

    return wrapped










def wrapDeep(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

    Constr = expr.func

    if isSym(expr):
        return expr

    elif isInnerExpr(expr):
        wrappedExpr: MatrixExpr = wrapShallow(WrapType = WrapType, innerExpr = expr)
        return wrappedExpr

    elif Constr in [MatAdd, MatMul]:  #then split the polarizing operation over the arguments since any one of the args can be an inner expr.

        # NOTE: must also factor the result so that the next step of wrapNest works correctly (doesn't contain superfluous transposes)
        wrappedArgs: List[MatrixExpr] = list(map(lambda a: factor(WrapType, wrapDeep(WrapType, a)), expr.args))

        exprWithPartsWrapped: MatrixExpr = Constr(*wrappedArgs)

        exprOverallWrapped: MatrixExpr = wrapShallow(WrapType = WrapType, innerExpr = exprWithPartsWrapped)

        return exprOverallWrapped

    
    else: # else is Trace, Transpose, or Inverse or any other constructor
        innerExpr = expr.arg

        return Constr( wrapDeep(WrapType = WrapType, expr = innerExpr) )






# Making expr group transpose that goes deep until innermost expression (largest depth) and applies the group algo there
# Drags out even the inner transposes (aggressive grouper)
def polarize(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr:
    '''Given an expression with innermost args nested as components inside another expression, (many nestings), this function tries to drag / pull / force out all the transposes from the groups of expressions of matmuls / matmadds and from individual symbols.

    Tries to create one nesting level that mentions transpose (there can be other nestings with inverse inside for instance but the outermost nesting must be the only one with transpose).

    There must be no layering of transpose in the nested expressions in the result returned by this function -- polarization of transpose to the outer edges.'''


    def algoPolarize(byType: MatrixExpr, expr: MatrixExpr) -> MatrixExpr:
        # Need to factor out transposes and ripple them out first before passing to wrap algo because the wrap algo won't reach in and factor or bring out inner transposes, will just overlay on top of them without simplifying (bad, since yields more complicated expression)
        fe = factor(byType = byType, expr = expr) # no need for ripple out because factor does this implicitly

        wfe = wrapDeep(WrapType = byType, expr = fe)

        # Must bring out the extra transposes that are brought out by the wrap algo and then cut out extra ones.
        fwfe = factor(byType = byType, expr = wfe)

        return fwfe

    def hack_polarizeMatAddMul(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr: 
        assert expr.is_MatAdd or expr.is_MatMul 

        polarizedArgs: List[MatrixExpr] = list(map(lambda a: hack_polarizeGroup(byType = byType, expr = a), expr.args))

        Constr = expr.func 

        return Constr(*polarizedArgs)


    countTopTypes = lambda byType, pairs: sum(list(map(lambda pair: True if pair[0][0] == byType else False, pairs)) )
    

    def hack_polarizeGroup(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr: 
        # TODO fix if this extra step of factoring and wrapping deep results in more transposes then  polarize again or just leave at the previous factor

        p = algoPolarize(byType = byType, expr = expr)
        pp = algoPolarize(byType = byType, expr = p )
        fe = factor(byType, expr)

        p_ds = digger(p)
        pp_ds = digger(pp)
        f_ds = digger(fe) # first factoring

        countMap: Dict[MatrixExpr, int] = dict([
            (p, countTopTypes(byType, p_ds)),
            (pp, countTopTypes(byType, pp_ds)),
            (fe, countTopTypes(byType, f_ds)),
        ])

        # Get the first (get first expression corresponding to first minimum, since that is one of the ones that reduces num transpose)
        countMapSorted: List[Tuple[MatrixExpr, int]] = sorted(countMap.items(), key = operator.itemgetter(1)) # sorting by values

        result: MatrixExpr = countMapSorted[0][0]

        return result
        # TODO if this function works make it more efficient by integrating the algoPolarize into the function itself (like put a list of functinos and apply the params to each arg in a map)


    # Compare the two results for the Addition case: 

    if expr.is_MatAdd or expr.is_MatMul:
        resultComp: MatrixExpr = hack_polarizeMatAddMul(byType, expr)
        resultGroup: MatrixExpr = hack_polarizeGroup(byType, expr)

        c_ds = digger(resultComp)
        g_ds = digger(resultGroup)

        if countTopTypes(byType, g_ds) > countTopTypes(byType, c_ds):
            return resultComp
        else: 
            return resultGroup # favor the group result even when num byTypes are equal for each expression
    
    # else apply the group algo
    resultGroup: MatrixExpr = hack_polarizeGroup(byType, expr)

    return resultGroup 






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

#----------------------------------------------------------------------
#
# FUNCTIONS FOR THE AUTOMATIC DIFFERENTIATION  OF MATRICES WITH SYMPY
# 
#----------------------------------------------------------------------
 
from sympy import *
from sympy.printing.str import StrPrinter
from sympy.printing.latex import LatexPrinter
 
 
 
#####  M  E  T  H  O  D  S
 
 
 
def matrices(names):
    ''' Call with  A,B,C = matrix('A B C') '''
    return symbols(names,commutative=False)
 
 
# Transformations
 
d = Function("d",commutative=False)
inv = Function("inv",commutative=False)
 
class t(Function):
    ''' The transposition, with special rules
        t(A+B) = t(A) + t(B) and t(AB) = t(B)t(A) '''
    is_commutative = False
    def __new__(cls,arg):
        if arg.is_Add:
            return Add(*[t(A) for A in arg.args])
        elif arg.is_Mul:
            L = len(arg.args)
            return Mul(*[t(arg.args[L-i-1]) for i in range(L)])
        else:
            return Function.__new__(cls,arg)
 
 
# Differentiation
 
MATRIX_DIFF_RULES = { 
         
        # e =expression, s = a list of symbols respsect to which
        # we want to differentiate
         
        Symbol : lambda e,s : d(e) if (e in s) else 0,
        Add :  lambda e,s : Add(*[matDiff(arg,s) for arg in e.args]),
        Mul :  lambda e,s : Mul(matDiff(e.args[0],s),Mul(*e.args[1:]))
                      +  Mul(e.args[0],matDiff(Mul(*e.args[1:]),s)) ,
        t :   lambda e,s : t( matDiff(e.args[0],s) ),
        inv : lambda e,s : - e * matDiff(e.args[0],s) * e
}
 
def matDiff(expr,symbols):
    if expr.__class__ in MATRIX_DIFF_RULES.keys():
        return  MATRIX_DIFF_RULES[expr.__class__](expr,symbols)
    else:
        return 0
 
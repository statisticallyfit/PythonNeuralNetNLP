
print("hi")
print(1987+1)

# <codecell> One lint inputs
print("Hello world!")

# %% Render LaTex
import sympy as sp
x, y, z = sp.symbols('x y z')
x
y
z

# TODO
f = sp.sin(x * y) + sp.cos(y * z)
f

sp.integrate(f, x)


import numpy as np
t = np.linspace(0, 20, 500)
t



# <codecell> mypy testing
from typing import List, Set, Dict, Tuple, Optional

# For simple built-in types, just use the name of the type
x: int = 1
x: bool = True
x: str = "test"
x: bytes = b"test"

# For collections, the name of the type is capitalized, and the
# name of the type inside the collection is in brackets
x: List[int] = [1]
x: Set[int] = {6, 7}
x

# Same as above, but with type comment syntax
x = [1]  # type: List[int]
x

# For mappings, we need the types of both keys and values
x: Dict[str, float] = {'field': 2.0}
x


# For tuples, we specify the types of all the elements
x: Tuple[int, str, float] = (3, "yes", 7.5)

# Use Optional[] for values that could be None
def some_function() -> int:
    x = 2 + 3
    return x


x: Optional[int] = some_function()
x

# Mypy understands a value can't be None in an if-statement
if x is not None:
    print(x.upper())
print(x)



# <codecell>
from typing import Union
import mypy
x: Union[int, str] = 1
x: Union[int, str] = 1.1  # Error!


l = []
l: List[int] = []       # Create empty list with type List[int]
d: Dict[str, int] = {}  # Create empty dictionary (str -> int)


# <codecell>
i, found = 0, False # type: (int, bool)      # OK
# <codecell>
(i, found) = 0, False # type: int, bool      # OK
# <codecell>
i, found = (0, False) # type: int, bool      # OK
# <codecell>
(i, found) = (0, False) # type: (int, bool)  # OK
(i, found)



name = "John"
age: int = 49
import datetime
birthday = datetime.date.today
age: str = 'ffity'
if age < birthday:
    something = name + age



def main():
    x = some_function()
    print(x)

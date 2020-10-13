

from typing import * 

# Finds indices where a condition is met
def find(xs: List[Any], condition: bool) -> List[int]:
    indices = [i  for i in range(0, len(xs))  if (condition)]

    return indices


# Finds indices where an expression matches elements in the given xs list. 
def findWhere(xs: List[Any], expr: Any) -> List[int]:
    indices = [i  for i in range(0, len(xs))  if (xs[i] == expr)]

    return indices
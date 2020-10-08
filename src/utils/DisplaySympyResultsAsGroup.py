from IPython.display import display

from typing import List, Any


def showGroup(group: List[Any]) -> None:
    list(map(lambda elem : display(elem), group))

    return None
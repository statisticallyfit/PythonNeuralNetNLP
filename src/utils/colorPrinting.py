from IPython.core.display import HTML

#color="255,127,80"  # coral

def TODO(textTodo: str, color: str = "red"):
    source = "<h1 style='color: {})'>{}</h1>".format(color, textTodo)
    return HTML(source)


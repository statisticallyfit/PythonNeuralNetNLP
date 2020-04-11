Source: https://github.com/explosion/thinc/blob/master/examples/01_intro_model_definition_methods.ipynb

# Intro to Thinc's `Model` class, model definition, and methods

Thinc uses a functional-programming approach to model definition, effective for:
* complicated network architectures and,
* use cases where different data types need to be passed through the network to reach specific subcomponents.

This notebook shows how to compose Thinc models and use the `Model` class and its methods.

*Principles:*
* Thinc provides layers (functions to create `Model` instances)
* Thinc tries to avoid inheritance, preferring function composition.


```python
import numpy
from thinc.api import Linear, zero_init

nI: int = 16
nO: int = 10
NUM_HIDDEN: int = 128


nIn = numpy.zeros((NUM_HIDDEN, nI), dtype="f")
nOut = numpy.zeros((NUM_HIDDEN, nO), dtype="f")
```


```python
nIn.shape
nOut.shape
```




    (128, 10)




```python
model = Linear(nI = nI, nO = nO, init_W = zero_init)
model

model.get_dim("nI")
model.get_dim("nO")

print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}.")
```

    Initialized model with input dimension nI=16 and output dimension nO=10.


*Key Point*: Models support dimension inference from data. You can defer some or all of the dimensions.


```python
modelDeferDims = Linear(init_W = zero_init)
modelDeferDims
print(f"Initialized model with no input/ouput dimensions.")
```

    Initialized model with no input/ouput dimensions.



```python
X = numpy.zeros((NUM_HIDDEN, nI), dtype="f")
Y = numpy.zeros((NUM_HIDDEN, nO), dtype="f")

# Here is where the dimension inference happens: during initialization of the model
modelDeferDims.initialize(X = X, Y = Y)
modelDeferDims

# We can see that dimension inference has occurred:
modelDeferDims.get_dim("nI")
modelDeferDims.get_dim("nO")

print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}.")
```

    Initialized model with input dimension nI=16 and output dimension nO=10.


## [Combinators](https://thinc.ai/docs/api-layers#combinators)

There are functions like `chain` and `concatenate` which are called [`combinators`](https://thinc.ai/docs/api-layers#combinators). *Combinators* take one or more models as arguments, and return another model instance, without introducing any new weight parameters.

Combinators are layers that express higher-order functions: they take one or more layers as arguments and express some relationship or perform some additional logic around the child layers.

### [`chain()`](https://thinc.ai/docs/api-layers#chain)
**Purpose of `chain`**: The `chain` function wires two models together with a feed-forward relationship. Composes two models `f` and `g` such that they become layers of a single feed-forward model that computes `g(f(x))`.

Also supports chaining more than 2 layers.
* NOTE: dimension inference is useful here.


```python
from thinc.api import chain, glorot_uniform_init

NUM_HIDDEN: int = 128
X = numpy.zeros((NUM_HIDDEN, nI), dtype="f")
Y = numpy.zeros((NUM_HIDDEN, nO), dtype="f")

# Linear layer multiplies inputs by a weights matrix and adds a bias vector
# layer 1: Linear layer wih only the output dimension provided
# layer 2: Linear layer with all dimensions deferred
modelChained = chain(layer1 = Linear(nO = NUM_HIDDEN, init_W = glorot_uniform_init),
                     layer2 = Linear(init_W = zero_init), )
modelChained
```




    <thinc.model.Model at 0x7efefad64730>




```python
# Initializing model
modelChained.initialize(X = X, Y = Y)
modelChained

modelChained.layers
```




    [<thinc.model.Model at 0x7effb03f98c8>, <thinc.model.Model at 0x7effb03f96a8>]




```python
nI: int = modelChained.get_dim("nI")
nI
nO: int = modelChained.get_dim("nO")
nO

nO_hidden = modelChained.layers[0].get_dim("nO")
nO_hidden


print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}.")
print(f"The size of the hidden layer is {nO_hidden}.")
```

    Initialized model with input dimension nI=16 and output dimension nO=10.
    The size of the hidden layer is 128.


## [`concatenate()`](https://thinc.ai/docs/api-layers#concatenate)

**Purpose of `concatenate()`**: the `concatenate` combinator function produces a layer that *runs the child layer separately* and then *concatenates their outputs together*. Useful for combining features from different sources. (Thinc uses this to build spacy's embedding layers).  Composes two or more models `f`, `g`, etc, such that their outputs are concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`.
* NOTE: functional approach here


```python
from thinc.api import concatenate

modelConcat = concatenate(Linear(nO = NUM_HIDDEN), Linear(nO = NUM_HIDDEN))
modelConcat
modelConcat.layers

# Initializing model, and this is where dimension inference occurs (for nI)
modelConcat.initialize(X = X)

# Can see that dimension nI was inferred
nI: int = modelConcat.get_dim("nI")
nI

# Can see that dimension nO is now twice the NUM_HIDDEN which we passed in: 256 = 128 + 128 since concatenation occurred.
nO: int = modelConcat.get_dim("nO")
nO
print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}.")
```

    Initialized model with input dimension nI=16 and output dimension nO=256.


## [`clone()`](https://thinc.ai/docs/api-layers#clone)
Some combinators work on a layer and a numeric argument. The `clone` combinator creates a number of copies of a layer and chains them together into a deep feed-forward network.

**Purpose of `clone`**: Construct `n` copies of a layer with distinct weights. For example, `clone(f, 3)(x)` computes `f(f'(f''(x)))`

* NOTE: shape inference is useful here: we want the first and last layers to have different shapes so we can avoid giving any dimensions into the layer we clone. Then we just have to specify the first layer's output size and let the res of the dimensions be inferred from the data.


```python
from thinc.api import clone

modelClone = clone(orig = Linear(), n = 5)
modelClone
modelClone.layers
```




    [<thinc.model.Model at 0x7efefad64ea0>,
     <thinc.model.Model at 0x7efefad64f28>,
     <thinc.model.Model at 0x7efefad647b8>,
     <thinc.model.Model at 0x7efefad64598>,
     <thinc.model.Model at 0x7efefad64510>]




```python
modelClone.layers[0].set_dim("nO", NUM_HIDDEN)
modelClone.layers[0].get_dim("nO")

# Initializing the model here
modelClone.initialize(X = X, Y = Y)

nI: int = model.get_dim("nI")
nI
nO: int = model.get_dim("nO")
nO

# num hidden is still 128
modelClone.layers[0].get_dim("nO")

print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}.")
```

    Initialized model with input dimension nI=16 and output dimension nO=10.


Can apply `clone` to model instances that have child layers, making it easier to define complex architectures. For instance: usually we want to attach an activation and dropout to a linear layer and then repeat that substructure a number of times.


```python
from thinc.api import Relu, Dropout

def hiddenLayer(dropout: float = 0.2):
    return chain(Linear(), Relu(),  Dropout(dropout))

modelCloneHidden = clone(hiddenLayer(), 5)
modelCloneHidden
```




    <thinc.model.Model at 0x7efefad0a9d8>



## [`with_array()`](https://thinc.ai/docs/api-layers#with_array)
Some combinators are unary functions (they take only one model). These are usually **input and output transformations*. For instance:
**Purpose of `with_array`:** produce a model that flattens lists of arrays into a single array and then calls the child layer to get the flattened output. Then, it reverses the transformation on the output. (In other words: Transforms sequence of data into a continguous two-dim array on the way into and out of a model.)


```python
from thinc.api import with_array, Model

modelWithArray: Model = with_array(layer = Linear(nO = 4, nI = 2))
modelWithArray

Xs = [modelWithArray.ops.alloc2f(d0 = 10, d1 = 2, dtype = "f")]
Xs
Xs[0].shape

modelWithArray.initialize(X = Xs)
modelWithArray

# predict(X: InT) -> OutT: call the model's `forward` function with `is_train = False` and return the output instead of the tuple `(output, callback)`.
Ys = modelWithArray.predict(X = Xs)
Ys

print(f"Prediction shape: {Ys[0].shape}.")
```

    Prediction shape: (10, 4).


## Example of Concise Model Definition with Combinators
Combinators allow you to wire complex models very concisely.

Can take advantage of Thinc's **operator overloading** which lets you use infox notation. Must be careful to use **in a contextmananger** to avoid unexpected results.

**Example network**:

1. Below, the network expects a list of arrays as input, where each array has two columns with different numeric identifier features.
2. The two arrays are embedded using separate embedding tables
3. The two resulting vectors are added
4. Then passed through the `Maxout` layer with layer normalization and dropout.
5. The vectors pass through two pooling functions (`reduce_max` and `reduce_mean`) and the results are concatenated.
6. The concatenated results are passed through two `Relu` layers with dropout and residual connections.
7. The vectors are passed through an output layer, which has a `Softmax` activation.


```python
from thinc.api import add, chain, concatenate, clone
from thinc.api import with_array, reduce_max, reduce_mean, residual
from thinc.api import Model, Embed, Maxout, Softmax

nH: int = 5 # num hidden layers

with Model.define_operators({">>": chain, "|":concatenate, "+":add, "**":clone}):
    modelOp: Model = (
        with_array(layer =
                   # Embed: map integers to vectors using fixed-size lookup table.
                   (Embed(nO = 128, column = 0) + Embed(nO = 64, column=1))
        >> Maxout(nO = nH, normalize = True, dropout = 0.2)
    )
    >> (reduce_max() | reduce_mean())
    >> residual(layer = Relu() >> Dropout(rate = 0.2)) ** 2
    >> Softmax()
)

modelOp
modelOp.layers
modelOp.attrs
modelOp.param_names
modelOp.grad_names
modelOp.dim_names
modelOp.ref_names
modelOp.define_operators
modelOp.walk
modelOp.to_dict
```




    <bound method Model.to_dict of <thinc.model.Model object at 0x7efefad20b70>>



## Using A Model
Defining the model:


```python
from thinc.api import Linear, Adam
import numpy

nI, nO, nH = 10, 10, 128
nI, nO, nH

X = numpy.zeros((nH, nI), dtype="f")
dY = numpy.zeros((nH, nO), dtype="f")

modelBackpropExample: Model = Linear(nO = nO, nI = nI)
```

Initialize the model with a sample of the data:


```python
modelBackpropExample.initialize(X=X, Y=dY)
```




    <thinc.model.Model at 0x7efefad20bf8>



Run some data through the model:


```python
Y = modelBackpropExample.predict(X = X)
Y
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)



Get a callback to backpropagate:


```python
# begin_update(X: InT) -> Tuple[OutT, Callable[[InT], OutT]]
# Purpose: Run the model over a batch of data, returning the output and a callback to complete the backward pass.
# Return: tuple (Y, finish_update), where Y = batch of output data, and finish_update = callback that takes the gradient with respect to the output and an optimizer function to return the gradient with respect to the input.
Y, backprop = modelBackpropExample.begin_update(X = X)
Y, backprop
```




    (array([[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]], dtype=float32),
     <function thinc.layers.linear.forward.<locals>.backprop(dY:thinc.types.Floats2d) -> thinc.types.Floats2d>)




Run the callback to calculate the gradient with respect to the inputs.

`backprop()`:
* is a callback to calculate gradient with respect to inputs.
* only increments the parameter gradients, doesn't actually change the weights. To increment the weights, call `model.finish_update()` and pass an optimizer
* If the model has trainable parameters, gradients for the parameters are accumulated internally, as a side-effect.




```python
dX = backprop(dY)
dX
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)



Incrementing the weights now by calling `model.finish_update()` and by passing an optimizer.

`finish_update(optimizer: Optimizer) -> None`
* update parameters with current gradients
* the optimizer is called with each parameter and gradient of the model


```python
adamOptimizer = Adam()

modelBackpropExample.finish_update(optimizer = adamOptimizer)
modelBackpropExample
```




    <thinc.model.Model at 0x7efefad20bf8>



Get and set dimensions, parameters, attributes, by name:


```python
modelBackpropExample.get_dim("nO")
# weights matrix
W = modelBackpropExample.get_param("W")
W

modelBackpropExample.attrs["something"] = "here"

modelBackpropExample.attrs.get("foo", "bar")
```




    'bar'



Get parameter gradients and increment them explicitly:


```python
dW = modelBackpropExample.get_grad("W")
dW

modelBackpropExample.inc_grad(name = "W", value = 1 + 0.1)
modelBackpropExample.get_grad("W")
```




    array([[1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
           [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]], dtype=float32)



Can serialize model to bytes and to dist and load them back with `from_bytes` and `from_disk`


```python
modelBytes = modelBackpropExample.to_bytes()
modelBytes
```




    b'\x84\xa5nodes\x91\x84\xa5index\x00\xa4name\xa6linear\xa4dims\x82\xa2nO\n\xa2nI\n\xa4refs\x80\xa5attrs\x91\x81\xa9something\xc4\x05\xa4here\xa6params\x91\x82\xa1W\x85\xc4\x02nd\xc3\xc4\x04type\xa3<f4\xc4\x04kind\xc4\x00\xc4\x05shape\x92\n\n\xc4\x04data\xc5\x01\x90\x19\xc0~\xbe2gk\xbe\x9c\xbd\xa2\xbe\xbf\xabs\xbd\x986\x86<\xfcx\x1a=\xa8\xc2\x8c\xbe3Y\xcd>qt\xde>\x06\x8b\xd7=m\xf9;\xbe\x04i\x83\xbeZ\x93\xba>\xabU\xf2>t\xd9Q\xbe&1r\xbcWA\x98\xbe)\xd8\x90>\xf0\x95\x9c=A\xaf\xad\xbc\xf9uo>\x8a\x14\xaa>\x87\x0f\x9e\xbe\xf5\x10\x81\xbeFo>\xbe\x96tm\xbd\xc7T\xf1>[\xc4\x96>\xba#\xd7\xbe\xc1.W\xbe\xa8\xefM>+2\xa7\xbe;n\x97\xbd\x14\x8b\xb0>\xb9>\x97>x\xa5\xcc>\x13\xc5j\xbe;\x97\xa7=\xde\xbc\xe0\xbe\x16:\xb2>K\xf01\xbd2\xaa\xe1\xbc8Y\x19=U\rQ>\x89\xb6\xe8\xbe\xe2h\xf4\xbd>\x9a\x96\xbe\'7\xd9\xbe\x0e\xb6\xbc\xbc\x12\x1b\x04?\x07\xdd\xae>\xb8\x885\xbcH\x9d\xb1\xbe\xfa\xcb\x03\xbeO\xcb\xd6>\x88\x1b\xef\xbee\x18\x13\xbe<\x07\xe4\xbe\x0b\xd1\xa9\xbe\x00\x07\x96\xbe0\x8d\'>\x17\xf8\xe3>`@\xee\xbeH\x94\x98>p\xbd"<3;\xdb>c\xd7\xf3\xbeU\x11Y\xbe\x90\x8c2\xbe\xb4\x0fV>\xecl\xde\xbe\xbb\x88\x8d\xbe\x90K4>\xf5\xe2}>\xdee\xc7>\x91>\x80;\x95.;>\xc8\xe9\x00>\xa9\xccb\xbd\x9aC3\xbeM\xf7\x83\xbe\xa9g\xef\xbeg\xbd\xe7\xbe\xb8\x98\xa3>\xdal\xd4>H\x91\xc2>=\xd4\xd2>(\xc8\xa3>\x98\xc5\xde>UER>f6\x0f\xbe\xa5\xc4V=\xcbw\x1c\xbe9r\x02\xbfE\xb1\x9e>=cX>82\xcb>La\xb7\xbe\x7f\xf8\xf2\xbeT\xcc\xb4>\xa1b\x85\xc4\x02nd\xc3\xc4\x04type\xa3<f4\xc4\x04kind\xc4\x00\xc4\x05shape\x91\n\xc4\x04data\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa5shims\x91\x90'



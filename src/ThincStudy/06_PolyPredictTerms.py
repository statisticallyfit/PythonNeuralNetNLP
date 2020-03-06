# %% markdown
# Source: [https://github.com/explosion/thinc/blob/master/examples/06_predicting_like_terms.ipynb](https://github.com/explosion/thinc/blob/master/examples/06_predicting_like_terms.ipynb)
#
# # Predicting Like Polynomial Terms With Thinc
# **Goal:** teach a model to predict there are $n$ like terms in a polynomial expression. Using [Mathy](https://mathy.ai/) to generate math problems and thinc to build a regression model to output number of like terms in each input problem.
# Example: Only two kinds of terms are "like terms" in $60 + 2x^3 - 6x + x^3 + 17x$. There are $4$ like terms here. (??? how do they count this)

# %% markdown
# ## 1. Sketch a Model
# Before we get started it can be good to have an idea of what input/output shapes we want for our model.
#
# We'll convert text math problems into lists of lists of integers, so our example (X) type can be represented using thinc's `Ints2d` type.
#
# The model will predict how many like terms there are in each sequence, so our output (Y) type can represented with the `Floats2d` type.
#
# Knowing the thinc types we want enables us to create an alias for our model, so we only have to type out the verbose generic signature once.
# %% codecell
from typing import List
from thinc.api import Model
from thinc.types import Ints2d, Floats1d

# Creating aliases for the types
ModelX = Ints2d
ModelY = Floats1d
ModelT = Model[List[ModelX], ModelY]
# %% codecell
ModelX
# %% codecell
ModelY
# %% codecell
ModelT

# %% markdown
# ## 2. Encode Text Inputs
# Must encode the mathy-generated ascii-math problems into integers that the model can process, by building a vocabulary of all the possible characters we will see and map each input character to its index in the list.
#
# For math problems our vocabulary will include all the characters of the alphabet, numbers 0-9, and special characters like $*, -, .$ etc.
# %% codecell
from typing import List

from thinc.api import Model
from thinc.types import Ints2d, Floats1d
from thinc.api import Ops, get_current_ops

vocab: str = " .+-/^*()[]-01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# %% markdown
# Can refer aan element in the string using the `index()` function, to get the index of that element:
# %% codecell
vocab.index('3')
# %% codecell
vocab.index('A')

# %% codecell
def encodeInput(text: str) -> ModelX:
    # Gets the current backend object
    ops: Ops = get_current_ops()
    # To store the indices of the elements in the vocab string.
    indices: List[List[int]] = []

    for letter in text:
        if letter not in vocab:
            raise ValueError(f"'{letter}' missing from vocabulary in text: {text}")

        # After checking that the letter can be found in the vocabulary (to avoid index error) we append the index to the list of indices.
        indices.append([vocab.index(letter)])

    return ops.asarray2i(indices) # converts type to Ints2d

# %% markdown
# **Try It:** Try this out on some fixed data
# %% codecell
outputs: ModelX = encodeInput("4+2")
outputs
# %% codecell
assert outputs[0][0] == vocab.index("4")
assert outputs[1][0] == vocab.index("+")
assert outputs[2][0] == vocab.index("2")


# %% markdown
# ## 3. Generate Math Problems
# Use Mathy to generate random polynomial problems with an arbitrary number of like terms (taken in as an argument). The generated problems act as training data for our model.
# %% codecell
from typing import List, Optional, Set
import random
from mathy.problems import gen_simplify_multiple_terms

def generatePolyProblems(numLikeTerms: int, exclude: Optional[Set[str]] = None) -> List[str]:
    if exclude is None:
        exclude = set()

    problems: List[str] = []

    while len(problems) < numLikeTerms:
        text, complexity = gen_simplify_multiple_terms(
            num_terms = random.randint(a = 2, b = 6), # generate a rand integer in range [a,b] inclusive
            noise_probability = 1.0,
            noise_terms = random.randint(2,10),
            op = ["+", "-"]
            )

        assert text not in exclude, "duplicate problem was generated!"

        exclude.add(text) # add to memory, so that no duplicate can later be generated
        problems.append(text) # add this problem to list of problems, for later usage

    return problems
# %% markdown
# **Try It:**
# %% codecell
generatePolyProblems(numLikeTerms = 10)

# %% markdown
# ## 4. Count Like Terms
# Need a function that can count like terms in each problem and return the value for later use as a label.
#
# Going to use a few helpers from mathy to enumerate the terms and compare them to see if they are like.
# %% codecell
from typing import Optional, List, Dict
from mathy import MathExpression, ExpressionParser, get_terms, get_term_ex, TermEx
from mathy.problems import mathy_term_string


parser: ExpressionParser = ExpressionParser()


def countLikeTerms(inputProblem: str) -> int:

    expression: MathExpression = parser.parse(input_text = inputProblem)
    termNodes: List[MathExpression] = get_terms(expression = expression)
    nodeGroups: Dict[str, List[MathExpression]] = {}

    for termNode in termNodes:
        ex: Optional[TermEx] = get_term_ex(node = termNode)

        # Testing that the unknown variable is not None
        assert ex is not None, f"invalid expression {termNode}"

        key: str = mathy_term_string(variable = ex.variable,
                                exponent = ex.exponent)

        # Adding the term node to thenode groups at  key value.
        if key == "":
            key = "const"
        if key not in nodeGroups:
            # Adding term node list of math exprs for the first time in the dict of node groups at the value key.
            nodeGroups[key] = [termNode]
        else:
            # If the key may be in node groups dict, then append the current term node at that key value.
            nodeGroups[key].append(termNode)

        # Initializing number of like terms
    numLikeTerms: int = 0

    for key, termNode in nodeGroups.items():
        if len(termNode) <= 1:
            continue

        # Value of nodeGroups is list of math expressions, so now that must contain the number of like terms.
        numLikeTerms += len(termNode)

    return numLikeTerms

# %% codecell
# Seeing the expression parser:
parser

# %% markdown
# **Try It:**
# %% codecell
assert countLikeTerms("4x - 2y + q") == 0
# %% codecell
assert countLikeTerms("x + x + z") == 2
# %% codecell
assert countLikeTerms("4x + 2x - x + 7") == 3

# %% markdown
# Counts TOTAL number of terms that are LIKE (so since there are 3 terms with exponent two, 2 terms with exponent one, 2 terms with exponent 3, and only 1 term with exponent six, that results in $3 + 2 + 2 \rightarrow 7$ like terms)
# %% codecell
assert countLikeTerms("7x^3 + 8x^6 + 2x + 3x^2 + x^2 + 9x + 9x^3 + 7x^2") == 7

# %% markdown
# ## 5. Generate Problem / Answer Pairs
# Now that we can gernate problems, count number of like terms in them, and encode their text to integers, we have the pieces required to generate random problems and answers that we can train a neural network with.
#
# Let's write a function that returns a tuple of: the problem text, its encoded example form, and the output label.
# %% codecell
from typing import Tuple
from thinc.api import Ops, get_current_ops

def toExample(inputProblem: str) -> Tuple[str, ModelX, ModelY]:
    ops: Ops = get_current_ops()
    encodedInput: ModelX = encodeInput(text = inputProblem)
    numLikeTerms: int = countLikeTerms(inputProblem = inputProblem)

    # NOTE: the Y vector of type ModelY is just made by vectorizing the number (int) of like terms in the given input problem.
    return inputProblem, encodedInput, ops.asarray1f([numLikeTerms])


# %% markdown
# **Try It:**
# %% codecell
inputProblem, X, Y = toExample(inputProblem = "x + 2x")
# %% codecell
inputProblem
# %% codecell
X
# %% codecell
Y

# %% codecell
assert X[0] == 46
assert X[0] == vocab.index("x")
assert vocab.find('x') == 46

# %% codecell
assert X[1] == 0
assert vocab.find(' ') == 0
assert X[1] == vocab.index(' ')

# %% codecell
assert X[2] == 2
assert vocab.find('+') == 2
assert vocab.index('+') == X[2]

# %% codecell
assert X[4] == 14
assert vocab.find('2') == 14
assert vocab.index('2') == X[4]

# %% markdown
# ## 6. Build a Model
# Now that we can generate X, Y values, must define the model and verify it can process a single input / output (using Thinc and the `define_operators` context manager to connect pieces using overloaded operators for `chain` and `clone` operations)
# %% codecell
from typing import List
from thinc.model import Model, OutT
from thinc.api import concatenate, chain, clone, list2ragged
from thinc.api import reduce_sum, Mish, with_array, Embed, residual


def buildModel(numHiddenLayers: int, dropout: float = 0.1) -> ModelT:

    with Model.define_operators({">>": chain, "|": concatenate, "**": clone}):

        model: Model = (
            # Iterate over each element in the batch
            with_array(
                # Embed the vocab indices
                layer = Embed(nO = numHiddenLayers, nV = len(vocab), column = 0)
                # Activate each batch of embedding sequences separately first
                # Mish = dense layer with Mish activation
                >> Mish(nO = numHiddenLayers, dropout = dropout)
            )

            # Convert to ragged so we can use the reduction layers
            >> list2ragged()

            # Sum the features for each batch input
            >> reduce_sum()

            # Process with a small resnet
            >> residual(layer = Mish(nO = numHiddenLayers, normalize = True)) ** 4

            # Convert (batchSize, numHiddenLayers) ==> (batchSize, 1)
            >> Mish(nO = 1) # todo: so Mish nO acts only on the second dimension? never on the first dimension (batchSize)?
        )

        return model


# %% markdown
# **Try It:** passing an example through the model to make sure the sizes are being built correctly.
# %% codecell
inputProblem, X, Y = toExample(inputProblem = "14x + 2y - 3x + 7x")
# %% codecell
inputProblem
# %% codecell
X
assert X.shape == (18, 1)
assert X.ndim == 2
# %% codecell
Y
# %% codecell
polyModel: Model = buildModel(numHiddenLayers = 12)
polyModel
# %% codecell
polyModel.initialize(X = [X], Y = polyModel.ops.asarray(Y, dtype="f"))
# %% codecell
from thinc.model import OutT


yPred: OutT = polyModel.predict(X = [X]) # calls model's forward function
yPred
# %% codecell
assert yPred.shape == (1,1)
assert yPred.ndim == 2


# %% markdown
# 7. Generate Training Datasets
# Now that we can generate examples and have model to process them it is time to generate random unique training and evaluation datasets.
#
# For this, need to write another helper function to generate $n$ training examples and respects an exclude list to avoid overlaping examples from the training and test sets.
# %% codecell
from typing import Tuple, Optional, Set, List

# Assing a type alias
DatasetTuple = Tuple[List[str], List[ModelX], List[ModelY]]

def generateDataset(size: int, exclude: Optional[Set[str]] = None) -> DatasetTuple:

    ops: Ops = get_current_ops()
    inputProblemList: List[str] = generatePolyProblems(numLikeTerms = size, exclude = exclude)
    encodedExamples: List[ModelX] = []
    outputLabels: List[ModelY] = []

    for i, inputProb in enumerate(inputProblemList):
        inputProb, x, y = toExample(inputProblem = inputProb)
        encodedExamples.append(x)
        outputLabels.append(y)

    return inputProb, encodedExamples, outputLabels


# %% markdown
# **Try It:** Generate a small dataset
# %% codecell
inputProblems, x, y = generateDataset(size = 10)
assert len(inputProblems) == 10
assert len(x) == 10
assert len(y) == 10
# %% codecell
x
# %% codecell
y
# %% codecell
inputProblems


# %% markdown
# ## 8. Evaluate Model Performance
# Need to create a function to check a trained model against a given dataset and return a $0-1$ score of how accurate it was.
#
# Use this function to print score as training progresses and print final test prediction at the end of training.
# %% codecell
from typing import List
from wasabi import msg


def evaluateModel(model: ModelT, *, isPrintProblems: bool = False, inputProblemList: List[str],
                  X: List[ModelX], Y: List[ModelY]):

    yEval: ModelY = model.predict(X = X)
    numCorrect: int = 0
    numPrint: int = 12

    if isPrintProblems:
        msg.divider(f"eval samples max({numPrint}")


    for inputProb, yAnswer, yGuess in zip(inputProblemList, Y, yEval):
        # Polishing the guess for Y
        yGuess = round(float(yGuess))
        # Record if the guess was correct
        isCorrect: bool = yGuess == int(yAnswer)
        printFn = msg.fail

        if isCorrect:
            numCorrect += 1
            printFn = msg.good
        # If not correct ...
        if isPrintProblems and numPrint > 0:
            numPrint -= 1
            printFn(f"Answer[{int(yAnswer[0])}] Guess[{yGuess}] Text: {inputProb}")

    if isPrintProblems:
        print(f"Model predicted {numCorrect} out of {len(X)} correctly.")

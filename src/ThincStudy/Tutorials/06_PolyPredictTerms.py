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

from numpy.core._multiarray_umath import ndarray
from thinc.api import Model
from thinc.types import Ints2d, Floats1d, SizedGenerator

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
# **Try It**
#
# Try this out on some fixed data
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
# **Try It**
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
# **Try It**
# %% codecell
assert countLikeTerms("4x - 2y + q") == 0
# %% codecell
assert countLikeTerms("x + x + z") == 2
# %% codecell
assert countLikeTerms("4x + 2x - x + 7") == 3

# %% markdown
# Counts TOTAL number of terms that are LIKE (so since there are $3$ terms with exponent two, $2$ terms with exponent one, $2$ terms with exponent three, and only $1$ term with exponent six, that results in $3 + 2 + 2 \rightarrow 7$ like terms)
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
# **Try It**
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
# **Try It**
#
# Passing an example through the model to make sure the sizes are being built correctly.
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
# ## 7. Generate Training Datasets
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

    return inputProblemList, encodedExamples, outputLabels


# %% markdown
# **Try It**
#
# Generate a small dataset
# %% codecell
import numpy as np

inputProblems, x, y = generateDataset(size = 10)
inputProblems
# %% codecell
assert len(inputProblems) == 10
assert len(x) == 10
assert len(y) == 10
# %% codecell
arrX: ndarray = np.array(x)
print("shape = {}".format(arrX.shape))
print("ndim = {}".format(arrX.ndim))
# %% codecell
x
# %% codecell
arrY: ndarray = np.array(y)
print("shape = {}".format(arrY.shape))
print("ndim = {}".format(arrY.ndim))
# %% codecell
y



# %% markdown
# ## 8. Evaluate Model Performance
# Need to create a function to check a trained model against a given dataset and return a $0-1$ score of how accurate it was.
#
# Use this function to print score as training progresses and print final test prediction at the end of training.
# %% codecell
from typing import List
from wasabi import msg


def evaluateModel(model: ModelT, *,
                  isPrintProblems: bool = False,
                  inputProblemList: List[str],
                  X: List[ModelX],
                  Y: List[ModelY]) -> float:

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

    return numCorrect / len(X)

# %% markdown
# **Try It**
# Trying it out with an untrained model, which will do a really bad job.
# %% codecell
inputProblemList, X, Y = generateDataset(size = 128)
inputProblemList
# %% codecell
# X # not showing, too long
arrX: ndarray = np.array(X)
print("shape = {}".format(arrX.shape))
print("ndim = {}".format(arrX.ndim))
# %% codecell
arrY: ndarray = np.array(Y)
print("shape = {}".format(arrY.shape))
print("ndim = {}".format(arrY.ndim))

# %% markdown
# Build and initialize the model:
# %% codecell
# Does it matter if type is Model or ModelT???
polyModel: ModelT = buildModel(numHiddenLayers = 12)
polyModel
# %% codecell
type(polyModel)
# %% codecell
polyModel.initialize(X = X, Y = polyModel.ops.asarray(Y, dtype="f"))
polyModel
# %% codecell
evaluateModel(model = polyModel, inputProblemList = inputProblemList, X = X, Y = Y)

# %% markdown
# ## 9. Train and Evaluate a Model
# The final helper function is to help train and evaluate a model given two input datasets. The function does these things ...
#
# 1. Create an Adam optimizer for minimizing the model's prediction error.
# 2. Loop over the given training dataset (epoch) number of times.
# 3. For each epoch, make batches of `(batchSize)` examples. For each batch $X$, predict the number of like terms $Yh$ and subtract the known answers $Y$ to get the prediction error. Update the model using the optimizer with the calculated error.
# 4. After each epoch, check the model performance against the evaluation dataset.
# 5. Save the model weights for the best score out of all the training epochs.
# 6. After the training is done, restore the best model and print results from the evaluation set.
# %% codecell
from thinc.api import Adam

# Typing help
from thinc.types import SizedGenerator # a generator with length that can repeatedly call the generator function.
from thinc.optimizers import Optimizer

from wasabi import msg
import numpy
from tqdm.auto import tqdm

def trainAndEvaluate(model: ModelT,
                     trainTuple: DatasetTuple, evalTuple: DatasetTuple, *,
                     learnRate: float = 3e-3,
                     batchSize: int = 64,
                     numEpochs: int = 48) -> float:

    # Unpack the tuple values
    (trainTexts, trainX, trainY) = trainTuple
    (evalTexts, evalX, evalY) = evalTuple

    # Send a message to console
    msg.divider("Train and Evaluate Model...")
    msg.info(f"Batch size = {batchSize}\tEpochs = {numEpochs}\tLearning Rate = {learnRate}")



    # Part 1: Training the model
    print(f"Training the Model ...")

    # Step 1: create optimizer
    adamOptimizer: Optimizer = Adam(learn_rate = learnRate)
    # Variables for recording model info
    bestScore: float = 0.0
    bestModel: Optional[bytes] = None

    # Step 2: looping over the training dataset numEpoch number of times.
    for epoch in range(numEpochs):
        loss: float = 0.0

        # Step 3: make batches of batchSized examples.
        batches: SizedGenerator = model.ops.multibatch(batchSize,
                                                       trainX,
                                                       trainY,
                                                       shuffle = True)

        # Step 3 ... For each batch X, ...
        for X, Y in tqdm(batches, leave = False, unit = "batches"):

            # Converting the Y array to another type
            Y = model.ops.asarray(Y, dtype = "float32")

            # Step 3: For each batch X, predict the number of like terms (Yh)
            Yh, backprop = model.begin_update(X = X)
            # Step 3: subtract the correct number of like terms (Y) from the predicted number of like terms (Yh) to get the error
            error = Yh - Y
            # Step 3: Backpropagate the errors over the computational graph to update gradients.
            backprop(error)
            # Step 3: Update the loss with the error
            loss += (error ** 2).sum()
            # Step 3: Update the model using the optimizer with the calculated error
            # todo: how is the loss above incorporated into updating the model here?
            model.finish_update(optimizer = adamOptimizer)

        # Step 4: Preparing to check model performance against evaluation dataset.
        score: float = evaluateModel(model = model,
                                     inputProblemList = evalTexts,
                                     X = evalX,
                                     Y = evalY)

        # Record the best score
        if score > bestScore:
            bestModel = model.to_bytes() # todo why convret to bytes?
            bestScore = score
        print(f"Epoch: {epoch}\tScore: {score:.2f}\tLoss: {loss:.2f}")

    if bestModel is not None:
        model.from_bytes(bytes_data = bestModel) # todo why conversion to bytes above??


    # Part 2: Now evaluating the model
    print(f"Evaluating with Best Model...")

    score = evaluateModel(model = model,
                          inputProblemList = evalTexts,
                          isPrintProblems=True,
                          X = evalX,
                          Y = evalY)

    print(f"Final score: {score}")

    return score

# %% markdown
# Generate the dataset first so we can iterate on the model without having to spend time generating examples for each run. This also ensures we have the same dataset across different model runs, to make it easier to compare performance.
# %% codecell
trainSize: int = 1024 * 8
testSize: int = 2048
seenTexts: Set[str] = set()

# DatasetTuple = Tuple[List[str], List[ModelX], List[ModelY]]

with msg.loading(f"Generating train dataset with {trainSize} examples ..."):
    trainDataset: DatasetTuple = generateDataset(size = trainSize, exclude = seenTexts)

msg.good(f"Train set created with {trainSize} examples.")

with msg.loading(f"Generating eval dataset with {testSize} examples ..."):
    evalDataset: DatasetTuple = generateDataset(size = testSize, exclude = seenTexts)

msg.good(f"Eval set created with {testSize} examples.")

# DatasetTuple = Tuple[List[str], List[ModelX], List[ModelY]]
# Getting second in first tuple (List[ModelX]) and selecting three elements from it, which we shall use for shape inference when initializing the model.
initX: List[ModelX] = trainDataset[1][:2]
initY: List[ModelY] = trainDataset[2][:2]

# %% markdown
# Build, train, evaluate the model:
# %% codecell
mainPolyModel: ModelT = buildModel(numHiddenLayers= 64)
mainPolyModel
# %% codecell
mainPolyModel.initialize(X = initX, Y = initY) # shape inference.
# %% codecell
trainAndEvaluate(model = mainPolyModel,
                 trainTuple = trainDataset,
                 evalTuple = evalDataset,
                 learnRate = 2e-3,
                 batchSize = 64,
                 numEpochs = 16)

# %% markdown
# # TODO !
# ## Intermediate Exercise: Towards Fewer Epochs
# The model we built can train up to ~80% given 100 or more epochs. Improve the model architecture so that it trains to a similar accuracy while requiring fewer epochs or a smaller dataset size.
# %% codecell
from typing import List
from thinc.model import Model
from thinc.types import Array2d, Array1d
from thinc.api import chain, clone, list2ragged, reduce_mean, Mish, with_array, Embed, residual

def customModel(numHiddenLayers: int, dropout: float = 0.1) -> Model[List[Array2d], Array2d]:
    # Put your custom architecture here
    return buildModel(numHiddenLayers = numHiddenLayers, dropout = dropout)

model = customModel(64)
model.initialize(X = initX, Y = initY)
trainAndEvaluate(model = mainPolyModel,
                 trainTuple = trainDataset,
                 evalTuple = evalDataset,
                 learnRate = 2e-3,
                 batchSize = 64,
                 numEpochs = 16)

# %% markdown
# # TODO !
# ## Advanced Exercise: Encode with BiLSTM
# Rewrite the model to encode the whole expression with a BiLSTM, and then generate pairs of terms, using the BiLSTM vectors. Over each pair of terms, predict whether the terms are alike or unlike.
# %% codecell
from dataclasses import dataclass
from thinc.types import Array2d, Ragged
from thinc.model import Model


@dataclass
class Comparisons:
    data: Array2d  # Batch of vectors for each pair
    indices: Array2d  # Int array of shape (N, 3), showing the (batch, term1, term2) positions

def pairify() -> Model[Ragged, Comparisons]:
    """Create pair-wise comparisons for items in a sequence. For each sequence of N
    items, there will be (N**2-N)/2 comparisons."""
    ...

def predictOverPairs(model: Model[Array2d, Array2d]) -> Model[Comparisons, Comparisons]:
    """Apply a prediction model over a batch of comparisons. Outputs a Comparisons
    object where the data is the scores. The prediction model should predict over
    two classes, True and False."""
    ...

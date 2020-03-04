# %% markdown
# # Predicting Like Polynomial Terms
#
# Remember in Algebra how you had to combine "like terms" to simplify problems?
#
# You'd see expressions such as `60 + 2x^3 - 6x + x^3 + 17x` in which there are **5** total terms but only **4** are "like terms".
#
# `2x^3` and `x^3` are like, and `-6x` and `17x` are like, while `60` doesn't have any like siblings.
#
# Can we teach a model to predict that there are `4` like terms in the above expression?
#
# Let's give it a shot using [Mathy](https://mathy.ai) to generate math problems and [thinc](https://github.com/explosion/thinc) to build a regression model that outputs the number of like terms in each input problem.
# %% codecell
!pip install "thinc>=8.0.0a0" mathy
# %% markdown
# ### Sketch a Model
#
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

ModelX = Ints2d
ModelY = Floats1d
ModelT = Model[List[ModelX], ModelY]
# %% markdown
# ### Encode Text Inputs
#
# Mathy generates ascii-math problems and we have to encode them into integers that the model can process.
#
# To do this we'll build a vocabulary of all the possible characters we'll see, and map each input character to its index in the list.
#
# For math problems our vocabulary will include all the characters of the alphabet, numbers 0-9, and special characters like `*`, `-`, `.`, etc.
# %% codecell
from typing import List
from thinc.api import Model
from thinc.types import Ints2d, Floats1d
from thinc.api import Ops, get_current_ops

vocab = " .+-/^*()[]-01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def encode_input(text: str) -> ModelX:
    ops: Ops = get_current_ops()
    indices: List[List[int]] = []
    for c in text:
        if c not in vocab:
            raise ValueError(f"'{c}' missing from vocabulary in text: {text}")
        indices.append([vocab.index(c)])
    return ops.asarray2i(indices)
# %% markdown
# #### Try It
#
# Let's try it out on some fixed data to be sure it works.
# %% codecell
outputs = encode_input("4+2")
assert outputs[0][0] == vocab.index("4")
assert outputs[1][0] == vocab.index("+")
assert outputs[2][0] == vocab.index("2")
print(outputs)
# %% markdown
# ### Generate Math Problems
#
# We'll use Mathy to generate random polynomial problems with a variable number of like terms. The generated problems will act as training data for our model.
# %% codecell
from typing import List, Optional, Set
import random
from mathy.problems import gen_simplify_multiple_terms

def generate_problems(number: int, exclude: Optional[Set[str]] = None) -> List[str]:
    if exclude is None:
        exclude = set()
    problems: List[str] = []
    while len(problems) < number:
        text, complexity = gen_simplify_multiple_terms(
            random.randint(2, 6),
            noise_probability=1.0,
            noise_terms=random.randint(2, 10),
            op=["+", "-"],
        )
        assert text not in exclude, "duplicate problem generated!"
        exclude.add(text)
        problems.append(text)
    return problems
# %% markdown
# #### Try It
# %% codecell
generate_problems(10)
# %% markdown
# ### Count Like Terms
#
# Now that we can generate input problems, we'll need a function that can count the like terms in each one and return the value for use as a label.
#
# To accomplish this we'll use a few helpers from mathy to enumerate the terms and compare them to see if they're like.
# %% codecell
from typing import Optional, List, Dict
from mathy import MathExpression, ExpressionParser, get_terms, get_term_ex, TermEx
from mathy.problems import mathy_term_string

parser = ExpressionParser()

def count_like_terms(input_problem: str) -> int:
    expression: MathExpression = parser.parse(input_problem)
    term_nodes: List[MathExpression] = get_terms(expression)
    node_groups: Dict[str, List[MathExpression]] = {}
    for term_node in term_nodes:
        ex: Optional[TermEx] = get_term_ex(term_node)
        assert ex is not None, f"invalid expression {term_node}"
        key = mathy_term_string(variable=ex.variable, exponent=ex.exponent)
        if key == "":
            key = "const"
        if key not in node_groups:
            node_groups[key] = [term_node]
        else:
            node_groups[key].append(term_node)
    like_terms = 0
    for k, v in node_groups.items():
        if len(v) <= 1:
            continue
        like_terms += len(v)
    return like_terms
# %% markdown
# #### Try It
# %% codecell
assert count_like_terms("4x - 2y + q") == 0
assert count_like_terms("x + x + z") == 2
assert count_like_terms("4x + 2x - x + 7") == 3
# %% markdown
# ### Generate Problem/Answer pairs
#
# Now that we can generate problems, count the number of like terms in them, and encode their text into integers, we have the pieces required to generate random problems and answers that we can train a neural network with.
#
# Let's write a function that will return a tuple of: the problem text, its encoded example form, and the output label.
# %% codecell
from typing import Tuple
from thinc.api import Ops, get_current_ops

def to_example(input_problem: str) -> Tuple[str, ModelX, ModelY]:
    ops: Ops = get_current_ops()
    encoded_input = encode_input(input_problem)
    like_terms = count_like_terms(input_problem)
    return input_problem, encoded_input, ops.asarray1f([like_terms])
# %% markdown
# #### Try It
# %% codecell
text, X, Y = to_example("x+2x")
assert text == "x+2x"
assert X[0] == vocab.index("x")
assert Y[0] == 2
print(text, X, Y)
# %% markdown
# ### Build a Model
#
# Now that we can generate X/Y values, let's define our model and verify that it can process a single input/output.
#
# For this we'll use Thinc and the `define_operators` context manager to connect the pieces together using overloaded operators for `chain` and `clone` operations.
# %% codecell
from typing import List
from thinc.model import Model
from thinc.api import concatenate, chain, clone, list2ragged
from thinc.api import reduce_sum, Mish, with_array, Embed, residual

def build_model(n_hidden: int, dropout: float = 0.1) -> ModelT:
    with Model.define_operators({">>": chain, "|": concatenate, "**": clone}):
        model = (
            # Iterate over each element in the batch
            with_array(
                # Embed the vocab indices
                Embed(n_hidden, len(vocab), column=0)
                # Activate each batch of embedding sequences separately first
                >> Mish(n_hidden, dropout=dropout)
            )
            # Convert to ragged so we can use the reduction layers
            >> list2ragged()
            # Sum the features for each batch input
            >> reduce_sum()
            # Process with a small resnet
            >> residual(Mish(n_hidden, normalize=True)) ** 4
            # Convert (batch_size, n_hidden) to (batch_size, 1)
            >> Mish(1)
        )
    return model
# %% markdown
# #### Try It
#
# Let's pass an example through the model to make sure we have all the sizes right.
# %% codecell
text, X, Y = to_example("14x + 2y - 3x + 7x")
m = build_model(12)
m.initialize([X], m.ops.asarray(Y, dtype="f"))
mY = m.predict([X])
print(mY.shape)
assert mY.shape == (1, 1)
# %% markdown
# ### Generate Training Datasets
#
# Now that we can generate examples and we have a model that can process them, let's generate random unique training and evaluation datasets.
#
# For this we'll write another helper function that can generate (n) training examples and respects an exclude list to avoid letting examples from the training/test sets overlap.
# %% codecell
from typing import Tuple, Optional, Set, List

DatasetTuple = Tuple[List[str], List[ModelX], List[ModelY]]

def generate_dataset(
    size: int,
    exclude: Optional[Set[str]] = None,
) -> DatasetTuple:
    ops: Ops = get_current_ops()
    texts: List[str] = generate_problems(size, exclude=exclude)
    examples: List[ModelX] = []
    labels: List[ModelY] = []
    for i, text in enumerate(texts):
        text, x, y = to_example(text)
        examples.append(x)
        labels.append(y)

    return texts, examples, labels
# %% markdown
# #### Try It
#
# Generate a small dataset to be sure everything is working as expected
# %% codecell
texts, x, y = generate_dataset(10)
assert len(texts) == 10
assert len(x) == 10
assert len(y) == 10
# %% markdown
# ### Evaluate Model Performance
#
# We're almost ready to train our model, we just need to write a function that will check a given trained model against a given dataset and return a 0-1 score of how accurate it was.
#
# We'll use this function to print the score as training progresses and print final test predictions at the end of training.
# %% codecell
from typing import List
from wasabi import msg

def evaluate_model(
    model: ModelT,
    *,
    print_problems: bool = False,
    texts: List[str],
    X: List[ModelX],
    Y: List[ModelY],
):
    Yeval = model.predict(X)
    correct_count = 0
    print_n = 12
    if print_problems:
        msg.divider(f"eval samples max({print_n})")
    for text, y_answer, y_guess in zip(texts, Y, Yeval):
        y_guess = round(float(y_guess))
        correct = y_guess == int(y_answer)
        print_fn = msg.fail
        if correct:
            correct_count += 1
            print_fn = msg.good
        if print_problems and print_n > 0:
            print_n -= 1
            print_fn(f"Answer[{int(y_answer[0])}] Guess[{y_guess}] Text: {text}")
    if print_problems:
        print(f"Model predicted {correct_count} out of {len(X)} correctly.")
    return correct_count / len(X)

# %% markdown
# #### Try It
#
# Let's try it out with an untrained model and expect to see a really sad score.
# %% codecell
texts, X, Y = generate_dataset(128)
m = build_model(12)
m.initialize(X, m.ops.asarray(Y, dtype="f"))
# Assume the model should do so poorly as to round down to 0
assert round(evaluate_model(m, texts=texts, X=X, Y=Y)) == 0
# %% markdown
# ### Train/Evaluate a Model
#
# The final helper function we need is one to train and evaluate a model given two input datasets.
#
# This function does a few things:
#
#  1. Create an Adam optimizer we can use for minimizing the model's prediction error.
#  2. Loop over the given training dataset (epoch) number of times.
#  3. For each epoch, make batches of (batch_size) examples. For each batch(X), predict the number of like terms (Yh) and subtract the known answers (Y) to get the prediction error. Update the model using the optimizer with the calculated error.
#  5. After each epoch, check the model performance against the evaluation dataset.
#  6. Save the model weights for the best score out of all the training epochs.
#  7. After all training is done, restore the best model and print results from the evaluation set.
# %% codecell
from thinc.api import Adam
from wasabi import msg
import numpy
from tqdm.auto import tqdm

def train_and_evaluate(
    model: ModelT,
    train_tuple: DatasetTuple,
    eval_tuple: DatasetTuple,
    *,
    lr: float = 3e-3,
    batch_size: int = 64,
    epochs: int = 48,
) -> float:
    (train_texts, train_X, train_y) = train_tuple
    (eval_texts, eval_X, eval_y) = eval_tuple
    msg.divider("Train and Evaluate Model")
    msg.info(f"Batch size = {batch_size}\tEpochs = {epochs}\tLearning Rate = {lr}")

    optimizer = Adam(lr)
    best_score: float = 0.0
    best_model: Optional[bytes] = None
    for n in range(epochs):
        loss = 0.0
        batches = model.ops.multibatch(batch_size, train_X, train_y, shuffle=True)
        for X, Y in tqdm(batches, leave=False, unit="batches"):
            Y = model.ops.asarray(Y, dtype="float32")
            Yh, backprop = model.begin_update(X)
            err = Yh - Y
            backprop(err)
            loss += (err ** 2).sum()
            model.finish_update(optimizer)
        score = evaluate_model(model, texts=eval_texts, X=eval_X, Y=eval_y)
        if score > best_score:
            best_model = model.to_bytes()
            best_score = score
        print(f"{n}\t{score:.2f}\t{loss:.2f}")

    if best_model is not None:
        model.from_bytes(best_model)
    print(f"Evaluating with best model")
    score = evaluate_model(
        model, texts=eval_texts, print_problems=True, X=eval_X, Y=eval_y
    )
    print(f"Final Score: {score}")
    return score

# %% markdown
# We'll generate the dataset first, so we can iterate on the model without having to spend time generating examples for each run. This also ensures we have the same dataset across different model runs, to make it easier to compare performance.
# %% codecell
train_size = 1024 * 8
test_size = 2048
seen_texts: Set[str] = set()
with msg.loading(f"Generating train dataset with {train_size} examples..."):
    train_dataset = generate_dataset(train_size, seen_texts)
msg.good(f"Train set created with {train_size} examples.")
with msg.loading(f"Generating eval dataset with {test_size} examples..."):
    eval_dataset = generate_dataset(test_size, seen_texts)
msg.good(f"Eval set created with {test_size} examples.")
init_x = train_dataset[1][:2]
init_y = train_dataset[2][:2]
# %% markdown
# Finally, we can build, train, and evaluate our model!
# %% codecell
model = build_model(64)
model.initialize(init_x, init_y)
train_and_evaluate(
    model, train_dataset, eval_dataset, lr=2e-3, batch_size=64, epochs=16
)
# %% markdown
# ### Intermediate Exercise
#
# The model we built can train up to ~80% given 100 or more epochs. Improve the model architecture so that it trains to a similar accuracy while requiring fewer epochs or a smaller dataset size.
# %% codecell
from typing import List
from thinc.model import Model
from thinc.types import Array2d, Array1d
from thinc.api import chain, clone, list2ragged, reduce_mean, Mish, with_array, Embed, residual

def custom_model(n_hidden: int, dropout: float = 0.1) -> Model[List[Array2d], Array2d]:
    # Put your custom architecture here
    return build_model(n_hidden, dropout)

model = custom_model(64)
model.initialize(init_x, init_y)
train_and_evaluate(
    model, train_dataset, eval_dataset, lr=2e-3, batch_size=64, epochs=16
)
# %% markdown
# ### Advanced Exercise
#
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

def predict_over_pairs(model: Model[Array2d, Array2d]) -> Model[Comparisons, Comparisons]:
    """Apply a prediction model over a batch of comparisons. Outputs a Comparisons
    object where the data is the scores. The prediction model should predict over
    two classes, True and False."""
    ...

# %% markdown
# # Tutorial 1: FARM Building Blocks
#
# ## TASK 1: Text Classification
#
# GermEval 2018 (GermEval2018) (https://projects.fzai.h-da.de/iggsa/) is an open data set containing texts that need to be classified by whether they are offensive or not. There are a set of coarse and fine labels, but here we will only be looking at the coarse set which labels each example as either OFFENSE or OTHER. To tackle this task, we are going to build a classifier that is composed of Google's BERT language model and a feed forward neural network prediction head.
# %% codecell
import torch
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import MLFlowLogger

from typing import *
# %% markdown
# ### STEP 1: Setup
# Adjust the working directory to the current folder path
# %% codecell
import os
os.getcwd()

os.chdir("/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP/src/FARMHaystackStudy/")

# %% markdown
# Setup to be able to import my util functions in other folders:
# %% codecell
import sys

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

UTIL_PATH: str = PATH + "/src/utils/ModelUtil/"

FARM_PATH: str = PATH + "/src/FARMHaystackStudy/TutorialsFARM/"


sys.path.append(PATH)
sys.path.append(UTIL_PATH)
sys.path.append(FARM_PATH)

sys.path

# %% markdown
# Farm allows simple logging of many parameters & metrics. Let's use MLflow framework to track our experiment ...
# %% codecell
mlLogger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
mlLogger.init_experiment(experiment_name="Public_FARM", run_name="Tutorial1_Colab")

# %% codecell
# We need to fetch the right device to drive the growth of our model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Devices available: {}".format(device))

# %% markdown
# ### STEP 2: Data Handling
# Here we initialize a tokenizer to preprocess text. This is the BERT Tokenizer which uses byte pair encoding method (currently loaded with a German model)
# %% codecell
tokenizer = Tokenizer.load(pretrained_model_name_or_path = "bert-base-german-cased",
                           do_lower_case = False)
# %% codecell
tokenizer
# %% markdown
# To prepare the data for the model, we need a set of functions to transform data files into PyTorch Datasets.
# We group these together in Processor objects.
# We will need a new Processor object for each new source of data.
# The abstract class is in `farm.data_handling.processor.Processor`
# %% codecell
LABEL_LIST: List[str] = ["OTHER", "OFFENSE"]
BATCH_SIZE: int = 32
MAX_SEQ_LEN: int = 128
METRIC: str = "f1_macro"

processor = TextClassificationProcessor(tokenizer = tokenizer,
                                        max_seq_len = MAX_SEQ_LEN,
                                        data_dir = "data/germeval18",
                                        label_list = LABEL_LIST,
                                        metric = METRIC,
                                        label_column_name = "coarse_label")

# %% markdown
# We need a `DataSilo` to keep our train, dev, and test sets separate. The `DataSilo` will call the functions in the `Processor` to generate these sets.
#
# From the `DataSilo` we can fetch a PyTorch `DataLoader` object which will be passed on to the model.
# %% codecell
dataSilo  = DataSilo(processor = processor,
                     batch_size = BATCH_SIZE)
# %% markdown
# ### STEP 3: Modeling
# In FARM, we make a strong distinction between the language model and prediction head so that you can mix and match different building blocks for your needs.
#
# For example, in the transfer learning paradigm, you might have the one language model that you will be using for both document classification and NER. Or you perhaps you have a pretrained language model which you would like to adapt to your domain, then use for a downstream task such as question answering.
#
# All this is possible within FARM and requires only the replacement of a few modular components, as we shall see below.
#
# Let's first have a look at how we might set up a model.
#
# #### Language Model
# * The language model is the foundation on which modern NLP systems are built. They encapsulate a general understanding of sentence semantics and are not specific to any one task.
# * Here we are using Google's BERT model as implemented by HuggingFace. The model being loaded is a German model that we trained.
# * Can also change the MODEL_NAME_OR_PATH to point to a BERT model that you have saved or download one connected to the HuggingFace repository. See https://huggingface.co/models for a list of available models
# %% codecell
MODEL_NAME_OR_PATH = "bert-base-german-cased"
languageModel = LanguageModel.load(MODEL_NAME_OR_PATH)
# %% codecell
languageModel



# %% markdown
# #### Prediction Head
# * A Prediction head is a model that processes the output of the language model for a specific task. It will look different depending on the task (text classification, NER, QA ...)
# * Prediction heads should generate logits over the available prediction classes and contain methods to convert these logits to losses or predictions.
#
# Here we use `TextClassificationHead` prediction head which receives a single fixed length sentence vector and processes it using a feed forward neural network.
# * NOTE: `layer_dims` is a list of dimensions = `[input_dims, hidden_1_dims, hidden_2_dims, ..., output_dims]`
# * Using a single layer network that takes in a vector length 768 (default size of BERT output), and the prediction head outputs a vector of length 2 (number of classes in the GermEval18 coarse dataset)
# %% codecell
predictionHead = TextClassificationHead(num_labels = len(LABEL_LIST))
# %% codecell
predictionHead
# %% markdown
# #### Adaptive Model
# The language model and prediction head are coupled together in the `AdaptiveModel`, which is a class that takes care of model saving and loading. Also coordinates cases where there is more than one prediction head.
#
# Its parameter `EMBEDS_DROPOUT_PROB` is the probability that an element of the output vector from the language model will be set to zero.
# %% codecell
EMBEDS_DROPOUT_PROB: float = 0.1

model = AdaptiveModel(language_model = languageModel,
                      prediction_heads = [predictionHead],
                      embeds_dropout_prob = EMBEDS_DROPOUT_PROB,
                      lm_output_types = ["per_sequence"],
                      device = device)
# %% codecell

model

# %% markdown
# ### STEP 4: Training
# Here we initialize a BERT Adam optimizer with linear warmup and warmdown. Can set learning rate, warmup proportion and number of epochs to train for.
# %% codecell
LEARNING_RATE: float = 2e-5
NUM_EPOCHS: int = 1

modelOpt, optimizer, learnRateSchedule = initialize_optimizer(
          model = model,
          device = device,
          learning_rate = LEARNING_RATE,
          n_batches = len(dataSilo.loaders["train"]),
          n_epochs = NUM_EPOCHS
)
# %% codecell
# NOTE: the modelOpt (after optimizer initialization and previous model seem to be exactly the same (tested using getParamInfo() for each from my ModelUtils to see if the tensor numbers differed some how but they seem the same)
modelOpt

# %% markdown
# Training loop here can trigger evaluation using the dev data and can trigger evaluation after training using the test data. 
# %% codecell
NUM_GPU: int = 1 # positive if CUDA is available, else 0

trainer = Trainer(
    model = modelOpt, 
    optimizer = optimizer, 
    data_silo = dataSilo, 
    epochs = NUM_EPOCHS, 
    n_gpu = NUM_GPU, 
    lr_schedule = learnRateSchedule,
    device = device
)
# %% codecell
trainer
# %% codecell
modelTrain = trainer.train()
# %% codecell
modelTrain
# %% markdown
# ### STEP 5: Inference
# Test the model on a sample (doing inference)
# %% codecell
from farm.infer import Inferencer 
from pprint import PrettyPrinter

modelInfer = Inferencer(
    processor = processor,
    model = modelTrain, 
    task_type = "text_classification", 
    gpu = True
)

basicTexts: List[Dict[str, str]] = [
    {"text" : "Martin ist ein Idiot"},
    {"text" : "Martin Müller spielt Handball in Berlin"}
]

result = modelInfer.inference_from_dicts(dicts = basicTexts)

PrettyPrinter().pprint(result)
# %% markdown
# Can see it was very confident that the second text about handball wasn't offensive while the first one was. 
# %% markdown

# ## TASK 2: Named Entity Recognition (NER)

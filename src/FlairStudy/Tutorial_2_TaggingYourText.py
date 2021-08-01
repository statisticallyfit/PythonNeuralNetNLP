# %% [markdown]
# # Tutorial 2: Tagging your Text
# ## Tagging with Pre-Trained Sequence Tagging Models
# Using a pre-trained model for named entity recognition (NER), trained over the English CoNLL-03 task , and can recognize 4 different entity types. 
# 
# %%
from flair.models import SequenceTagger
from flair.data import Sentence 
from flair.models import MultiTagger
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

# %%
tagger: SequenceTagger = SequenceTagger.load("ner")
# %%
tagger
# %% [markdown]
# Using the `predict()` method of the tagger on a sentence will add predicted tags to the tokens in the sentence. 
# 
# Using a sentence with two named entities: 
# %%

sentence = Sentence("George Washington went to Washington.")

# Predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.to_tagged_string())

# %% [markdown]
# ## Getting Annotated Spans
# Many sequence labeling methods annotate spans that consist of multiple words like "George Washington" in the example sentence. 
# 
# To directly get such spans in a tagged sentence, do: 
# %%
for entity in sentence.get_spans("ner"):
    print(entity)
# %% [markdown]

# %% [markdown]
# **Another (longer) Example for Getting Annotated Spans: **
# %%
folkAirSentence = Sentence("When she doesn't answer, I fugre there's no more point in conversation. I steer her toward the kitchens. We'll have to pass by guards; there's no other way out. She has pasted on a horrible rictus of a smile, but at least she has enough self-possession for that. More worrying is the way she can't stop staring at things. As we walkt toward the guards, the intensity of her gaze is impossible to disguise. I improvise, trying to sound as though I am reciting a memorized message, without inflection in the words. 'Prince Cardan says we are to attend him.' One of the guards turns to the other. 'Balekin won't like that.' I try not to react, but it's hard. I just stand there and wait. If they lunge at us, I am going to have to kill them. 'Very well,' the first guard says. 'Go. But inform Cardan that his brother demands he brings both of you back this time.' ")

folkAirSentence
# %%
tagger.predict(folkAirSentence)
print(folkAirSentence.to_tagged_string())

# %%
# Getting the annotated spans: 
for entity in folkAirSentence.get_spans("ner"):
    print(entity)
# %%
print(folkAirSentence.to_dict(tag_type = "ner"))
# %%
mistbornSentence = Sentence("In Hemalurgy, the type of metal used in a spike is important, as is the positioning of that spike on the body. For instance, steel spikes take physical Allomantic powers—the ability to burn pewter, tin, steel, or iron—and bestow them upon the person receiving the spike. Which of these four is granted, however, depends on where the spike is placed. Spikes made from other metals steal Feruchemical abilities. For example, all of the original Inquisitors were given a pewter spike, which—after first being pounded through the body of a Feruchemist—gave the Inquisitor the ability to store up healing power. (Though they couldn't do so as quickly as a real Feruchemist, as per the law of Hemalurgic decay.) This, obviously, is where the Inquisitors got their infamous ability to recover from wounds quickly, and was also why they needed to rest so much.")

tagger.predict(mistbornSentence)

print(mistbornSentence.to_tagged_string())
# %%
# Not all correct - Hemalurgy is not a location, it is a type of skill. 
for entity in mistbornSentence.get_spans("ner"):
    print(entity)


# %% [markdown]
# # Multi-Tagging
# Sometimes you want to predict several types of annotation at once, like NER and POS tags. You can use a `MultiTagger` object: 
# %%
#from flair.models import MultiTagger

# load tagger for POS and NER
tagger = MultiTagger.load(['pos', 'ner'])
tagger
# %%
# Predict sentence with both models
tagger.predict(mistbornSentence)
print(mistbornSentence)


# %% [markdown]
# ## [List of Pre-Trained Sequence Tagger Models](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#list-of-pre-trained-sequence-tagger-models)
# Can choose which pre-trained model you load by passing appropriate string to the `load()` method of the `SequenceTagger` class. 
# 
# **Example: Chunking**
# 
# [Meaning of the chunk tags](https://huggingface.co/flair/chunk-english-fast?text=The+happy+man+has+been+eating+at+the+diner)
# %%
chunkTagger: SequenceTagger = SequenceTagger.load("chunk-fast")
chunkTagger
# %%
chunkTagger.predict(folkAirSentence)
print(folkAirSentence)



# %% [markdown]
# ## Experimental: Semantic Frame Detection
# 
# For English, Flair provides a pre-trained model that detects semantic frames in text using Propbank 3.0 frames. 
# 
# Provides a word sense disambiguation for frame evoking words. 
# 
# ### Example 1: George and Hat
# %%
# Load model
semanticFrameTagger = SequenceTagger.load('frame')
# %%
semanticFrameTagger
# %%
# Make English sentence
s1 = Sentence("George returned to Berlin to return his hat.")
s2 = Sentence("He had a look at different hats.")


# Predict NER tags  
semanticFrameTagger.predict(s1)
semanticFrameTagger.predict(s2)

# Print sentence with predicted tags
print(s1.to_tagged_string())
print(s2.to_tagged_string())
# %% [markdown]
# ### Example 2: Drive
# %%
# Load the model
semanticFrameTagger = SequenceTagger.load("frame")
semanticFrameTagger
# %%
# Make English sentence
#kiwiSentence = Sentence("The girl sliced open the furry brown kiwi to reveal a juicy green interior, while the kiwi sang merrily on the branch in the tropical forest where kiwi hung from branches.")


# checkSentence = Sentence("Sally left the porcelain on the table before she left to visit her grandmother.")

#sentence = Sentence("While Marie drove the cart, she admired the driving ambition of the steady, plodding horses pulling her through the country as the pull of ocean waves tugged her eyes to the store.")

sentence = Sentence("Marie drove the cart through the countryside, admiring the drive of the plodding horses pulling it. She drove a nail through the tarp to protect the lumber from rain. The farmer drove away gophers from his crop. The mayor drove people into poverty with the new tax rules. The storms and tides drove the boats toward shore. ")

#checkSentence = Sentence("The girl checked her arrow before letting it fly, and after watching it whip smoothly into the target, she checked her watch for the time of day, and then remembering an urgent appointment, she hurriedly checked to make sure her equipment was packed away before cashing a check in the bank and leaving to see her friend. ")

# Predict NER tags for semantic frames
semanticFrameTagger.predict(sentence)

# Print sentence with predicted tags
print(sentence.to_tagged_string())
# %% [markdown]
# ### Example 3: Firing
# %%
sentence = Sentence("The general fired four gunshot rounds, while the second general fired the lieutenants.Curiosity sparked my imagination. The flame sparked the bonfire that ravaged the forest.")

semanticFrameTagger.predict(sentence)

print(sentence.to_tagged_string())
# %% [markdown]
# ### Example 4: Absorb
sentence = Sentence("The villagers were absorbed in their own affairs so did not notice how the fortifications were absorbing the floodwaters.")

semanticFrameTagger.predict(sentence)

print(sentence.to_tagged_string())

# %% [markdown]
# ## Tagging a List of Sentences
# Often, you want to tag an entire text corpus. Then you need to split the corpus into sentences and pass a list of `Sentence` objects to the `.predict()` method. 
# 
# For instance, can use the sentence splitter of segtok: 
# %%
#from flair.models import SequenceTagger
#from flair.tokenization import SegtokSentenceSplitter

# Example 5: Text ("Fell") with many sentences

# %%
text: str = "The rock fell through the air. The responsibility fell on his shoulders to protect the herd from the thunderstorm. Multiple animals fell into order to evade lightning strikes."

# initialize sentence splitter
splitter = SegtokSentenceSplitter()

# Use splitter to split text into multiple (list) of sentences
sentences = splitter.split(text)

# Predict tags for sentences
tagger = SequenceTagger.load("frame")
tagger.predict(sentences)

# Iterate through sentences and print predicted labels
for sentence in sentences: 
    print(sentence.to_tagged_string())

# TODO HELP: why doesn't this Frame model differentiate the different senses of the word "fell"?
# 1) "fell" as in object falling through the air
# 2) "fell" as in an intangible weight being laid on someone.
# 3) "fell" as in organize themselves in line

# %% [markdown]
# ## Tagging with Pre-Trained Text Classification Models
# Using pre-trained mdoel for detecting positive or negative comments. This model is trained over a mix of product and movie review datasets and can recognize positive and negative sentiment in english text. 
# %% codecell
from flair.models import TextClassifier

# load tagger
classifier = TextClassifier.load("sentiment")
classifier
# %% [markdown]
# All required is to use `predict()` method of the classifier on a sentence. This adds the predicted label to the sentence. 
# %%
# Predict for example sentence
sentence = Sentence("enormously entertaining for moviegoers of any age")
classifier.predict(sentence)

# check prediction
print(sentence)
# %% codecell
sentence = Sentence("A real critical thinker of this day and age; offers deeper insights than any other film about the sad, horrifying state of humanity today")

classifier.predict(sentence)

print(sentence)

# %% [markdown]
# ## Communicative Functions Text Classification
# %% codecell
# TODO doesn't work now
#functionsClassifier = TextClassifier.load('communicative-functions')
#functionsClassifier
# %% codecell

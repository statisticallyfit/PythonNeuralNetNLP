
# %%
from flair.data import Sentence

# %% [markdown] 
# Create a sentence
# %%
cowSentence: Sentence = Sentence("The cow jumped over the moon. The little dog laughed to see such sport, and the dish ran away with the spoon.")
cowSentence

# %%
#for token in sentence: 
#    print(token)

for i in range(0, len(cowSentence) + 1):
    print(cowSentence.get_token(i))

# %% [markdown]

# ## Tokenization
# TODO: how to use the `TagSentenceSplitter` that splits based on an internal tag in the sentence (so perhaps can create semantic boudnaries this way?): https://github.com/flairNLP/flair/blob/master/flair/tokenization.py#L539


# TODO is there an xlnet tokenizer? 


# %% [markdown]
# # Adding Labels

# ## Adding labels to tokens
# A `Token` has fields for linguistic annotation, such as lemmas, part-of-speech tags or named entity tags. You can add a tag by specifying the tag type and the tag value. 

# Adding an NER tag of type `color` to the word `green` which means we've tagged this word as an entity of type color. 
# %%

# Add a tag to a word in the sentence so that word green can be of type color
sentence: Sentence = Sentence("The grass is green.")

sentence[3].add_tag(tag_type = 'ner', tag_value = 'color')

# Print the sentence with all the tags of this type: 
print(sentence.to_tagged_string())

# %%
# Tagging the cow-jumped-over-the-moon sentence: 
for token in cowSentence: 
    print(token)

cowSentence[1].add_tag(tag_type = 'ner', tag_value = "animal")
cowSentence[5].add_tag(tag_type = 'ner', tag_value = "astronomical object")
cowSentence[9].add_tag(tag_type = 'ner', tag_value = "animal")
cowSentence[18].add_tag(tag_type = 'ner', tag_value = "kitchen utensil")
cowSentence[23].add_tag(tag_type = 'ner', tag_value = "kitchen utensil")

# %%
print(cowSentence.to_tagged_string())
# %% [markdown]
# Manually-added tags have confidence score of `1.0`
# %%
token = cowSentence[5]
# get the `ner` tag of the token
tag = token.get_tag('ner')

print(f' "{token}" is tagged as "{tag.value}" with confidence score "{tag.score}"')

# %% [markdown]
# ## Adding labels to Sentences
# You can add a `Label` to an entire `Sentence`. For instance the example below shows how to add the label 'sports' to a sentence, thereby labeling it as belonging it to the sports "topic". 
# %%
sportSentence: Sentence = Sentence("France is the current world cup winner.")
# add a label to a sentence
sportSentence.add_label(label_type="topic", value="sports")

print(sportSentence)

# Alternatively, you can also create a sentence with label in one line. 
sportSentence: Sentence = Sentence("France is the current world cup winner.").add_label(label_type="topic", value = "sports")

print(sportSentence)


cowSentence.add_label(label_type = "topic name whatever I want", value = "nursery rhyme")
print(cowSentence)

# %% [markdown]
# ## Multiple Labels
sportSentence.add_label("topic", "sports")
sportSentence.add_label("topic", "soccer")
sportSentence.add_label("topic", "world cup")
sportSentence.add_label("topic", "culture")

sportSentence.add_label("language", "English")

print(sportSentence)

# %%
# ## Accessing a Sentence's Labels
# %%
for label in sportSentence.labels:
    print(label)
# %%
list(map(lambda lab: lab, sportSentence.labels))
# %% [markdown]
# To get only the labels in one layer of annotation: 
# %%
list(map(lambda labType: labType, sportSentence.get_labels(label_type = "topic")))
# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#named-entity-recognition](https://huggingface.co/transformers/usage.html#named-entity-recognition)
#
# # [Named Entity Recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER)
#
# ### Pipeline Method:
# Here is an example using pipelines to do [named entity recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER), trying to identify tokens as belonging to one of $9$ classes:
#
# * `O` = Outside of a named entity
# * `B-MIS` = Beginning of a miscellaneous entity right after another miscellaneous entity.
# * `I-MIS` = Miscellaneous entity
# * `B-PER` = Beginning of a person's name right after another person's name.
# * `I-PER` = Person's name
# * `B-ORG` = Beginning of an organisation right after another organization.
# * `I-ORG` = Organization
# * `B-LOC` = Beginning of a location right after another location
# * `I-LOC` = Location
#
# Using fine-tuned model on CoNLL-2003 dataset.
# %% codecell
from transformers import pipeline, Pipeline

nlp: Pipeline = pipeline("ner")

sequence_1: str = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge which is visible from the window."

print(nlp(sequence_1))

# %% codecell
sequence_2: str = "The waterfall crashed onto the rocks below, meeting the last rays of afternoon sunlight and resulting in a cascade of brilliant colors that arced smoothly over the churning river. The trees rustled softly in the evening wind, and willows draped their long tresses over the water, where the orange sun turned them into a deep amber."

print(nlp(sequence_2))

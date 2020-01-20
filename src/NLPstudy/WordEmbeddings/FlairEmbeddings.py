from flair.embeddings import TransformerXLEmbeddings

# init embedding
embedding = TransformerXLEmbeddings()

# create a sentence
sentence = Sentence('The Berlin Zoological Garden is the oldest and best-known zoo in Germany .')

# embed words in sentence
embedding.embed(sentence)

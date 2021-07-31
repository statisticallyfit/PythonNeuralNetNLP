
# %% codecell
# TODO LEFT OFF HERE - doesn't work
# SOURCE = https://huggingface.co/jpelhaw/t5-word-sense-disambiguation


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

AutoModelForSeq2SeqLM.from_pretrained("jpelhaw/t5-word-sense-disambiguation")
AutoTokenizer.from_pretrained("jpelhaw/t5-word-sense-disambiguation")
# %%

input: str = 'question: which description describes the word " java " best in the following context? \
    descriptions:[  " A drink consisting of an infusion of ground coffee beans " , " a platform-independent programming lanugage",  or " an island in Indonesia to the south of Borneo " ] \
    context: I like to drink " java " in the morning .'


example = tokenizer.tokenize(input, add_special_tokens=True)

answer = model.generate(input_ids=example['input_ids'], 
                                attention_mask=example['attention_mask'], 
                                max_length=135)

# "a distinguishing trait"
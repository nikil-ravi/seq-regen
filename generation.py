import transformers
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2Model, BertModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import math

# returns masked token list, along with corresponding indices that need to be regenerated
def mask_out_tokens(tokens, percentage=0.15):
    mask_indices = np.random.choice(range(len(tokens)), math.floor(percentage * len(tokens)), replace=False)
    tokens = [tokens[i] if i not in mask_indices else "[MASK]" for i in range(len(tokens))]
    return tokens, mask_indices

def unmask(text, model="bert-base-uncased"):
    # TODO: might error out with non-MLM models- handle this
    unmasker = pipeline('fill-mask', model=model) 
    return unmasker(text)

def completion(prefix, model="gpt2"):
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    completion = generator(prefix, max_length=30, num_return_sequences=1)
    return completion

def get_text_features(text, config):
    model_config = config['model']
    if model_config == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
    elif model_config == "bert-base-uncased":
        tokenizer = GPT2Tokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
    elif model_config == "bert-large-uncased-whole-word-masking-finetuned-squad":
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    else:
        # TODO: add some nice way to load arbitrary models
        raise ValueError("model not supported")
    
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output




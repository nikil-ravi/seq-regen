import transformers
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2Model, BertModel


def tokens_to_regenerate():
    pass

def unmask(text):
    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    return unmasker(text)

def completion(prefix):
    pass

def get_text_features(text, config):
    model_config = config['model']
    if model_config == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
    elif model_config == "bert-base-uncased":
        tokenizer = GPT2Tokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output




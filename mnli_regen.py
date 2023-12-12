from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2Model, BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import math
import torch
import evaluate

# returns masked token list, along with corresponding indices that need to be regenerated
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

import datasets
d = datasets.load_dataset("glue", "mnli_matched")
reqd_indices = []
for i in range(len(d["validation"]["label"])):
    if d["validation"]["label"][i]==0:
        reqd_indices.append(i)
    if len(reqd_indices)>=50:
        break

sentences = [(d["validation"]["premise"][i], d["validation"]["hypothesis"][i]) for i in reqd_indices]

# d = datasets.load_dataset("gsm8k", "main")
# sentences = [(d["train"]["question"][i], d["train"]["answer"][i]) for i in range(50)]


# d = datasets.load_dataset("cnn_dailymail", "3.0.0")
# sentences = [(d["train"]["article"][i], d["train"]["highlights"][i]) for i in range(50)]

def mask_sequence(tokenized, sent1_lengths, orig_token_mask, seed_num, percentage_init=0.35, percentage_next=0.5):
    if orig_token_mask is None:
        token_mask = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_init), generator=torch.manual_seed(seed_num))
        token_mask = torch.logical_and(token_mask, tokenized["attention_mask"])
        for i in range(len(token_mask)):
            token_mask[i, :sent1_lengths[i]] = 0
    else:
        token_mask = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_next), generator=torch.manual_seed(seed_num))
        token_mask = torch.logical_and(token_mask, orig_token_mask)
    
    inp_ids = tokenized["input_ids"]
    inp_ids[token_mask==1]=tokenizer.mask_token_id
    tokenized["input_ids"] = inp_ids
    non_pad_lengths = torch.sum(tokenized["attention_mask"], dim=-1)
    return tokenized, token_mask, non_pad_lengths


def get_current_likelihoods(tokenized, orig_token_mask):

    indices_masked_originally = (orig_token_mask == 1).nonzero(as_tuple=True)
    inp_ids = tokenized["input_ids"]
    row_pos, col_pos = indices_masked_originally

    single_masked_token_sequences = []
    attention_masks = []
    token_at_position = []

    # Create sequences with a single token masked - positions are given by (row_pos[i], col_pos[i])
    for i in range(len(row_pos)):
        seq = torch.clone(inp_ids[row_pos[i]])
        token_at_position.append(seq[col_pos[i]])
        seq[col_pos[i]] = tokenizer.mask_token_id
        single_masked_token_sequences.append(seq)
        attention_masks.append(tokenized["attention_mask"][row_pos[i]])
    single_masked_token_sequences = torch.stack(single_masked_token_sequences, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    token_at_position = torch.stack(token_at_position, dim=0)

    # Split single mask sequences into mini-batches, pass through model, get probability of actual token replacing the single mask
    num_mini_batches = len(single_masked_token_sequences)//30
    single_masked_token_sequences_mini_batches = torch.chunk(single_masked_token_sequences, num_mini_batches)
    attention_masks_mini_batches = torch.chunk(attention_masks, num_mini_batches)
    col_pos_mini_batches = torch.chunk(col_pos, num_mini_batches)

    del single_masked_token_sequences, attention_masks

    single_masked_token_probs_all = []
    for mb in range(num_mini_batches):
        single_masked_tokenized = {'input_ids':single_masked_token_sequences_mini_batches[mb], 'attention_mask':attention_masks_mini_batches[mb]}
        single_masked_token_probs = torch.nn.Softmax(dim=-1)(model(**single_masked_tokenized).logits)
        single_masked_token_probs_all.append(single_masked_token_probs)
    
    del single_masked_token_sequences_mini_batches, attention_masks_mini_batches, col_pos_mini_batches
    # Combine mini-batches into single tensor
    single_masked_token_probs_all = torch.cat(single_masked_token_probs_all, dim=0)

    # Create tensor with probability 1 where orig_token_mask=0 and 
    # with probability of that token (assuming it was masked in the input) given all other tokens (unmasked in input) where orig_token_mask=1
    token_probs_curr = torch.ones(tokenized["attention_mask"].shape)
    for i in range(len(row_pos)):
        token_probs_curr[row_pos[i], col_pos[i]] = single_masked_token_probs_all[row_pos[i], col_pos[i], token_at_position[i]]
    # print(token_probs_curr)
    return token_probs_curr


def mask_sequence_based_on_past_likelihoods(tokenized, sent1_lengths, orig_token_mask, token_probs_hist, seed_num, 
                                       percentage_init=0.35, percentage_of_min_likelihoods_to_mask=0.2):
    if percentage_of_min_likelihoods_to_mask>=percentage_init:
        raise ValueError("percentage_of_min_likelihoods_to_mask must be less than percentage_init since we cannot re-mask tokens that have not been masked earlier!")
    if orig_token_mask is None:
        token_mask = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_init), generator=torch.manual_seed(seed_num))
        token_mask = torch.logical_and(token_mask, tokenized["attention_mask"])
        for i in range(len(token_mask)):
            token_mask[i, :sent1_lengths[i]] = 0
    else:
        # token_mask = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_next), generator=torch.manual_seed(seed_num))
        token_mask = torch.zeros(tokenized["attention_mask"].shape)
        indices_to_mask = torch.topk(token_probs_hist, k=2, dim=-1, largest=False).indices
        token_mask = token_mask.scatter(dim=1, index=indices_to_mask, value=1)

        # token_mask2 = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_next), generator=torch.manual_seed(seed_num))
        # token_mask = torch.logical_or(token_mask, token_mask2)
        
        # token_mask = torch.where(token_probs_hist<0.3,1,0)
        token_mask = torch.logical_and(token_mask, orig_token_mask)
    
    inp_ids = tokenized["input_ids"]
    inp_ids[token_mask==1]=tokenizer.mask_token_id
    tokenized["input_ids"] = inp_ids
    non_pad_lengths = torch.sum(tokenized["attention_mask"], dim=-1)
    return tokenized, token_mask, non_pad_lengths


def mask_sequence_based_on_past_likelihoods_with_exploration(tokenized, sent1_lengths, orig_token_mask, token_probs_hist, seed_num, 
                                       percentage_init=0.35, percentage_next = 0.15, percentage_of_min_likelihoods_to_mask=0.2):
    if percentage_of_min_likelihoods_to_mask>=percentage_init:
        raise ValueError("percentage_of_min_likelihoods_to_mask must be less than percentage_init since we cannot re-mask tokens that have not been masked earlier!")
    if orig_token_mask is None:
        token_mask = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_init), generator=torch.manual_seed(seed_num))
        token_mask = torch.logical_and(token_mask, tokenized["attention_mask"])
        for i in range(len(token_mask)):
            token_mask[i, :sent1_lengths[i]] = 0
    else:
        # token_mask = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_next), generator=torch.manual_seed(seed_num))
        token_mask = torch.zeros(tokenized["attention_mask"].shape)
        indices_to_mask = torch.topk(token_probs_hist, k=2, dim=-1, largest=False).indices
        token_mask = token_mask.scatter(dim=1, index=indices_to_mask, value=1)

        token_mask2 = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_next), generator=torch.manual_seed(seed_num))
        token_mask = torch.logical_or(token_mask, token_mask2)
        
        # token_mask = torch.where(token_probs_hist<0.3,1,0)
        token_mask = torch.logical_and(token_mask, orig_token_mask)
    
    inp_ids = tokenized["input_ids"]
    inp_ids[token_mask==1]=tokenizer.mask_token_id
    tokenized["input_ids"] = inp_ids
    non_pad_lengths = torch.sum(tokenized["attention_mask"], dim=-1)
    return tokenized, token_mask, non_pad_lengths


def mask_sequence_based_on_current_likelihoods(tokenized, sent1_lengths, orig_token_mask, token_probs_curr, seed_num, 
                                       percentage_init=0.35, percentage_of_min_likelihoods_to_mask=0.2):
    if percentage_of_min_likelihoods_to_mask>=percentage_init:
        raise ValueError("percentage_of_min_likelihoods_to_mask must be less than percentage_init since we cannot re-mask tokens that have not been masked earlier!")
    
    if orig_token_mask is None:
        token_mask = torch.bernoulli(torch.full(tokenized["attention_mask"].shape, percentage_init), generator=torch.manual_seed(seed_num))
        token_mask = torch.logical_and(token_mask, tokenized["attention_mask"])
        for i in range(len(token_mask)):
            token_mask[i, :sent1_lengths[i]] = 0
    else:
        token_mask = torch.zeros(tokenized["attention_mask"].shape)
        indices_to_mask = torch.topk(token_probs_curr, k=3, dim=-1, largest=False).indices
        # print("indices_to_mask", indices_to_mask)
        token_mask = token_mask.scatter(dim=1, index=indices_to_mask, value=1)
        
        # token_mask = torch.where(token_probs_hist<0.3,1,0)
        token_mask = torch.logical_and(token_mask, orig_token_mask)

    inp_ids = tokenized["input_ids"]
    inp_ids[token_mask==1]=tokenizer.mask_token_id
    tokenized["input_ids"] = inp_ids
    non_pad_lengths = torch.sum(tokenized["attention_mask"], dim=-1)
    return tokenized, token_mask, non_pad_lengths


def unmask_tokens(tokenized, token_mask, non_pad_lengths, output, masked_sentence_hist, unmasked_sentence_hist):
    unmasked_tokens_list = []
    # print("token_mask", token_mask)
    for i in range(len(output.logits)):
        # Masked sentence
        masked_sentence_hist[i].append(tokenizer.decode(tokenized["input_ids"][i][1:non_pad_lengths[i]-1])) 

        output_probs = torch.softmax(output.logits, dim=-1)
        predicted_tokens = torch.argmax(output_probs, dim=-1)[i]

        unmasked_tokens = torch.where(token_mask[i]==1, predicted_tokens, tokenized["input_ids"][i])
        unmasked_tokens_list.append(unmasked_tokens)

        # Sentence with mask replaced
        unmasked_sentence_hist[i].append(tokenizer.decode(unmasked_tokens[1:non_pad_lengths[i]-1])) 

    return torch.stack(unmasked_tokens_list, axis=0)


def unmask_tokens_and_return_likelihoods(tokenized, token_mask, non_pad_lengths, output, masked_sentence_hist, unmasked_sentence_hist, token_probs_hist):
    unmasked_tokens_list = []
    output_probs = torch.nn.Softmax(dim=-1)(output.logits)
    predicted_tokens_all = torch.argmax(output_probs, dim=-1)
    predicted_token_prob_all = torch.amax(output_probs, dim=-1)
    for i in range(len(output.logits)):
        # Masked sentence
        masked_sentence_hist[i].append(tokenizer.decode(tokenized["input_ids"][i][1:non_pad_lengths[i]-1])) 
        unmasked_tokens = torch.where(token_mask[i]==1, predicted_tokens_all[i], tokenized["input_ids"][i])
        token_probs_hist[i] = torch.where(token_mask[i]==1, predicted_token_prob_all[i], token_probs_hist[i])

        unmasked_tokens_list.append(unmasked_tokens)

        # Sentence with mask replaced
        unmasked_sentence_hist[i].append(tokenizer.decode(unmasked_tokens[1:non_pad_lengths[i]-1])) 

    return torch.stack(unmasked_tokens_list, axis=0)

def unmask_tokens_with_topk_sampling_and_return_likelihoods(tokenized, token_mask, non_pad_lengths, output, masked_sentence_hist, unmasked_sentence_hist, token_probs_hist):
    unmasked_tokens_list = []
    output_probs = torch.nn.Softmax(dim=-1)(output.logits)

    topk = torch.topk(output_probs, k=3, dim=-1, largest=True)

    predicted_tokens_all = []
    predicted_token_prob_all = []
    for i in range(len(topk.values)):
        chosen_indices = torch.multinomial(topk.values[i], num_samples=1)
        chosen_indices = torch.squeeze(chosen_indices, dim=-1)
        continuous_indices = torch.arange(chosen_indices.size(0))
        predicted_token_prob_all.append(topk.values[i][continuous_indices, chosen_indices])
        predicted_tokens_all.append(topk.indices[i][continuous_indices, chosen_indices])

    predicted_token_prob_all = torch.stack(predicted_token_prob_all, dim=0)
    predicted_tokens_all = torch.stack(predicted_tokens_all, dim=0)

    for i in range(len(output.logits)):
        # Masked sentence
        masked_sentence_hist[i].append(tokenizer.decode(tokenized["input_ids"][i][1:non_pad_lengths[i]-1])) 
        unmasked_tokens = torch.where(token_mask[i]==1, predicted_tokens_all[i], tokenized["input_ids"][i])
        token_probs_hist[i] = torch.where(token_mask[i]==1, predicted_token_prob_all[i], token_probs_hist[i])

        unmasked_tokens_list.append(unmasked_tokens)

        # Sentence with mask replaced
        unmasked_sentence_hist[i].append(tokenizer.decode(unmasked_tokens[1:non_pad_lengths[i]-1])) 

    return torch.stack(unmasked_tokens_list, axis=0)


def print_regen_results(sentences, masked_sentence_hist, unmasked_sentence_hist, num_regen, show_mask=False):
    for i in range(len(sentences)):
        print("ORIG0", sentences[i])
        for j in range(num_regen):
            if show_mask:
                print("MASK{}".format(j), masked_sentence_hist[i][j])
            print("RGEN{}".format(j), unmasked_sentence_hist[i][j])
        print("########################################")


tokenized_sent1 = tokenizer([i[0]+" Therefore, " for i in sentences])
sent1_lengths = [len(i) for i in tokenized_sent1["attention_mask"]]
joint_sentences = [i[0]+" Therefore, "+ i[1] for i in sentences]
tokenized = tokenizer(joint_sentences, padding=True, return_tensors='pt')

num_regen = 8
orig_token_mask = None
token_mask = None
masked_sentence_hist = [list() for _ in range(len(sentences))]
unmasked_sentence_hist = [list() for _ in range(len(sentences))]
token_probs_hist = torch.ones(tokenized["attention_mask"].shape)
token_probs_curr = None
setting = "likelihood"

# # Current likelihoods
if setting=="current_likelihood":
    for iter in range(num_regen):
        tokenized, token_mask, non_pad_lengths = mask_sequence_based_on_current_likelihoods(tokenized, sent1_lengths, 
                                                                orig_token_mask, token_probs_curr, seed_num=iter)
        if iter==0:
            orig_token_mask = token_mask
        output = model(**tokenized)
        unmasked_tokens = unmask_tokens(tokenized, token_mask, non_pad_lengths, output, 
                                                            masked_sentence_hist, unmasked_sentence_hist)
        tokenized["input_ids"] = unmasked_tokens
        token_probs_curr = get_current_likelihoods(tokenized, orig_token_mask)

# Historical Likelihoods with/without exploration
elif setting=="likelihood":
    for iter in range(num_regen):
        tokenized, token_mask, non_pad_lengths = mask_sequence_based_on_past_likelihoods(tokenized, sent1_lengths, orig_token_mask, token_probs_hist, seed_num = iter)
        if orig_token_mask is None:
            orig_token_mask = token_mask
        output = model(**tokenized)
        unmasked_tokens = unmask_tokens_and_return_likelihoods(tokenized, token_mask, non_pad_lengths, output, 
                                                            masked_sentence_hist, unmasked_sentence_hist, token_probs_hist)
        tokenized["input_ids"] = unmasked_tokens

elif setting=="likelihood_plus_exploration":
    for iter in range(num_regen):
        tokenized, token_mask, non_pad_lengths = mask_sequence_based_on_past_likelihoods_with_exploration(tokenized, sent1_lengths, orig_token_mask, token_probs_hist, seed_num = iter)
        if orig_token_mask is None:
            orig_token_mask = token_mask
        output = model(**tokenized)
        unmasked_tokens = unmask_tokens_and_return_likelihoods(tokenized, token_mask, non_pad_lengths, output, 
                                                            masked_sentence_hist, unmasked_sentence_hist, token_probs_hist)
        tokenized["input_ids"] = unmasked_tokens

# # Random masking
elif setting=="random":
    for iter in range(num_regen):
        tokenized, token_mask, non_pad_lengths = mask_sequence(tokenized, sent1_lengths, orig_token_mask, seed_num = iter)
        if orig_token_mask is None:
            orig_token_mask = token_mask
        output = model(**tokenized)
        unmasked_tokens = unmask_tokens(tokenized, token_mask, non_pad_lengths, output, masked_sentence_hist, unmasked_sentence_hist)
        tokenized["input_ids"] = unmasked_tokens

print_regen_results(joint_sentences, masked_sentence_hist, unmasked_sentence_hist, num_regen, show_mask=True)

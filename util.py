from collections import OrderedDict
from transformers import AutoTokenizer, BertModel, BertForMaskedLM
from sklearn.neighbors import NearestNeighbors
from gzip import GzipFile
import os
import pickle
import re
import torch
import tensorflow

path = os.path.dirname(os.path.realpath(__file__))
use_classic = True
if use_classic:
    model_path = 'KB/bert-base-swedish-cased'
    bert_model = BertModel.from_pretrained(model_path,
                                  return_dict=True)
    bert_model = bert_model.eval()
    mlm_model = BertForMaskedLM.from_pretrained(model_path,
                                            return_dict=True)
    mlm_model = mlm_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    model_path = '/Users/luidu652/Documents/Single-bert-base-Swedish-CT-STSb-TF'
    bert_model = BertModel.from_pretrained(model_path,
                                           return_dict=True, from_tf=True)
    bert_model = bert_model.eval()
    mlm_model = BertForMaskedLM.from_pretrained(model_path,
                                            return_dict=True, from_tf=True)
    mlm_model = mlm_model.eval()
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

E = bert_model.embeddings.word_embeddings.weight
word_embeddings = E.detach().numpy()
# load pretrained nn model if available
modelpath = f'{path}/nn_model.gzip'
if os.path.exists(modelpath):
    with GzipFile(modelpath, 'rb') as ifile:
        nn_model = pickle.loads(ifile.read())
elif False:#input('generate new nearest neighbor model? (y/n)\n>') == 'y':
    nn_model = NearestNeighbors(n_neighbors=4, metric='cosine')
    nn_model.fit(word_embeddings)
    with GzipFile(path, 'wb') as ofile:
        ofile.write(pickle.dumps(nn_model))


def find_nearest_neighbour(sent, n=None, token=None):
    """
    Parameters:
              sentence (str):
                          a sentence with the token in context (specify
                          index the token of interest as token parameter)
    """
    tokens = tokenizer.tokenize(sent)
    tok_id = tokenizer.convert_tokens_to_ids(tokens)
    if token:
        id_ = tokens.index(token)
        tok_id = tok_id[id_]
        similarity, neighbor_ids = nn_model.kneighbors([word_embeddings[tok_id]], n_neighbors=n)
        neighbors = [tokenizer.convert_ids_to_tokens(id_) for id_ in neighbor_ids]
        # print(f"{token}: {neighbors}")
    else:
        similarity, neighbor_ids = nn_model.kneighbors([word_embeddings[id_] for id_ in tok_id], n_neighbors=n+3)
        neighbors = [list(OrderedDict.fromkeys(tokenizer.convert_ids_to_tokens(ids))) for ids in neighbor_ids]
        # for i, t in enumerate(tokens):
            # print(f"{t}: {neighbors[i]}")
    return list(zip(*neighbors, *similarity))[:n]


def get_token_id(sentence, token, show_tokens=False):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence).input_ids)
    idx = [idx for idx, tok in enumerate(tokens) if tok == token]
    if show_tokens:
        print(tokens)
    return -1 if len(idx) == 0 else idx[0]


def compute_cosine_similarity(x1, x2):
    x1 = x1.reshape(1, -1)
    x2 = x2.reshape(1, -1)
    return torch.nn.CosineSimilarity()(x1, x2).item()


def compute_cosine_similarities_for_token(sentence, token, dim=-1):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = bert_model(**inputs, output_hidden_states=True)
    # size of embeds_*: (sentence_id=1, num_tokens, 768)
    # embeds_in, embeds_out = outputs.hidden_states[0], outputs.hidden_states[dim]
    token_id = get_token_id(sentence, token)
    for dim in range(len(outputs.hidden_states)):
        embeds_out = outputs.hidden_states[dim]
        # size of tok_embed_*: (768)
        # tok_embed_in, tok_embed_out = embeds_in[0, token_id, :], embeds_out[0, token_id, :]
        tok_embed_out = embeds_out[0, token_id, :]
        # cosims_in = [(i, compute_cosine_similarity(tok_embed_in, E[i]))
        #              for i in range(E.size(0))]
        cosims_out = [(i, compute_cosine_similarity(tok_embed_out, E[i]))
                      for i in range(E.size(0))]
        # cosims_in = sorted(cosims_in, key=lambda x: x[1], reverse=True)[:10]
        cosims_out = sorted(cosims_out, key=lambda x: x[1], reverse=True)[:10]
        # cosims_in = [(tokenizer.convert_ids_to_tokens(id_), sim) for id_, sim in cosims_in]
        cosims_out = [(tokenizer.convert_ids_to_tokens(id_), sim) for id_, sim in cosims_out]
        print(dim, ', '.join([f'{tok} ({sim:>1.3f})' for tok, sim in cosims_out]))
    print(tokenizer.convert_ids_to_tokens([inputs['input_ids'][0][token_id]]), token_id , len(inputs['input_ids'][0]))
    # return cosims_in, cosims_out
    return cosims_out


def find_nearest_neighbour_context(sent, n=None, token=None):
    """
    Parameters:
              sentence (str):
                          a sentence with the token in context (specify
                          index the token of interest as token parameter)
    """
    tokens = tokenizer.tokenize(sent)
    tok_id = tokenizer.convert_tokens_to_ids(tokens)
    inputs = tokenizer(sent, return_tensors='pt')
    outputs = bert_model(**inputs, output_hidden_states=True)
    emb_in, emb_out = outputs.hidden_states[0], outputs.hidden_states[-1]
    if token:
        id_ = tokens.index(token)
        tok_id = tok_id[id_]
        # print(tok_id, id_)
        # print(tokens[id_])
        # print(emb_in.shape, emb_out.shape)
        tok_emb_in, tok_emb_out = emb_in[0, id_, :], emb_out[0, id_, :]
        similarity, neighbor_ids = nn_model.kneighbors([tok_emb_in.detach().numpy()], n_neighbors=n)
        # print(neighbor_ids)
        neighbors_in = [tokenizer.convert_ids_to_tokens(id_) for id_ in neighbor_ids]
        # print(f"INPUT: {token}: {neighbors_in}")
        similarity, neighbor_ids = nn_model.kneighbors([tok_emb_out.detach().numpy()], n_neighbors=n)
        # print(neighbor_ids)
        neighbors_out = [tokenizer.convert_ids_to_tokens(id_) for id_ in neighbor_ids]
        # print(f"OUTPUT: {token}: {neighbors_out}")

    return neighbors_in + neighbours_out

## MLM
def get_mask_index(input_ids, tokenizer):
  x = input_ids[0]
  is_masked = torch.where(x == tokenizer.mask_token_id, x, 0)
  mask_idx = torch.nonzero(is_masked)
  return mask_idx.item()


def get_top_k_predictions(pred_logits, mask_idx, top_k):
  probs = torch.nn.functional.softmax(pred_logits[0, mask_idx, :], dim=-1)
  top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
  top_k_pct_weights = [100 * x.item() for x in top_k_weights]
  top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
  return list(zip(top_k_tokens, top_k_pct_weights))


def get_predictions(sentence, tokenizer, model=mlm_model, n=20):
  inputs = tokenizer(sentence, return_tensors="pt")
  outputs = bert_model(**inputs)
  mask_idx = get_mask_index(inputs.input_ids, tokenizer)
  top_preds = get_top_k_predictions(outputs.logits, mask_idx, n)
  return top_preds


def mask_sent(sent, term):
    sent = re.sub(rf'\b[^\b\s.!?]*{term}[^\b\s.!?]*\b', '[MASK]', sent)
    print(sent)
    return sent

# example
sents = ['Kortare arbetstider är en fråga om ökad livskvalité, som medger mer tid för barn och familj. I många slitsamma yrken kan också en kortare arbetstid bidra till färre arbetsskador och en förbättrad hälsa.',
         'En väl fungerande arbetskraftsinvandring bidrar till ökat välstånd och tillväxt i landet samt möjliggör för individer att utvecklas och skapa en bättre livssituation.',
         'Revideringen har resulterat i att de svenska utsläppen av växthusgaser har ökat för år 1990 i förhållande till de uppgifter som Sverige redovisade i den andra svenska nationalrapporten om klimatförändringar.',
         'En allt rörligare arbetsmarknad och ökad arbetslöshet har medfört att många individer inte har kvar en förankring i ett yrkesområde på samma sätt som tidigare.',
         'Att godkänna avvikelser som arbetsgivaren först efter påpekanden har korrigerat skulle emellertid kunna öppna upp för missbruk och leda till att legitimiteten för bestämmelserna minskade . ']
terms = ['hälsa', 'tillväxt', 'klimat', 'arbetslöshet', 'missbruk']
n = 50
collected_terms = {}
if False:#for i, term in enumerate(terms):
    #print('BERT WORD EMBEDDINGS:')
    w_emb_nn = find_nearest_neighbour(term, n)[0]
    wc_emb_nn = find_nearest_neighbour(sents[i], n, term)[0]
    union = set(w_emb_nn).union(wc_emb_nn)
    overlap = [el for el in w_emb_nn if el in wc_emb_nn]
    #print('OVERLAP:', overlap)
    #masked_sent = mask_sent(sents[i], term)
    print()
    #print("MLM:")
    #mlm_nn, prob = zip(*get_predictions(masked_sent, tokenizer, n=n))
    #print(mlm_nn)
    #print()
    #overlap = [el for el in mlm_nn if el in overlap]
    #union = union.union(mlm_nn)
    #print('OVERLAP:', overlap)
    #print()
    print('UNION size:', len(union))
    print(union)
    collected_terms[term] = union
    print('____'*10)


for n in []:#range(len(sents)):
    print(f'SENTENCE: {sents[n]}, TERM: {terms[n]}')
    dim = compute_cosine_similarities_for_token(sents[n], terms[n])
    

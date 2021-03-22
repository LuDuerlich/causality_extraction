import csv
import base64
import streamlit as st
from streamlit.hashing import _CodeHasher
import pandas as pd
import torch
import time
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

import pickle
import datetime
from io import BytesIO
from sklearn.neighbors import NearestNeighbors
from collections import OrderedDict
import gzip
import os
import re
import math
import logging
from urllib import parse

st.set_page_config(page_title='demo app',
                   page_icon=':mag:',
                   layout='centered',
                   initial_sidebar_state='expanded')

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO)


def get_table_download_link(table):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    table.to_excel(writer, index=False, sheet_name=f'Sökresultat',
                   float_format="%.2f")

    writer.save()
    output_val = output.getvalue()
    b64 = base64.b64encode(output_val)

    link = f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="sökresultat.xlsx">spara resultat</a>'
    return link


@st.cache
def generate_prompts(cause=None, effect=None):
    """
    insert topic and expansions into prompt templates.
    This is a local version that does not include query expansion
    for efficiency reasons.
    """

    prompt_dict = {
        '"bero på"': ['X beror på [MASK]'],
        '"bidra till"': ['[MASK] bidrar till X'],
        'framkalla': ['[MASK] framkallar X'],
        'förorsaka': ['[MASK] förorsakar X'],
        '"leda till"': ['[MASK] leder till X'],
        'medföra': ['[MASK] medför X'],
        'orsaka': ['[MASK] orsakar X'],
        '"på grund av"': ['X på grund av [MASK]'],
        'påverka': ['[MASK] påverkar X'],
        'resultera': ['[MASK] resulterar i X'],
        '"till följd av"': ['X till följd av [MASK]'],
        '"vara ett resultat av"': ['X är ett resultat av [MASK]'],
        'vålla': ['[MASK] vållar X']}

    templates = [template.lstrip('(alternativ: ').rstrip(')')
                 for keyword_templates in prompt_dict.values()
                 for template in keyword_templates]

    def fill_templates(term, templates, placeholder='X'):
        return [template.replace(placeholder, term) for template in templates]

    # topic_terms = [topic]
    # generate prompts
    prompts = []
    if effect:

        prompts = [prompt for term in [effect]
                   for prompt in fill_templates(term, templates)]
        if cause:
            prompts = [prompt for term in [cause]
                       for prompt in fill_templates(term, prompts, '[MASK]')]
    elif cause:
        prompts = [prompt for term in [cause]
                   for prompt in fill_templates(term, templates, '[MASK]')]

    return prompts


@st.cache(allow_output_mutation=True)
def load_binary(emb_file):
    if emb_file.endswith('.gzip') or emb_file.endswith('.gz'):
        with gzip.GzipFile(emb_file, 'rb') as ifile:
            embeddings = pickle.loads(ifile.read())
    elif emb_file.endswith('.pickle'):
        with open(emb_file, 'rb') as ifile:
            embeddings = pickle.load(ifile)
    return embeddings


@st.cache(allow_output_mutation=True)
def load_documents(input_emb, input_meta):
    """
    load prefiltered text and embeddings.
    """
    start = time.time()
    print(f'{time.asctime()} loading embeddings')
    docs = {}
    docs['embeddings'] = load_binary(input_emb)
    docs['meta'] = load_binary(input_meta)
    print(f'{time.asctime()} load_documents() took {time.time()-start} s ')
    return docs


@st.cache(allow_output_mutation=True)
def init_ct_model():
    from transformers import AutoModel, AutoTokenizer
    import torch

    print(f'{time.asctime()} loading model')
    on_gpu = torch.cuda.is_available()
    print(f'{time.asctime()} GPU available: {on_gpu}')
    model_name = tok_name = "Contrastive-Tension/BERT-Base-Swe-CT-STSb"
    print(f'{time.asctime()} loading tokeniser: {tok_name}')
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    print(f'{time.asctime()} loading BERT model: {model_name}')
    model = AutoModel.from_pretrained(model_name, from_tf=True)
    model.eval()
    return on_gpu, tokenizer, model


@st.cache
def embed_text(samples, prefix='', save_out=True):
    """
    embed samples using the swedish STS model
    """
    import torch
    from torch.utils.data import DataLoader, SequentialSampler

    print(f'{time.asctime()} embedding {len(samples)} sentences ...')
    embeddings = []
    batch_size = 100
    with torch.no_grad():
        if on_gpu:
            model.to('cuda')
        dataloader = DataLoader(
            samples,
            sampler=SequentialSampler(samples),
            batch_size=batch_size
        )
        print(f'{time.asctime()} {math.ceil(len(samples)/batch_size)} batches')
        for i, batch in enumerate(dataloader):
            if i % 100 == 0:
                print(f'{time.asctime()} at batch {i}')
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
            if on_gpu:
                inputs = inputs.to('cuda')
            out = model(**inputs)
            b_embeddings = mean_pool(out, inputs['attention_mask'])
            embeddings.append(b_embeddings.cpu())
        embeddings = torch.cat(embeddings)
    if save_out:
        filename = f'{prefix}{len(samples)}_embeddings.gzip'
        with gzip.GzipFile(filename, 'wb') as embeddings_out:
            embeddings_out.write(pickle.dumps(embeddings))
            print(f'{time.asctime()} saved embeddings to {filename}')
        return embeddings, filename
    print(f'{time.asctime()} done')
    return embeddings


def mean_pool(model_out, input_mask):
    """following sentence_transformers"""
    import torch

    embeddings = model_out[0]
    attention_mask = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * attention_mask, 1)
    n = torch.clamp(attention_mask.sum(1), min=1e-9)
    return sum_embeddings / n


# @st.cache
def display_result(state, term, doc_id, filter):
    """
    display a single match if it matches the filter
    """
    start = time.time()
    match = state.ranking[(term, state.scope)][doc_id]
    stats = ['rank', 'count', 'distance', 'nb_matches']
    if isinstance(doc_id, tuple):
        text, doc_id, sent_nb, match_emb_id = doc_id
    doc_id = doc_id.split('_')[-1].split('.')[0]
    doc_title, date = ids_to_date[doc_id][0]
    date = datetime.datetime.fromisoformat(date)

    def format_stats(k, match=match):
        return (": ".join([k, f"{match[k]:>1.3f}"])
                if isinstance(match[k], float)
                else ": ".join([k,
                                f"{sum(match[k]) / match['count']:>1.3f}"
                                if isinstance(match[k], list)
                                else str(match[k])]))\
                                     if k in match else ''

    if date.year in range(filter['time_from'], filter['time_to']):
        # if we extract page ids from the html we might even be able to
        # link the approximate location of the match
        # (ids seem to be a little off)
        # todo add newline for lists (or remove them from matches)
        displayed_sents = 0
        doc_title_text = doc_title
        continuation = re.findall(r'd(\d+)$', doc_id)
        html_link = f'https://data.riksdagen.se/dokument/{doc_id}'
        if continuation:
            doc_title_text = f'{doc_title}, del {continuation[0]}'

        if 'matched_text' in match.keys():
            st.header(f'[{doc_title_text}]({html_link})')

            stats_header = f'({", ".join([format_stats(k) for k in stats])})'
            st.subheader(stats_header)
            for sent in sorted(match['matched_text'],
                               key=lambda x: (
                                   match['matched_text'][x]['distance'],
                                   -1 * match['matched_text'][x]['rank'])):
                sentence_match = match['matched_text'][sent]
                print(sentence_match, sent)
                if displayed_sents == 3:
                    break
                sent_stats = ', '.join([format_stats(k, sentence_match)
                                        for k in stats])
                render_sentence(sentence_match['text']['content'],
                                sentence_match,
                                state,
                                sentence_match['text']['emb_id'],
                                sent_stats, doc_title_text,
                                section=sent.split(':')[-1].strip("'"))
                #                 st.markdown(sent.split(':')[-1].strip("'"))
                # st.markdown(match['matched_text'][sent]['text']['content']
                #         + f" {match['matched_text'][sent]['rank']:>2.3f}")

                displayed_sents += 1

        else:
            stats = ', '.join([format_stats(k) for k in stats])
            render_sentence(text, match, state, match_emb_id,
                            doc_title_text, stats, html_link)
        print(f'{time.asctime()} display_result({term})',
              f'took {time.time()-start} s ')
        return True
    return False
# st.number_input('Enter a number')


def render_sentence(text, match, state, emb_id, doc_title,
                    stats, html_link=None, section=None):
    # target = text.split('**')[1]
    if not hasattr(state, 'more_like') or not state.more_like_buttons:
        state.more_like_buttons = {}
    if not state.result:
        state.result = []
    res = {}
    if section:
        st.subheader(section)
        res['section'] = section
    target = text.strip("'")
    st.markdown(target)
    res['vänster kontext'], res['träff'],\
        res['höger kontext'] = target.split('**')
    st.markdown(stats)
    res['stats'] = stats
    # st.markdown(f'rank: {sum(match["rank"])/match["count"]:>1.3f},
    # distance: {sum(match["distance"])/match["count"]:>1.3f}, count: {match["count"]}')

    if state.debug == 1:
        debug_stats = pd.DataFrame({'rank': match["rank"],
                                    'distance': match["distance"]})
        st.table(debug_stats)
        st.write(f'embedding id: {emb_id}')
    res['doc'] = doc_title
    if html_link:
        st.markdown(
            f'Här hittar du dokumentet: [{doc_title}]({html_link})')
        res['html'] = html_link
    preset_params = parse.urlencode({'emb_id': emb_id,
                                     'time_from': state.time_from,
                                     'time_to': state.time_to,
                                     'n_results': state.n_results},
                                    doseq=True)
    st.markdown(
        '[visa fler resultat som liknar avsnittet!]' +
        f'(http://localhost:8501/?{preset_params})')
    state.result.append(res)


def order_results_by_sents(distances, neighbours, prompts, text):
    print(f'{time.asctime()} start sorting by sents')
    match_dict = {}
    ranked_dict = OrderedDict()

    def rank_func(x):
        return (len(match_dict[x]['rank']))
    # , sum(match_dict[x]['distance'], match_dict[x]['count'])

    for i, prompt in enumerate(prompts):
        for j, n in enumerate(neighbours[i]):
            contents = " ".join([text[n][-3],
                                 '**' + text[n][-2] + '**',
                                 text[n][-1]])
            doc_id, id = text[n][0].split(':', 1)
            embedding_id = n
            key = (contents, doc_id, id, embedding_id)
            if key not in match_dict:
                match_dict[key] = {'rank': [],
                                   'count': 0,
                                   'nb_matches': 0,
                                   'distance': []}
            match_dict[key]['rank'].append(j)
            match_dict[key]['distance'].append(distances[i][j])
            match_dict[key]['count'] += 1
    for key in sorted(match_dict, key=rank_func, reverse=True):
        ranked_dict[key] = match_dict[key]
    print(f'{time.asctime()} stop sorting')
    return ranked_dict


@st.cache(allow_output_mutation=True)
def run_ranking(prompts, train, n=30, sorting_func=order_results_by_sents,
                emb_id=None):
    start = time.time()
    if emb_id is None:
        embeddings = embed_text(prompts, save_out=False)
    else:
        embeddings = torch.unsqueeze(train['embeddings'][emb_id], dim=0)
    outpath = f'{len(train)}_nn.gzip'
    if not os.path.exists(outpath):
        nn = NearestNeighbors(n_neighbors=40, metric='cosine', p=1)
        nn.fit(train['embeddings'])
        with gzip.GzipFile(outpath, 'wb') as data_out:
            data_out.write(pickle.dumps(nn))
    else:
        nn = load_binary(outpath)
    distance, neighbours = nn.kneighbors(embeddings, n_neighbors=n)
    print(f'{time.asctime()} run_ranking({prompts})',
          f'took {time.time()-start} s ', sorting_func)
    return sorting_func(distance, neighbours, prompts, train['meta'])


@st.cache
def order_results_by_documents(distances, neighbours, prompts, text,
                               n_neighbours=10):
    """
    groups matches by document and orders according to avg document rank and
    similarity (still needs to factor in average match count per document)
    """
    print('DOCUMENT based ORDERING')
    print(f'{time.asctime()} start sorting')
    match_dict = {}
    max_distances = []
    max_dist = 0
    topic_count = 0
    for i, prompt in enumerate(prompts):
        topic_count += 1
        for j, n in enumerate(neighbours[i]):
            match = {}
            contents = " ".join([text[n][-3],
                                 '**' + text[n][-2] + '**',
                                 text[n][-1]])
            match['content'] = contents
            match['doc_id'], id = text[n][0].split(':', 1)
            match['emb_id'] = n
            if match['doc_id'] not in match_dict:
                match_dict[match['doc_id']] = {'rank': 0,
                                               'count': 0,
                                               'nb_matches': 0,
                                               'distance': 0,
                                               'matched_text': {}}
            distance = distances[i][j]
            if id not in match_dict[match['doc_id']]['matched_text']:
                match_dict[match['doc_id']]['matched_text'][id] = {
                    'rank': [],
                    'count': 0,
                    'distance': [],
                    'text': match}
            match_dict[match['doc_id']]['matched_text'][id]['rank'].append(j)
            match_dict[match['doc_id']]['matched_text'][id]['distance'].append(
                distance)
            match_dict[match['doc_id']]['matched_text'][id]['count'] += 1
            distance = float(distance)
            if distance > max_dist:
                max_dist = distance
        max_distances.append(max_dist)
    for doc_id, neighbor in match_dict.items():
        count = 0
        for text in neighbor['matched_text']:
            stats = neighbor['matched_text'][text]
            avg_rank = (((topic_count - stats['count'])
                              * (n_neighbours + 1))
                             + sum([int(el) + 1 for el in stats['rank']])
                             ) / topic_count
            avg_distance = (((topic_count - stats['count']) * max_dist)
                                 + sum([float(el) for el in stats['distance']])
                                 ) / topic_count
            match_dict[doc_id]['count'] += stats['count']
            match_dict[doc_id]['rank'] += avg_rank
            match_dict[doc_id]['distance'] += avg_distance
            match_dict[doc_id]['nb_matches'] += 1
            count += 1
        match_dict[doc_id]['count'] /= count
        match_dict[doc_id]['rank'] /= count
        match_dict[doc_id]['distance'] /= count
    neighbours = OrderedDict()
    for doc in sorted(match_dict,
                      key=lambda x: (match_dict[x]['distance'],
                                     -1 * match_dict[x]['rank'])):
        neighbours[doc] = match_dict[doc]
    print(f'{time.asctime()} stop sorting')
    return neighbours


with open('ids_to_date.pickle', 'rb') as ifile:
    ids_to_date = pickle.load(ifile)
on_gpu, tokenizer, model = init_ct_model()


def main():
    print(f'{time.asctime()} start main')
    state = _get_state()
    for k, v in st.experimental_get_query_params().items():
        if v[0].isnumeric():
            state[k] = int(v[0])
        else:
            state[k] = v[0] if not v[0] == 'None' else None
    st.sidebar.title(":wrench: Inställningar")
    if not state.n_results:
        state.n_results = 10
    state.n_results = st.sidebar.slider('max antal matchningar',
                                        min_value=1, max_value=30,
                                        value=state.n_results)
    st.sidebar.markdown('---')
    select_options = ['ämne', 'mening']
    index = select_options.index(state.search_type) \
        if state.search_type else 0
    state.search_type = st.sidebar.radio('Söktyp:', select_options, index)
    st.sidebar.markdown('---')

    from_options = [i for i in range(1995, 2020, 1)]
    index = 0
    if state.time_from:
        index = from_options.index(state.time_from)
    state.time_from = st.sidebar.selectbox('fr.o.m', from_options, index=index)
    to_options = sorted([el for el in from_options if el > state.time_from],
                        reverse=True)
    if state.time_to:
        index = to_options.index(state.time_to)
    state.time_to = st.sidebar.selectbox('t.o.m', to_options, index=index)

    st.sidebar.markdown('---')
    select_options = ['enskilda meningar', 'dokument']
    index = state.scope if state.scope else 0
    state.scope = select_options.index(st.sidebar.radio('gruppering',
                                                        select_options, index))

    st.sidebar.markdown('---')
    select_options = ['off', 'on']
    index = state.debug if state.debug else 0
    state.debug = select_options.index(st.sidebar.radio('Debug',
                                                        select_options, index))

    update_query_params(state)

    # Display the selected page with the session state
    page_sent_search(state)

    # Mandatory to avoid rollbacks with widgets,
    # must be called at the end of your app
    # state.sync()
    print(f'{time.asctime()} end main')


def get_headers():
    # Hack to get the session object from Streamlit.

    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    # Multiple Session Objects?
    for session_info in session_infos:
        headers = session_info.ws.request.headers
        st.write(headers)
#    return headers


def page_sent_search(state):
    default = ''
    cause_default = ''
    effect_default = ''
    emb_id = None
    header = None
    query_params = get_query_params(state)
    # query_params = st.experimental_get_query_params()
    if state.debug:
        st.write(f'QUERY params: {query_params}')

    if 'emb_id' in query_params:
        emb_id = query_params['emb_id']
        if isinstance(emb_id, list):
            emb_id = emb_id[0]
        if isinstance(emb_id, str):
            emb_id = int(emb_id)
        if not state.train:
            state.train = load_documents(
                './filtered_vs_unfiltered_nn/full_matches_353599_embeddings.gzip',
                'meta.pickle.gz')
        default = state.train['meta'][emb_id][3]
        state.search_type = 1
    else:
        if state.query is not None:
            default = state.query
        if state.query_cause is not None:
            cause_default = state.query_cause
        if state.query_effect is not None:
            effect_default = state.query_effect
    if emb_id is not None:
        # there is a bug/undesired refreshing of the page that interferes here!
        # sometimes?
        st.title(":mag: Fler resultat som ...")
        start_search = True
        header = st.header('Resultat för meningsbaserat sökning')
    else:
        st.title(":mag: Sökning")
        if state.search_type == 'mening':
            state.query = st.text_input('Ange en mening',
                                        default)
            update_query_params(state)
        else:
            state.query_cause = st.text_input('Ange en orsak', cause_default)
            update_query_params(state)
            state.query_effect = st.text_input('Ange ett effekt',
                                               effect_default)
            update_query_params(state)

        # state.search_type = select_options.index(state.search_type)
        updated = update_query_params(state)
        if updated:
            start_search = st.button('skapa sökfrågan')
        else:
            start_search = True
    if start_search:
        if not header:
            st.header(f'Resultat för {state.search_type}sbaserat sökning')
        result_link = st.empty()
        if (state.query_cause or state.query_effect)\
           and state.search_type == 'ämne':

            prompts = generate_prompts(cause=state.query_cause,
                                       effect=state.query_effect)
            rank(state, prompts, emb_id=emb_id)
        elif default:
            st.markdown(f'## ”_{default}_”')
            rank(state, [default], emb_id=emb_id)
        if state.result:
            table = pd.DataFrame(state.result)
            result_link.markdown(get_table_download_link(table),
                                 unsafe_allow_html=True)


def rank(state, prompts, emb_id=None):
    st.markdown('---')
    # reset results
    state.result = []
    start = time.time()
    if not hasattr(state, 'train') or not state.train:
        state.train = load_documents(
            './filtered_vs_unfiltered_nn/full_matches_353599_embeddings.gzip',
            'meta.pickle.gz')
    if isinstance(prompts, str):
        term = prompts
        prompts = [prompts]
    elif isinstance(prompts, list) and len(prompts) == 1:
        term = prompts[0]
    else:
        term = ''
        if state.query_cause:
            term += f'Orsak: {state.query_cause}'
        if state.query_effect:
            term += f'; Verkan: {state.query_effect}'
        term = term.strip('; ')
        state.term = term
    if not state.ranking:
        state.ranking = {}
    ranking_key = (term, state.scope)
    if ranking_key not in state.ranking:
        sorting_func = order_results_by_documents if state.scope == 1\
            else order_results_by_sents
        state.ranking[ranking_key] = run_ranking(
            prompts, state.train,
            n=10, emb_id=emb_id,
            sorting_func=sorting_func)
    n_matches = 0
    logging.info(
        f'ranking {len(state.ranking[ranking_key])} documents for "{term}"')
    for el in state.ranking[ranking_key]:
        hit = display_result(state, term, el, {'time_from': state.time_from,
                                               'time_to': state.time_to})
        if hit:
            n_matches += 1
            if n_matches >= state.n_results:
                break
            st.markdown('---')
    print(f'{time.asctime()} ranking({prompts}) took {time.time()-start} s ')
    print(f'{time.asctime()} all matches displayed')


def page_dashboard(state):
    st.title(":chart_with_upwards_trend: Dashboard page")
    display_state_values(state)


def display_state_values(state):
    st.write('state size:', len(state._state['data']),
             state._state['data'].keys())
    st.write('Query:', state.query)
    st.write('more like:', state.more_like_buttons)
    if state.train:
        st.write('Train:', len(state.train))
    st.write('Checkboxes:', state.checkboxes)
    if state.ranking:
        n_rank = len(state.ranking)
        st.write('Ranking:', n_rank,
                 sum([len(v) for v in state.ranking.values()])/n_rank)
    st.write('Button:', state.button)

    if st.button("Clear state"):
        state.clear()


def update_query_params(state):
    params = ['emb_id', 'time_from', 'time_to', 'debug',
              'n_results', 'search_type', 'scope', 'query',
              'query_cause', 'query_effect']
    old_states = st.experimental_get_query_params() 
    updated_states = {}
    has_changes = False
    for param in params:
        if state[param]:
            if not has_changes or param not in old_states\
               or state[param] != old_states[param]:
                has_changes = True
            updated_states[param] = state[param]
                      
    st.experimental_set_query_params(**updated_states)
    return has_changes


def get_query_params(state):
    query_params = st.experimental_get_query_params()
    params = ['emb_id', 'time_from', 'time_to', 'debug',
              'n_results', 'search_type', 'scope', 'query',
              'query_cause', 'query_effect']
    for p in params:
        if p not in query_params and state[p] is not None:
            query_params[p] = state[p]
    return query_params


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """
        Rerun the app with all state values up to date
        from the beginning to fix rollbacks.
        """

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(
                    self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(
            self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()

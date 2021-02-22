#!/usr/bin/env python
# import re
# from search_terms import filtered_expanded_dict
from bs4 import BeautifulSoup
import pickle
import spacy
import os
path = os.path.dirname(os.path.realpath('__file__'))
model_path = f'{path}/spacy_model/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
model = spacy.load(model_path)
with open('BERT_topics.pickle', 'rb') as ifile:
    topic_terms = pickle.load(ifile)
topics = [term for topic in topic_terms for term in topic]


def compute_path_length(hit, parsed_text):
    caus, top = find_tokens(BeautifulSoup(hit),
                            parsed_text)
    shortest = None
    tokens = [None, None]
    parsed_text = parsed_text.split()
    deps_to_heads = {i: int(el.split('//')[-1]) for i, el
                     in enumerate(parsed_text)}
    heads_to_deps = {}
    for dep, head in deps_to_heads.items():
        if head not in heads_to_deps:
            heads_to_deps[head] = []
        heads_to_deps[head].append(dep)
    for c in caus:
        if shortest == 1:
            break
        c_ix = parsed_text.index(c)
        for t in top:
            path = find_path(c_ix,
                             parsed_text.index(t),
                             deps_to_heads, heads_to_deps)
            if path and (shortest is None or shortest > path):
                shortest = path
                tokens = [t, c]
                if path == 1:
                    break
    return shortest, tokens


def find_tokens(markdown, parsed_text):
    """
    match causality and topic terms to the respective spacy tokens
    """

    # text = markdown.b.get_text(" ", strip=True)
    causality_hits, topic_hits = split_keywords(markdown)
    causality_tok = []
    topic_tok = []

    # doc = model(text)
    parsed_text = parsed_text.split()
    for i, token in enumerate(parsed_text):
        if len(causality_tok) < len(causality_hits):
            for context in causality_hits:
                left_context, term, right_context = context.split(',')
                if term == token.split('//')[0]:
                    # check left context
                    if left_context and\
                       parsed_text[max(0, i-1)].split('//')[0] != left_context:
                        continue

                    # check right context
                    if right_context and \
                       parsed_text[min(i+1, len(parsed_text))].split('//')[0]\
                       != right_context:
                        continue

                    causality_tok.append(token)
                    continue
        if len(topic_tok) < len(topic_hits):
            for term in topic_hits:
                term = term.split(',')[1]
                if term in token.split('//')[0]:
                    topic_tok.append(token)
                    continue
    return causality_tok, topic_tok


def get_context(em):
    """get context words for a keyword hit"""
    previous = em.previous_sibling.split()
    if previous:
        previous = previous[-1].strip()
    else:
        previous = ""
    following = em.next_sibling.split()
    if following:
        following = following[0].strip()
    else:
        following = ""
    em = em.text.strip()
    return [previous, em, following]


def shortest_path(causality_tokens, topic_tokens):
    """
    find the shortest path between any of the causality and\
    topic tokens
    """
    path_length = None
    for caus in causality_tokens:
        for top in topic_tokens:
            length = count_path(caus, top)
            if length == 0:
                return 0
            elif not path_length or length < path_length:
                path_length = length
    return path_length


def count_path(causality_tok, topic_tok):
    """
    count the length of the path to the
    root of the tree or the token if it happens
    to be head of the other token
    """
    distance = 0
    is_root = True
    head = topic_tok.head
    while head.head != head:
        head = head.head
        distance += 1
        if head == causality_tok:
            is_root = False
    return distance, is_root


def find_path(t1, t2, deps_to_heads, heads_to_deps):
    """
    find the path between two tokens.
    """
    path = 0
    # check if one of the terms is direct or indirect
    # head of the other
    path, is_connection = _check_dependants(t1, t2,
                                            deps_to_heads,
                                            heads_to_deps)
    if is_connection:
        return path

    path, is_connection = _check_dependants(t2, t1,
                                            deps_to_heads,
                                            heads_to_deps)
    if is_connection:
        return path

    # check if they share a parent
    path, is_connection = _check_head(t1, t2,
                                      deps_to_heads,
                                      heads_to_deps)

    if is_connection:
        return path


def _check_head(t1, t2, deps_to_heads, heads_to_deps):
    """
    check if t1 and t2 share the same ancestor
    and calculate the lenght of the path between
    them.
    """
    head = t1
    previous = None
    path, found = None, None
    nb_heads = 0
    while not found:
        previous = head
        head = deps_to_heads[head]
        nb_heads += 1
        if head == previous:
            return 0, 0

        path, found = _check_dependants(head, t2,
                                        deps_to_heads,
                                        heads_to_deps,
                                        previous)
        if found:
            path += nb_heads

    return path, found


def _check_dependants(t1, t2, deps_to_heads, heads_to_deps, stop=None):
    """
    check if t2 is dependant of t1 and
    calculate path length.
    """
    path = 0
    found = 0
    if stop and stop == t1:
        return path, found

    if t1 == t2:
        return 0, 1

    if t1 in heads_to_deps:
        for dep in heads_to_deps[t1]:

            if dep != t1:
                path, found = _check_dependants(dep, t2, deps_to_heads,
                                                heads_to_deps, stop)
            if found:
                path += 1

                break
    return path, found


def split_keywords(markdown):
    causality_hits = []
    topic_hits = []
    hits = markdown('em')
    hits = [get_context(em) for em in hits]
    hit_sequence = [','.join([el.strip(',') for el in hit]) for hit in hits]
    for segment in hit_sequence:
        if any([el for el in topics if el in segment]):
            topic_hits.append(segment)
        else:
            causality_hits.append(segment)

    return causality_hits, topic_hits

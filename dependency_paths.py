#!/usr/bin/env python
# from bs4 import BeautifulSoup
# import re
# from search_terms import filtered_expanded_dict
import pickle
import spacy
model_path = 'spacy_model/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
model = spacy.load(model_path)
with open('BERT_topics.pickle', 'rb') as ifile:
    topic_terms = pickle.load(ifile)
topics = [term for topic in topic_terms for term in topic]


def find_tokens(markdown):
    """
    match causality and topic terms to the respective spacy tokens
    """

    text = markdown.b.get_text(" ", strip=True)
    causality_hits, topic_hits = split_keywords(markdown)
    causality_tok = []

    topic_tok = []

    doc = model(text)
    for i, token in enumerate(doc):
        if len(causality_tok) < len(causality_hits):
            for context in causality_hits:
                left_context, term, right_context = context.split(',')
                if term == token.text.strip(','):

                    # check left context
                    if left_context and doc[max(0, i-1)].text != left_context:
                        continue

                    # check right context
                    if right_context and \
                       doc[min(i+1, len(doc))].text != right_context:
                        continue

                    causality_tok.append(token)
                    continue
        if len(topic_tok) < len(topic_hits):
            for term in topic_hits:
                term = term.split(',')[1]
                if term in token.text:
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
            print(length)
            if length == 0:
                return 0
            elif not path_length or length < path_length:
                print(f'update path: {length}')
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


def _path(t1, t2):
    """
    find the path between two tokens.
    """
    path = 0
    # check if one of the terms is direct or indirect
    # head of the other
    path, is_connection = _check_children(t1, t2)
    if is_connection:
        return path
    path, is_connection = _check_children(t2, t1)
    if is_connection:
        return path
    # check if they share a parent
    path, is_connection = _check_parent(t1, t2)
    if is_connection:
        return path


def _check_parent(t1, t2):
    """
    check if t1 and t2 share the same ancestor
    and calculate the lenght of the path between
    them.
    """
    parent = t1.head
    if parent == t1:
        return 0, 0
    path, found = _check_children(parent, t2, t1)
    if found:
        return path + 1, found
    else:
        path, found = _check_parent(parent, t2)
        if found:
            path += 1
    return path, found


def _check_children(t1, t2, stop=None):
    """
    check if t2 is descendant of t1 and
    calculate path length.
    """
    children = list(t1.children)
    path = 0
    found = 0
    if stop and stop == t1:
        return path, found
    if t1 == t2:
        return 0, 1
    for child in children:
        path, found = _check_children(child, t2)
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

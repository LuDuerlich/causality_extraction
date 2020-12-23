from bs4 import BeautifulSoup
from spacy.tokens.span import Span
from search_terms import expanded_dict
import spacy
import pytest
import copy
import re

with open("samples/hit_samplereconstructed.xml") as ifile:
    mark_up = BeautifulSoup(ifile.read())
# markup
matches = mark_up.find_all('match')
queries = mark_up.find_all('query')

# string representations
hits = [match.text for match in matches]
model_path = 'spacy_model/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
model = spacy.load(model_path)
sentencizer = model.create_pipe('sentencizer')
model.add_pipe(sentencizer)


def try_language_models(m1=None, m2=None, m3=None, debug=False, ix=[0]):
    """compares two or more language models on the first
    match in hits. If no model names are given, default
    is to test English, Norwegian and Multilingual
    Parameters:
               m1, m2, m3 (str):
                         name of or (if needed) path to model
                         directory
               debug (bool):
                         shows sentences generated by each model
                         if set to True
               ix (iter):
                         indeces of the examples to look at in hits
    """

    models = []
    if not m1 and not m2:
        m1 = "en_core_web_sm"
        m2 = "nb_core_news_sm"
        m3 = "xx_ent_wiki_sm"

    for m in [m1, m2, m3]:
        if m:
            model = spacy.load(m)
            sentencizer = model.create_pipe('sentencizer')
            model.add_pipe(sentencizer)
            models.append(model)
    for i in ix:
        if debug:
            print(f'at hit {i+1}')
        example = hits[i].strip()
        docs = []
        for model in models:
            docs.append(model(example))
        if len(docs) > 2:
            last = docs[2]
        else:
            last = None
            compare_segmentation(docs[0], docs[1], last, debug)
    if debug:
        print("Done!")
    return True


def compare_segmentation(a, b, c=None, debug=False):
    """check if two or three sentencizer segmentations are the same
    shows the corresponding sentences one by one if debug is True.
    Parameters:
               a, b, c (spacy.tokens.doc.Doc):
                      a Doc object created by running a pipeline
                      containing a sentecizer on a text.
                      (c is optional)
               debug (bool):
                      mode that displays sentences side by side
                      for each Doc.
    """

    a_gen = a.sents
    b_gen = b.sents
    if c:
        c_gen = c.sents
    for a_s in a_gen:
        # compare sentences on token basis
        a_s = [tok.text for tok in a_s]
        for b_s in b_gen:
            b_s = [tok.text for tok in b_s]
            if c and c_gen:
                for c_s in c_gen:
                    c_s = [tok.text for tok in c_s]
                    if debug:
                        print(f" 1) {a_s}\n 2) {b_s}\n 3) {c_s}\n")
                    else:
                        assert a_s == b_s,\
                            f"Sentences do not match up:\n 1) {a_s}\n 2) {b_s}"
                        assert a_s == c_s,\
                            f"Sentences do not match up:\n 1) {a_s}\n 3) {c_s}"
                    break
            elif debug:
                print(f" 1) {a_s}\n 2) {b_s}\n")
            else:
                assert a_s == b_s,\
                    f"Sentences do not match up:\n 1) {a_s}\n 2) {b_s}"
            break
    return True


def test_lms():
    """compare the two Swedish language models"""

    # we are interested in the Swedish models
    m1 = 'spacy_model/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
    m2 = 'spacy_model/sv_model_upos/sv_model0/sv_model_upos0-0.0.0/'

    # and potentially the multilingual one
    m3 = 'xx_ent_wiki_sm'
    assert try_language_models(m1, m2, ix=range(len(hits)))


def segment_match(match, new_match=None, highlight_query=False, context=2,
                  redo_boundaries=True, xml=True):
    """Split a whoosh match into sentences
    Parameters:
               match (bs4.element.Tag):
                     query match in context
               new_match (bs4.element.Tag):
                     empty match tag
               highlight_query (bool):
                     whether or not the query term(s) should be
                     highlighted within the matched sentence
               context (int):
                     how many sentences of context to display around
                     the matched sentence
               redo_boundaries (bool):
                     whether or not the original sentence boundaries
                     should be corrected to account for frequent
                     abbreviation errors

    for now uses the Swedish xpos model
    """

    full_text = match.text
    query_matches = None
    # potentially useful later if the match sequence gets segmented
    # into multiple sentences
    if match.b:
        query_matches = match("b")
    q_ex = re.compile(r" *(\b\w+\b *)?".join([m.text for m in query_matches]))

    match_sequence = match.em.text
    sents = []
    match_id = None
    match_by_term = None
    match_first_term = None
    sequence = list(model(full_text).sents)
    if redo_boundaries:
        sequence = redefine_boundaries(model(full_text))
    for i, sent in enumerate(sequence):
        if sent:
            sents.append(sent)
            if match_sequence in str(sent):
                match_id = i
            elif q_ex.search(str(sent)):
                match_by_term = i
            elif query_matches[0].text in str(sent):
                match_first_term = i
    # fall-back if the match sequence is now segmented differently
    if match_id is None and match_by_term is not None:
        match_id = match_by_term
    if match_id is None and match_first_term is not None:
        match_id = match_first_term
    if xml:
        return format_xml_match(sents, new_match, match_id,
                                context, highlight_query)
    else:
        return format_match(sents, match_id,
                            context, highlight_query)


def format_xml_match(sents, new_match, match_id, context, highlight_query):
    """format match in xml style. Returns a filled bs4.element.Tag
    Parameters:
               sents (list):
                     list of sentences (str)
               new_match (bs4.element.Tag):
                     empty match tag
               match_id (int):
                     id of the matched sentence within sents
               context (int):
                     how many sentences of context to display around
                     the matched sentence
               highlight_query (bool):
                     whether or not the query term(s) should be
                     highlighted within the matched sentence
    """
    soup = BeautifulSoup()
    if new_match is None:
        new_match = soup.new_tag("match")
    if match_id is not None:
        start = max(match_id - context, 0)
        end = min(match_id + context + 1, len(sents))
        new_match.append(" ".join([str(s) for s in sents[start:match_id]]))
        # correct for inconsistent tokenisation
        new_match.string = new_match.string
        # add a new tag for the highlighted sentence
        tag = soup.new_tag("em")
        tag.string = ""
        # highlight the query match if needed
        if highlight_query and query_matches:
            q_tags = {}
            tag_order = []
            for q in query_matches:
                q_tag = soup.new_tag("b")
                text = q.text
                q_tag.append(text)
                if text in q_tags:
                    q_tags[text].append(q_tag)
                else:
                    q_tags[text] = [q_tag]
                tag_order.append(text)
            for token in sents[match_id].split():
                text = str(token)
                if tag_order and text == tag_order[0]:
                    tag.append(" ")
                    tag.append(q_tags[text].pop(0))
                    tag_order.pop(0)
                    if not q_tags[text]:
                        del q_tags[text]
                else:
                    if text in ".,;:!?":
                        tag.append(text)
                    else:
                        tag.append(" " + text)
        else:
            tag.string = str(sents[match_id])
        new_match.append(tag)
        if end <= len(sents):
            new_match.append(
                f' {" ".join([str(s) for s in sents[match_id+1:end]])}')
        return new_match


def format_match(sents, match_id, context, highlight_query):
    """return match as match sentence and lists of left and
    right context sentences
    Parameters:
               sents (list):
                     list of sentences (str)
               match_id (int):
                     id of the matched sentence within sents
               context (int):
                     how many sentences of context to display around
                     the matched sentence
               highlight_query (bool):
                     whether or not the query term(s) should be
                     highlighted within the matched sentence
    """
    if match_id is not None:
        start = max(match_id - context, 0)
        end = min(match_id + context + 1, len(sents))
        left_context = [str(s) for s in sents[start:match_id]]
        # highlight the query match if needed
        # not sure if this is needed here
        if highlight_query and query_matches:
            pass
        else:
            match = sents[match_id]
        if end <= len(sents):
            right_context = [str(s) for s in sents[match_id+1:end]]
        return {"left": left_context, "right": right_context, "match": match}


def redefine_boundaries(sents):
    """correct sentence boundaries of spacy sentencizer
    based on rules for abbreviation and possessive markers
    Parameters:
               sents (spacy.tokens.span.Span):
                           a spacy sents generator of Span objects
    """

    ents = [str(ent) for ent in sents.ents]
    sents = list(sents.sents)
    abr_exp = re.compile(r"(m\.m|osv|etc)\.")
    poss_exp = re.compile(r"\w+:$")
    for i in range(len(sents)):
        if i >= len(sents):
            break
        has_abbrev = abr_exp.findall(str(sents[i]))[::-1]
        has_poss = poss_exp.findall(str(sents[i]))
        split_on_poss = (has_poss and
                         (i + 1 < len(sents) and str(sents[i+1])[:2] == "s "))
        if has_abbrev:
            pad = 0
            if type(sents[i]) == Span:
                tokens = list(sents[i].__iter__())
            else:
                tokens = sents[i].split()
            last = None
            while has_abbrev:
                nb_abbr = len(has_abbrev)
                if str(tokens[-1]) == "(":
                    pad = 1
                for j, t in enumerate(tokens):
                    if not has_abbrev:
                        break
                    if has_abbrev[-1] in str(t):
                        if j+1 < len(tokens) and\
                           (str(tokens[j+1]).istitle() and
                            str(tokens[j+1]) not in ents):
                            has_abbrev.pop(-1)
                            new_s = " ".join(
                                [tok.text for tok in tokens[j+1:]])
                            following = sents[i+1:]
                            sents[i] = " ".join(
                                [tok.text for tok in tokens[:j+1]])
                            sents[i+1] = new_s
                            sents = sents[:i+2]
                            sents.extend(following)
                if nb_abbr == len(has_abbrev):
                    has_abbrev.pop(-1)
        if split_on_poss:
            sents[i] = re.sub(r" ([.,;:!?])", r"\1",
                              str(sents[i]) + str(sents[i+1]))
            del sents[i+1]
        else:
            sents[i] = re.sub(r" ([.,;:!?])", r"\1", str(sents[i]))
    return sents


def format_output(matches, dest=None):
    """reformat query matches and write them to dest
    Parameter:
              matches (bs4.element.ResultSet):
                      result set of all matches in a document
              dest (BufferedWriter):
                      an opened file to write to (optional)
    """

    new_matches = BeautifulSoup()
    i = 0
    for match in matches:
        i += 1
        new_match = new_matches.new_tag("match")
        new_match.attrs = match.attrs
        segment_match(match, new_match, True)
        if new_match.text:
            new_matches.append(new_match)

    print(new_matches, file=dest)


def restructure_hit_sample():
    with open("new_format.xml", "w") as ofile:
        print("<xml>", file=ofile)
        format_output(matches, ofile)
        print("</xml>", file=ofile)


def hit_sample_to_txt(remove_non_kw=False):
    """print all matched sentences to text format and save
    context in separate file"""
    hits = "hit_sample.txt"
    context = "context.txt"
    if remove_non_kw:
        hits = f'filtered_{hits}'
        context = f'filtered_{context}'
    with open(hits, "w") as hits,\
         open(context, "w") as context:
        if remove_non_kw:
            for query in queries:
                if query['term'].split("(")[0] in expanded_dict:
                    matches = query.find_all("match")
                    for match in matches:
                        segments = segment_match(match, xml=False)
                        print(segments['match'].lstrip('\n'), file=hits)
                        print("\n".join(segments['left'] + segments['right']).lstrip('\n'),
                              file=context)
        else:
            for i, match in enumerate(matches):
                segments = segment_match(match, xml=False)
                print(segments['match'].lstrip('\n'), file=hits)
                print("\n".join(segments['left'] + segments['right']).lstrip('\n'),
                      file=context)


def compare_boundaries():
    for i, m in enumerate(matches):
        doc = model(m.text)
        s = list(doc.sents)
        sents = redefine_boundaries(doc)
        print(f"{i}:", len(s), len(sents))
        for j in range(max(len(s), len(sents))):
            if j < len(s):
                print("1:", s[j])
            else:
                print("1:")
            if j < len(sents):
                print("2:", sents[j])
            else:
                print("2:")
            print()
        if input("continue? (Y/n)\n> ").casefold() == "n":
            break

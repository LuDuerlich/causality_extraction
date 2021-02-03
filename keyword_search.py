from data_extraction import Text
from bs4 import BeautifulSoup
import copy
from html.entities import html5
import os
from postprocessing import model, redefine_boundaries
import tarfile
import re
import sys
# sys.path.append("/Users/luidu652/Documents/causality_extraction/")
from search_terms import search_terms, expanded_dict, incr_dict, decr_dict
# sys.path.append("/Users/luidu652/Documents/causality_extraction/whoosh/src/")
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED,\
    NGRAM, NUMERIC, FieldType
from whoosh.qparser import QueryParser
from whoosh.highlight import *
from whoosh import index, query, analysis
import random
import glob
import pickle
import unicodedata

analyzer = StandardAnalyzer(stoplist=[])
schema = Schema(doc_title=TEXT(stored=True, analyzer=analyzer,
                               lang='se'),
                sec_title=TEXT(stored=True, analyzer=analyzer,
                               lang='se'),
                target=TEXT(stored=True, phrase=True,
                            analyzer=analyzer, lang='se'),
                left_context=TEXT(stored=True, phrase=False,
                                  analyzer=analyzer, lang='se'),
                right_context=TEXT(stored=True, phrase=False,
                                   analyzer=analyzer, lang='se'),
                sent_nb=NUMERIC(stored=True, sortable=True))

htmlchars2ents = {char: entity for entity, char in html5.items()}
xmlchars2ents = {'"': 'quot;',
                 "'": 'apos;',
                 "<": 'lt;',
                 ">": 'gt;'}


def fix_file(filename):
    """replace magic characters in a markup file with entity characters"""
    if filename.endswith('.html'):
        markup = 'html'
    elif filename.endswith('.xml'):
        markup = 'xml'
    with open(filename) as ifile:
        soup = BeautifulSoup(ifile.read(), parser=markup)
    text = write_markup(str(soup), markup)
    out = filename.split('/')
    out = f"{'/'.join(out[:-1])}/new_{out[-1]}"
    print(out)
    with open(out, 'w') as ofile:
        ofile.write(text)


def write_markup(string, format='xml'):
    """replace all reserved html/xml characters with their character entity"""
    has_semicolon = ';' in string
    # get these two out of the way before considering other entities
    if '&' in string:
        string = string.replace('&', '&' + htmlchars2ents['&'])
    if format == 'html':
        if has_semicolon:
            string = string.replace(';', '&' + htmlchars2ents[';'])
            string = string.replace('&amp&semi;', '&amp;')

        for char in string:
            if char in htmlchars2ents and char not in '&;':
                string = string.replace(char, f'&{htmlchars2ents[char]}')
    elif format == 'xml':
        for char in string:
            if char in xmlchars2ents and char != '&':
                string = string.replace(f' {char} ',
                                        f' &{xmlchars2ents[char]} ')
    # account for highlighting and other markup
    if format == 'html':
        string = re.sub(
            r'&lt;link( \w+&equals;[&;a-zA-Z0-9]+)+&quot;&sol;&gt;(&NewLine;)?',
            '<link rel="stylesheet" href="style.css"/>', string)

    string = re.sub(r'&lt;&sol;([A-Za-z1-9]+)&gt;', r'&lt;/\1&gt;', string)
    string = re.sub(r'&lt;(/?[A-Za-z1-9]+)&gt;&NewLine;', r'<\1>\n', string)
    string = re.sub(r'&lt;(/?[A-Za-z1-9]+)&gt;', r'<\1>', string)
    return string


def strip_tags(string):
    """strip title tags from string"""
    return re.sub("<h1>|</h1>", "", string)


class BasicTokenizer(analysis.Composable):
    """
    This is an amputated version of huggingface transformers BasicTokenizer
    to have the keyword model operate on units that are compatible with the
    BERT model to be used later on

    Constructs a BasicTokenizer that will run basic tokenization (punctuation
    splitting, lower casing, etc.).

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization.
            Only has an effect when
            :obj:`do_basic_tokenize=True`
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified,
            then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    def __init__(self, do_lower_case=True, never_split=None,
                 tokenize_chinese_chars=True, strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.whitespace = re.compile(r"[\s\r\n\t]+")

    def __call__(self, value, never_split=None, positions=False, chars=False,
                 keeporiginal=False, removestops=True, start_pos=0, start_char=0,
                 tokenize=True, mode='', **kwargs):
        return self.tokenize(value, never_split, positions, chars,
                             removestops=removestops, mode=mode,
                             **kwargs)

    def tokenize(self, value, never_split=None, positions=False, chars=False,
                 keeporiginal=False, removestops=True, start_pos=0, start_char=0,
                 tokenize=True, mode='', **kwargs):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = (self.never_split.union(set(never_split))
                       if never_split else self.never_split)
        orig_text = value
        text = self._clean_text(value)
        orig_tokens = self.whitespace.finditer(text)
        split_tokens = []
        tokens = []
        t = Token(positions, chars,
                  removestops=removestops, mode=mode,
                  **kwargs)
        prevend = 0
        pos = start_pos
        for token in orig_tokens:
            if token[0] not in never_split:
                start = prevend
                end = token.start()
                text = value[start:end]
                if text:
                    t.original = text
                    t.boost = 1.0
                    t.stopped = False
                    if self.strip_accents:
                        text = self._run_strip_accents(text)
                        end = start + len(text)
                    t.text = text
                    t.startchar = start_char + start
                    t.endchar = start_char + end
                    t.pos = pos
                    pos += 1
                    split_tokens.extend(
                        self._run_split_on_punc(t,
                                                never_split,
                                                **kwargs))
            t = Token(positions, chars, removestops=removestops, mode=mode,
                      **kwargs)
            prevend = token.end()
        if prevend < len(value):
            if not hasattr(t, "text"):
                t.text = value[prevend:]
            t.boost = 1.0
            t.original = t.text
            t.stopped = False
            t.pos = pos
            t.startchar = prevend
            t.endchar = len(value)
            if chars:
                t.startchar = prevend
                t.endchar = len(value)
            if hasattr(t, "text") and t.text:
                split_tokens.extend(
                    self._run_split_on_punc(t,
                                            never_split,
                                            **kwargs))
        for t in split_tokens:
            yield t

    def _is_punctuation(self, char):
        """Checks whether `char` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _is_control(self, char):
        """Checks whether `char` is a control character."""
        # These are technically control characters but we count
        # them as whitespace characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, t, never_split=None, **kwargs):
        """Splits punctuation on a piece of text."""
        if not hasattr(t, "text"):
            print(t)
        if never_split is not None and t.text in never_split:
            # return [t]
            yield t
        chars = list(t.text)
        i = 0
        start_new_word = True
        current_tok = ""
        tok_start = 0
        punct_tok = Token(t.positions, t.chars,
                          removestops=t.removestops,
                          mode=t.mode)
        punct_tok.endchar = t.startchar + i
        punct_tok.boost = t.boost
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                if current_tok:
                    yield Token(t.positions,
                                t.chars,
                                removestops=t.removestops,
                                mode=t.mode,
                                text=current_tok,
                                boost=t.boost,
                                startchar=t.startchar + tok_start,
                                endchar=t.startchar + i,
                                pos=t.pos + tok_start)
                    tok_start = i + 1
                    current_tok = ""
                punct_tok = Token(t.positions, t.chars,
                                  removestops=t.removestops,
                                  mode=t.mode)
                punct_tok.boost = t.boost
                punct_tok.startchar = t.startchar + i
                punct_tok.text = char
                punct_tok.endchar = t.startchar + i + 1
                punct_tok.pos = t.pos + tok_start
                yield punct_tok
                tok_start = i + 1
                start_new_word = True
                punct_tok = Token(t.positions, t.chars,
                                  removestops=t.removestops,
                                  mode=t.mode)
            else:
                if current_tok:
                    current_tok += char
                else:
                    current_tok = char
            i += 1
        if hasattr(punct_tok, "text"):
            punct_tok.endchar = t.startchar + i
            yield punct_tok
            tok_start += 1
        if current_tok:
            yield Token(t.positions, t.chars,
                        removestops=t.removestops,
                        mode=t.mode,
                        text=current_tok,
                        boost=t.boost,
                        startchar=t.startchar + tok_start,
                        endchar=t.startchar + i,
                        pos=t.pos + tok_start)
            tok_start += 1

    def _clean_text(self, text):
        """Performs invalid character removal
        and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                # maintain token positions in raw raw texr
                output.append("#")
            else:
                output.append(char)
        return "".join(output)


class MyHighlighter(Highlighter):
    def __init__(self, fragmenter=None, scorer=None, formatter=None,
                 always_retokenize=False, order=FIRST,
                 analyzer=None):
        super().__init__(fragmenter, scorer, formatter,
                         always_retokenize, order)
        if analyzer:
            self.analyzer = analyzer
        else:
            self.analyzer = StandardAnalyzer(minsize=1, stoplist=[])

    def can_load_chars(self, results, fieldname):
        # Is it possible to build a mapping between the matched terms/docs and
        # their start and end chars for "pinpoint" highlighting (ie not require
        # re-tokenizing text)?

        if self.always_retokenize:
            # No, we've been configured to always retokenize some text
            return False
        if not results.has_matched_terms():
            # No, we don't know what the matched terms are yet
            return False
        if self.fragmenter.must_retokenize():
            # No, the configured fragmenter doesn't support it
            return False

        # Maybe, if the field was configured to store characters
        field = results.searcher.schema[fieldname]
        return field.supports("characters")

    @staticmethod
    def _load_chars(results, fieldname, texts, to_bytes):
        # For each docnum, create a mapping of text -> [(startchar, endchar)]
        # for the matched terms

        results._char_cache[fieldname] = cache = {}
        sorted_ids = sorted(docnum for _, docnum in results.top_n)

        for docnum in sorted_ids:
            cache[docnum] = {}

        for text in texts:
            btext = to_bytes(text)
            m = results.searcher.postings(fieldname, btext)
            docset = set(results.termdocs[(fieldname, btext)])
            for docnum in sorted_ids:
                if docnum in docset:
                    m.skip_to(docnum)
                    assert m.id() == docnum
                    cache[docnum][text] = m.value_as("characters")

    @staticmethod
    def _merge_matched_tokens(tokens):
        # Merges consecutive matched tokens together, so they are highlighted
        # as one

        token = None

        for t in tokens:
            if not t.matched:
                if token is not None:
                    yield token
                    token = None
                yield t
                continue

            if token is None:
                token = t.copy()
            elif t.startchar <= token.endchar:
                if t.endchar > token.endchar:
                    token.text += t.text[token.endchar - t.endchar:]
                    token.endchar = t.endchar
            else:
                yield token
                token = None
                # t was not merged, also has to be yielded
                yield t

        if token is not None:
            yield token

    def highlight_hit(self, hitobj, fieldname, text=None,
                      top=3, minscore=1, strict_phrase=False):
        results = hitobj.results
        schema = results.searcher.schema
        field = schema[fieldname]
        to_bytes = field.to_bytes
        from_bytes = field.from_bytes

        if text is None:
            if fieldname not in hitobj:
                raise KeyError("Field %r is not stored." % fieldname)
            text = hitobj[fieldname]

        # Get the terms searched for/matched in this field
        if results.has_matched_terms():
            bterms = (term for term in results.matched_terms()
                      if term[0] == fieldname)
        else:
            bterms = results.query_terms(expand=True, fieldname=fieldname)
        # Convert bytes to unicode
        words = frozenset(from_bytes(term[1]) for term in bterms)

        # If we can do "pinpoint" highlighting...
        if self.can_load_chars(results, fieldname):
            # Build the docnum->[(startchar, endchar),] map
            if fieldname not in results._char_cache:
                self._load_chars(results, fieldname, words, to_bytes)

            hitterms = (from_bytes(term[1]) for term in hitobj.matched_terms()
                        if term[0] == fieldname)

            # Grab the word->[(startchar, endchar)] map for this docnum
            cmap = results._char_cache[fieldname][hitobj.docnum]
            # A list of Token objects for matched words
            tokens = []
            charlimit = self.fragmenter.charlimit
            for word in hitterms:
                chars = cmap[word]
                for pos, startchar, endchar in chars:
                    if charlimit and endchar > charlimit:
                        break
                    tokens.append(Token(text=word, pos=pos,
                                        startchar=startchar, endchar=endchar))
            tokens.sort(key=lambda t: t.startchar)
            tokens = [max(group, key=lambda t: t.endchar - t.startchar)
                      for key, group in groupby(tokens, lambda t: t.startchar)]
            if fieldname == 'target':
                text = [[hitobj['left_context'], hitobj['target'],
                         hitobj['right_context']]]
                fragments = self.fragmenter.fragment_tokens(text,
                                                            tokens,
                                                            split_fields=True)
            else:
                fragments = self.fragmenter.fragment_matches(text, tokens)
        else:
            # Retokenize the text
            analyzer = results.searcher.schema[fieldname].analyzer
            # print("ANALYZER:", analyzer)
            tokens = analyzer(text, positions=True, chars=True, mode="index",
                              removestops=False)

            # Set Token.matched attribute for tokens that match a query term
            if strict_phrase:
                terms, phrases = results.q.phrases()
                tokens = set_matched_filter_phrases(tokens, text, terms,
                                                    phrases, self.analyzer)
            else:
                tokens = set_matched_filter(tokens, words)
            tokens = self._merge_matched_tokens(tokens)
            if fieldname == 'target':
                text = [hitobj['left_context'], hitobj['target'],
                        hitobj['right_context']]
                fragments = self.fragmenter.fragment_tokens(text,
                                                            tokens,
                                                            split_fields=True)
            else:
                fragments = self.fragmenter.fragment_tokens(text, tokens)
        fragments = top_fragments(fragments, top, self.scorer, self.order,
                                  minscore=minscore)
        output = self.formatter.format(fragments)
        return output


class ContextFragment(Fragment):
    """Fragment of a matched sentence with n sentences of context"""

    def __init__(self, text, matches, startchar=0, endchar=-1,
                 sent_boundaries=(0, -1)):
        super().__init__(text, matches,  startchar, endchar)
        self.sent_boundaries = sent_boundaries


class MyFormatter(Formatter):
    """HTML formatter
    highlights match sentence and term with HTML style tags."""

    def __init__(self, between="..."):
        """
        :param between: the text to add between fragments.
        """

        self.between = between

    def format_fragment(self, fragment, replace=False):
        """Returns an emphasised formatted version of the given text,
        using the "token" objects in the given :class:`Fragment`.

        :param fragment: a :class:`Fragment` object representing a list of
            matches in the text.
        :param replace: if True, the original text corresponding to each
            match will be replaced with the value of the token object's
            ``text`` attribute.
        """

        text = fragment.text
        index = fragment.sent_boundaries[0]
        # if not index:
        #     print(fragment.sent_boundaries)
        match_s_end = fragment.sent_boundaries[-1]
        output = [self._text(text[fragment.startchar:index])]
        output.append("<em>")
        for t in fragment.matches:
            if t.startchar is None:
                continue
            if t.startchar < index:
                continue
            if t.startchar > index:
                output.append(self._text(text[index:t.startchar]))
            output.append(self.format_token(text, t, replace))
            index = t.endchar
        output.append(self._text(text[index:match_s_end+1]))
        output.append("</em>")
        output.append(self._text(text[match_s_end+1:fragment.endchar+1]))
        # print('formatted:', "".join(output), fragment.sent_boundaries[0])
        return "".join(output), fragment.sent_boundaries[0]

    def format(self, fragments, replace=False):
        """Returns a bold formatted version of the given text, using a list of
        :class:`Fragment` objects.
        """

        formatted = list(filter(None, [self.format_fragment(f, replace=replace)
                                       for f in fragments]))
        return formatted

    def format_token(self, text, token, replace=False):
        return f"<b>{get_text(text, token, replace)}</b>"


def mksentfrag(text, tokens, startchar=None, endchar=None,
               charsbefore=0, charsafter=0, match_s_boundaries=(0, -1)):
    """Returns a :class:`Fragment` object based on the :class:`analysis.Token`
    objects in ``tokens`.
    """

    if startchar is None:
        startchar = tokens[0].startchar if tokens else 0
    if endchar is None:
        endchar = tokens[-1].endchar if tokens else len(text)
    startchar = max(0, startchar - charsbefore)
    endchar = min(len(text), endchar + charsafter)
    f = ContextFragment(text, tokens, startchar, endchar, match_s_boundaries)
    # print("FRAGMENT:", startchar, endchar,
    # match_s_boundaries, len(text), tokens)
    # print(text[startchar:endchar])
    return f


class MySentenceFragmenter(Fragmenter):
    """Breaks the text up on sentence end punctuation characters
    (".", "!", or "?"). This object works by looking in the original text for a
    sentence end as the next character after each token's 'endchar'.

    When highlighting with this fragmenter, you should use an analyzer that
    does NOT remove stop words, for example::

        sa = StandardAnalyzer(stoplist=None)
    """

    def __init__(self, maxchars=1000, sentencechars=".!?",
                 charlimit=None, context_size=2):
        """
        :param maxchars: The maximum number of characters allowed in a
            fragment.
        :param context_size: How many sentences left and right of the
            match to display.
        """

        self.maxchars = maxchars
        self.sentencechars = frozenset(sentencechars)
        self.charlimit = charlimit
        self.context_size = context_size

    def fragment_tokens(self, text, tokens):
        maxchars = self.maxchars
        sentencechars = self.sentencechars
        charlimit = self.charlimit
        context = self.context_size

        textlen = len(text)
        # startchar of first token in the current sentence
        first = None
        # Buffer for matched tokens in the current sentence
        tks = []
        sents = []
        is_right_context = False
        match_sent_id = []
        last_tks = []
        endchar = None
        # Number of chars in the current sentence
        currentlen = 0

        for t in tokens:
            startchar = t.startchar
            endchar = t.endchar
            if charlimit and endchar > charlimit:
                break
            if first is None:
                # correct previous sentence boundary
                if (sents and (t.text[0] in sentencechars or
                               t.text[-1] in sentencechars or
                               (t.endchar < len(text) and
                                text[t.endchar] in sentencechars))):
                    sents[-1] = (sents[-1][0], endchar)
                    if t.matched:
                        tks.append(t.copy())
                        match_sent_id.append(len(sents)-1)
                    continue
                elif (re.match("[a-zäöå]", text[t.startchar]) and sents):
                    first = sents[-1][0]
                    sents.pop(-1)
                    if t.matched:
                        tks.append(t.copy())
                        match_sent_id.append(len(sents))
                    continue

                else:
                    # Remember the startchar of the first token in a sentence
                    first = startchar
                    currentlen = 0

            tlength = endchar - startchar
            currentlen += tlength

            if t.matched:
                # print("MATCH", t)
                tks.append(t.copy())
            # If the character after the current token is end-of-sentence
            # punctuation, finish the sentence and reset
            if endchar < textlen and text[endchar] in sentencechars:
                # Don't break for two periods in a row (e.g. ignore "...")
                if endchar + 1 < textlen and\
                   text[endchar + 1] in sentencechars:
                    continue
                # If the sentence had matches and it's not too long,
                # save it and process the next sentences until the edge
                # of the context window is reached
                if tks and currentlen <= maxchars:
                    if not sents:
                        # insert dummy sent before actual sent
                        sents.append((first, endchar))
                    match_sent_id.append(len(sents))
                    is_right_context = True
                    last_tks.append(tks)
                if sents and is_right_context and\
                   match_sent_id[-1] + context == len(sents):
                    for i in match_sent_id:
                        if not last_tks:
                            break
                        current_endchar = sents[min(i+context,
                                                    len(sents) - 1)][-1]
                        # account for matches at the beginning of the document
                        current_startchar = sents[max(i-context, 0)][0]
                        yield mksentfrag(text, last_tks.pop(0),
                                         startchar=current_startchar,
                                         endchar=min(current_endchar, endchar),
                                         match_s_boundaries=sents[i])
                    # reset the variables for each match
                    match_sent_id = []
                    is_right_context = False
                # Reset the counts within a sentence
                sents.append((first, endchar))
                tks = []
                first = None
                currentlen = 0
        # If we get to the end of the text and there's still a sentence
        # in the buffer, yield it
        if first:
            sents.append((first, endchar))
        if tks:
            last_tks.append(tks)
            match_sent_id.append(len(sents)-1)
        if last_tks:
            match_sent_id = sorted(set(match_sent_id))
            for i in match_sent_id:
                if not last_tks:
                    break
                if match_sent_id:
                    # account for matches at the very beginning of the document
                    start_s = min(i, len(sents))
                    if not sents:
                        print(start_s, context, sents, last_tks, match_sent_id)
                        break

                    if start_s - context > len(sents):
                        print(start_s, context, sents)
                    current_startchar = sents[max(start_s - context, 0)][0]
                else:
                    startchar = sents[max(len(sents) - context, 0)][0]
                current_endchar = sents[min(i+context, len(sents) - 1)][-1]
                yield mksentfrag(text, last_tks.pop(0),
                                 startchar=sents[max(i-context, 0)][0],
                                 endchar=min(current_endchar, endchar),
                                 match_s_boundaries=sents[i])


class MyOldSentenceFragmenter(Fragmenter):
    """Breaks the text up on sentence end punctuation characters
    (".", "!", or "?"). This object works by looking in the original text for a
    sentence end as the next character after each token's 'endchar'.

    When highlighting with this fragmenter, you should use an analyzer that
    does NOT remove stop words, for example::

        sa = StandardAnalyzer(stoplist=None)
    """

    def __init__(self, maxchars=1000, sentencechars=".!?",
                 charlimit=None, context_size=2):
        """
        :param maxchars: The maximum number of characters allowed in a
            fragment.
        :param context_size: How many sentences left and right of the
            match to display.
        """

        self.maxchars = maxchars
        self.sentencechars = frozenset(sentencechars)
        self.charlimit = charlimit
        self.context_size = context_size

    def fragment_tokens(self, text, tokens, split_fields=False):
        maxchars = self.maxchars
        sentencechars = self.sentencechars
        charlimit = self.charlimit
        context = self.context_size
        if split_fields:
            # left context
            left_text = text[0].split("###")
            left_c = "\n".join(left_text[max(0, len(left_text)-context):])

            # right context
            right_text = text[-1].split("###")
            right_c = "\n".join(right_text[:min(len(right_text), context)])
            target = text[1]
            text[0] = '\n'.join(left_text)
            text[-1] = '\n'.join(right_text)
            text = ' '.join(text)
            boundaries = [text.index(target), text.index(target) + len(target)]
            # fix character positions
            matched_tokens = []
            for t in tokens:
                if t.matched:
                    matched_tokens.append(t.copy())
                    matched_tokens[-1].startchar += boundaries[0]
                    matched_tokens[-1].endchar += boundaries[0]
            yield mksentfrag(text, matched_tokens,
                             startchar=text.index(left_c),
                             endchar=text.index(right_c) + len(right_c),
                             match_s_boundaries=boundaries)
        else:
            textlen = len(text)
            # startchar of first token in the current sentence
            first = None
            # Buffer for matched tokens in the current sentence
            tks = []
            sents = []
            is_right_context = False
            match_sent_id = []
            last_tks = []
            endchar = None
            # Number of chars in the current sentence
            currentlen = 0

            for t in tokens:
                startchar = t.startchar
                endchar = t.endchar
                if charlimit and endchar > charlimit:
                    break

                if first is None:
                    # Remember the startchar of the first token in a sentence
                    first = startchar
                    currentlen = 0

                tlength = endchar - startchar
                currentlen += tlength

                if t.matched:
                    tks.append(t.copy())

                # If the character after the current token is end-of-sentence
                # punctuation, finish the sentence and reset
                if endchar < textlen and text[endchar] in sentencechars:
                    # Don't break for two periods in a row (e.g. ignore "...")
                    if endchar + 1 < textlen and\
                       text[endchar + 1] in sentencechars:
                        continue
                    # If the sentence had matches and it's not too long,
                    # save it and process the next sentences until the edge
                    # of the context window is reached
                    if tks and currentlen <= maxchars:
                        if not sents:
                            # insert dummy sent before actual sent
                            sents.append((first, endchar))
                        match_sent_id.append(len(sents))
                        is_right_context = True
                        last_tks.append(tks)
                    if sents and is_right_context and\
                       match_sent_id[-1] + context == len(sents):
                        for i in match_sent_id:
                            current_endchar = sents[min(i+context,
                                                        len(sents) - 1)][-1]
                            # account for matches at the beginning of the
                            # document
                            current_startchar = sents[max(i-context, 0)][0]
                            yield mksentfrag(text, last_tks.pop(0),
                                             startchar=current_startchar,
                                             endchar=min(current_endchar,
                                                         endchar),
                                             match_s_boundaries=sents[i])
                        # reset the variables for each match
                        match_sent_id = []
                        is_right_context = False
                    # Reset the counts within a sentence
                    sents.append((first, endchar))
                    tks = []
                    first = None
                    currentlen = 0
            # If we get to the end of the text and there's still a sentence
            # in the buffer, yield it
            if tks:
                last_tks.append(tks)
                match_sent_id.append(len(sents))
                sents.append((first, endchar))
            if last_tks:
                for i in match_sent_id:
                    current_endchar = sents[min(i+context, len(sents) - 1)][-1]
                    yield mksentfrag(text, last_tks.pop(0),
                                     startchar=sents[max(i-context, 0)][0],
                                     endchar=min(current_endchar, endchar),
                                     match_s_boundaries=sents[i])


def create_index(path="test_index/", ixname="test", random_files=False,
                 schema=schema, add=False, filenames=None, n=5):
    """Create or add to a specified index of files in documents.tar.

    Parameters:
               path (str):
                          path to index directory (should already exist)
                          (default='test_index'
               ixname (str):
                          name of the index to create / access
               random_files (bool):
                          whether or not to sample random files from
                          documents
               schema (Schema):
                          index schema specifying the fields within the
                          documents and how to index and access them
               add (bool):
                          whether or not to add the same previously
                          sampled files to the index
                          (useful to compare different index settings)
               n (int):
                          how many context sentences to store
                          (if None, stores all the context available
                          within the section)
    """

    if not path.endswith("/"):
        path += "/"
    if os.path.exists(path) and os.listdir(path):
        if input(
                "index directory not empty. delete content?(Y/n) > "
        ).casefold() != "y":
            ix = index.open_dir(path, indexname=ixname)
        else:
            print(f"clearing out {path} ...")
            os.system(f"rm -r {path}*")
            ix = index.create_in(path, schema, indexname=ixname)
    else:
        if not os.path.exists(path):
            print(f'creating directory {path} ...')
            os.system(f'mkdir {path}')
        ix = index.create_in(path, schema, indexname=ixname)
    writer = ix.writer()
    if random_files:
        with open("ix_files.pickle", "rb") as ifile:
            seen = pickle.load(ifile)
            print(f'loading {len(seen)} seen files!')
        if not add:
            files = seen
        else:
            with tarfile.open("documents.tar", "r") as ifile:
                summaries = [fi for fi in ifile.getnames() if
                             fi.startswith('documents/s_')]
            files = random.sample([el for el in summaries
                                   if el not in seen], 500)
            # I don't remember why this is here
            seen.extend(files)
            with open("bt_ix_files.pickle", "wb") as ofile:
                pickle.dump(seen, ofile)
    elif filenames:
        print('index selected files')
        files = filenames
    else:
        files = ["H2B34", "H2B340", "H2B341", "H2B341", "H2B342",
                 "H2B343", "H2B344", "H2B345", "H2B346", "H2B347",
                 "H2B348", "H2B349", "H2B35"]
    with tarfile.open("documents.tar", "r") as ifile:
        n_files = len(files)
        print(f'{n_files} to be indexed')
        for i, key in enumerate(files):
            text = Text("")
            if key.endswith("html"):
                text.from_html(ifile.extractfile(key).read())
                key = key.split("_")[1].split(".")[0]
            else:
                text.from_html(
                    ifile.extractfile(
                        f"documents/s_{key}.html"
                    ).read())
            for j, section in enumerate(text):
                document = model(" ".join(section.text))
                sents = redefine_boundaries(document)
                for k, sent in enumerate(sents):
                    left_ctxt = ['']
                    right_ctxt = ['']
                    if k > 0:
                        if n is None:
                            left_ctxt = sents[:k]
                        else:
                            left_ctxt = sents[max(k-n, 0):k]
                    if k + 1 < len(sents):
                        if n is None:
                            right_ctxt = sents[k+1:]
                        else:
                            right_ctxt = sents[k+1:min(len(sents), k+1+n)]
                    target = str(sents[k])
                    title = re.sub(r'\s+', ' ', section.title)
                    right_ctxt = re.sub(r'\s+', ' ', "###".join(right_ctxt))
                    left_ctxt = re.sub(r'\s+', ' ', "###".join(left_ctxt))
                    # apparently, whoosh does not preserve newlines, so we have to
                    # mark sentence boundaries another way
                    writer.add_document(doc_title=key,
                                        sec_title=title,
                                        left_context=left_ctxt,
                                        right_context=right_ctxt,
                                        target=target,
                                        sent_nb=k)
            if i % 50 == 0:
                print(f'at file {i} ({text.title, k}), it has {j+1} sections')
    writer.commit()


def escape_xml_refs(text):
    return re.sub(" > ", " &gt; ",
                  re.sub(" < ", " &lt; ",
                         re.sub("&", "&amp;", text)))


def print_to_file(keywords=["orsak", '"bidrar till"'], terms=[""], field=None):
    """Print all examples matching the  query term to an XML file.

    Parameters:
       keywords (list):
                        keywords to search for; here intended to indicate
                        causality (default=['orsak', '"bidrar till"'])
                        (note formatting for strict phrases with extra '"'
       terms (list):
                        additional terms to search for; e.g. 'klimatförendring'
                        (required: no)
       field (str):
                        the field to search if none is specified, 'target' or
                        'body' are searched depending on the schema
    """

    print('field:', field)
    if field is None:
        if 'target' in ix.schema._fields:
            field = 'target'
        elif 'body' in ix.schema._fields:
            field = 'body'
        else:
            raise Exception(f'Unknown schema without field specifier!' +
                            'If the index schema does not have a target' +
                            'or body field, specify another field to search!')
    print('field:', field)
    qp = QueryParser(field, schema=ix.schema)
    # sentf = MySentenceFragmenter(maxchars=1000)
    sentf = MyOldSentenceFragmenter(maxchars=100000, context_size=4)
    formatter = MyFormatter(between="\n...")
    highlighter = MyHighlighter(fragmenter=sentf,
                                scorer=BasicFragmentScorer(),
                                formatter=formatter)

    # highlighter = MyHighlighter(fragmenter=sentf,
    #                             analyzer=analyzer,
    #                             scorer=BasicFragmentScorer(),
    #                             formatter=formatter)
    punct = re.compile(r"[!.?]")
    filename = "example_queries_inflected.xml"
    print("Index:", ix, field)
    if terms[0]:
        filename = f"{terms[0]}_example_queries_inflected.xml"
    with ix.searcher() as s,\
         open(filename, "w") as output:
        print("<xml>", file=output)
        for term in terms:
            if term:
                _query = f"{term} AND ({'~2 OR '.join(keywords)})"
            else:
                _query = '~2 OR '.join(keywords)
            print(f"<query term='{_query}'>", file=output)
            parsed_query = qp.parse(_query)
            r = s.search(parsed_query, terms=True, limit=None)
            if field == 'target':
                matches = []
                print('results:', len(r))
                for m in r:
                    hits = highlighter.highlight_hit(
                        m, field, top=len(m.results),
                        strict_phrase='"' in _query)
                    for hit, start_pos in hits:
                        matches.append((hit, m, m['sent_nb']))
            else:
                matches = [(hit, m, start_pos) for m in r
                           for hit in highlighter.highlight_hit(
                                   m, field, top=len(m.results),
                                   strict_phrase='"' in _query)]

            for i, matched_s in enumerate(matches):
                print(format_match(matched_s[:-1], i), file=output)
                # matched_s.results.order = FIRST
            print("</query>", file=output)
        print("</xml>", file=output)


def print_sample_file(keywords=expanded_dict, same_matches=None):
    """Sample 10 matches per query term and print them to an XML file.

    Parameters:
       keywords (dict):
                        dictionary of inflected terms organised by their
                        uninflected form (default=expanded_dict)
       same_matches (dict):
                        dictionary of uninflected terms, their correspond-
                        ing matches and metadata. Use only if same queries
                        should be sampled (default=None)
    """

    filename = "hit_sample.xml"
    if same_matches:
        with open("annotations/match_ids.pickle", "rb") as ifile:
            match_order = pickle.load(ifile)
        filename = filename.split(".")[0] + "_reconstructed.xml"
    qp = QueryParser("body", schema=ix.schema)
    sentf = MyOldSentenceFragmenter(maxchars=1000, context_size=4)
    # analyzer = BasicTokenizer()  # | analysis.LowercaseFilter()
    # analyzer = StandardAnalyzer(stoplist=[])
    # print(analyzer)
    formatter = MyFormatter(between="\n...")
    highlighter = Highlighter(fragmenter=sentf,
                              scorer=BasicFragmentScorer(),
                              formatter=formatter)
    # highlighter = MyHighlighter(fragmenter=sentf,
    #                             analyzer=analyzer,
    #                             scorer=BasicFragmentScorer(),
    #                             formatter=formatter)
    punct = re.compile(r"[!.?]")
    with ix.searcher() as s,\
         open(filename, "w") as output:
        total_matches = 0
        print("<xml>", file=output)
        for key in list(expanded_dict):
            _query = '~2 OR '.join(expanded_dict[key])
            print(f"<query term='{key}({_query})'>", file=output)
            parsed_query = qp.parse(_query)
            r = s.search(parsed_query, terms=True, limit=None)
            matches = []
            nb_r = len(r)
            matches = [(hit, m, start_pos) for m in r
                       for hit in highlighter.highlight_hit(
                               m, "body", top=len(m.results),
                               strict_phrase='"' in _query)]
            total_matches += len(matches)

            # limit to ten matches only
            print("nb of matches before sampling:", len(matches))
            if same_matches:
                match_ids = find_matched(matches,
                                         same_matches[key],
                                         match_order[key])
            elif len(matches) > 10:
                match_ids = random.sample(matches, 10)
            else:
                match_ids = list(range(len(matches)))
            print(f"Query {key}: {len(matches)} matches")
            for i, nb in match_ids:
                i = int(i)
                if i < len(matches):
                    print(format_match(matches[i][:-1], nb, i), file=output)
            print("</query>", file=output)
        print("</xml>", file=output)
    print(f"{total_matches} total matches")


def find_matched(matches, match_dict, order):
    """match hits to previously sampled matches by document and section ids
    and target sentence and order by original match number.

    This is necessary because the previously retrieved match numbers do not
    align in all cases. (May unfortunately also retrieve duplicate senteces
    within the same section)"""

    match_ids = []
    same_matched = copy.deepcopy(match_dict)
    for i, m in enumerate(matches):
        for match in same_matched:
            if m[1]["doc_title"] == match['doc'] and\
               m[1]['sec_title'] == match['st'] and\
               match['t'] in m[0]:
                match_ids.append([i, match['nb']])
                same_matched.remove(match)
        if len(match_ids) == 10:
            ordered = []
            print(len(match_ids), match_ids, order)
            for o in order:
                for mi in match_ids:
                    if str(o) == mi[1]:
                        ordered.append(mi)
            return ordered
    print(len(match_ids))
    ordered = []
    for o in order:
        for mi in match_ids:
            if str(o) == mi[1]:
                ordered.append(mi)

    return ordered


def extract_sample():
    """extract matches and metadata from previously sampled queries"""

    with open("samples/hit_sample.xml") as ifile:
        soup = BeautifulSoup(ifile.read(), features='lxml')
    queries = {}
    for query in soup.find_all("query"):
        matches = []
        for match in query.find_all("match"):
            matches.append({"nb": match['match_nb'],
                            "doc": match['doc'],
                            "st": match['section'],
                            "t": "".join([str(el) for el in
                                          match.em.contents])})
        queries[query['term'].split("(")[0]] = matches
    return queries


def format_match(match, match_nb, org_num=None, format_='xml'):
    if format_ == 'xml':
        tag = 'match'
    elif format_ == 'html':
        tag = 'p'

    title = escape_xml_refs(match[1]['doc_title'])
    sec_title = escape_xml_refs(match[1]['sec_title'])
    xml_match = f"<{tag} match_nb='{match_nb}"
    if org_num:
        xml_match += f"({org_num})' "
    else:
        xml_match += f"' "
    xml_match += (f"doc='{strip_tags(title)}' " +
                  f"section='{sec_title}'>" + "\n")
    xml_match += write_markup(match[0]) + "\n"
    xml_match += f"</{tag}>"
    return xml_match


def query_document(ix, id_="GIB33", keywords=['"bero på"', 'förorsaka'],
                   format_='xml', year='', additional_terms=[],
                   field=None, context_size=2):
    """search for matches within one document
    Parameters:
               ix (FileIndex):
                  the index to search
               id_ (str):
                  the id of the document to search in the index
               keywords (list):
                  the (causality) query terms to match
               format (str):
                  the output format (either xml or html)
               year (str):
                  eventually the year to limit the search to
                  (for now just filename)
               additional_terms (list):
                  other terms to search in combination with the
                  keywords. To allow for multiple filters, this
                  is supposed to be a list of lists of words
               field (str):
                  the field to search; if the index schema has
                  a 'target' or 'body' field and no field is
                  prespecified, that fieldname is set.
               context_size (int):
                  the number of context sentences to the left and
                  right. Note that if the schema has designated fields
                  for left and right context, the context size
                  specified upon index creation is the upper limit.

    """
    sentf = MyOldSentenceFragmenter(maxchars=1000,
                                    context_size=context_size)
    formatter = MyFormatter(between="\n...")
    highlighter = MyHighlighter(fragmenter=sentf,
                                scorer=BasicFragmentScorer(),
                                formatter=formatter)
    if field is None:
        if 'target' in ix.schema._fields:
            field = 'target'
        elif 'body' in ix.schema._fields:
            field = 'body'
        else:
            raise Exception(f'Unknown schema without field specifier!' +
                            'If the index schema does not have a target or' +
                            'body field, specify another field to search!')
    qp = QueryParser(field, schema=ix.schema)
    text = Text('')
    text.from_html(f'documents/ft_{id_}.html')
    print(f'documents/ft_{id_}.html')
    matches_by_section = {}
    _query = ' OR '.join(keywords)
    if additional_terms:
        for term_list in additional_terms:
            print(term_list)
            terms = ' OR '.join(term_list)
            _query = f"({terms}) AND ({_query})"
    with ix.searcher() as searcher:
        query = qp.parse(
            f'((doc_title:{id_}) AND ({_query}))')
        res = searcher.search(query, terms=True, limit=None)
        if field == 'target':
            matches = []
            for m in res:
                # print(len(m), len(res), m['sec_title'])
                hits = highlighter.highlight_hit(
                        m, "target", top=len(m.results),
                        strict_phrase='"' in _query)
                # print(len(hits))
                for hit, start_pos in hits:
                    matches.append((hit, m, m['sent_nb']))
        else:
            matches = [(hit, m, start_pos) for m in res
                       for hit, start_pos in highlighter.highlight_hit(
                               m, field, top=len(m.results),
                               strict_phrase='"' in _query)]

        for i, match in enumerate(matches):
            if match[1]['sec_title'] not in matches_by_section:
                matches_by_section[match[1]['sec_title']] = {}
            matches_by_section[match[1]['sec_title']][match[-1]] = \
                format_match(match[:-1], i, format_=format_)

    if format_ == 'xml':
        parser = 'lxml'
    else:
        parser = 'html.parser'
    output = BeautifulSoup(parser=parser)
    head = output.new_tag(format_)
    output.append(head)
    if format_ == 'html':
        style = output.new_tag('style')
        style.append("""
        body {
	margin-top: 4%;
	margin-bottom: 8%;
	margin-right: 13%;
	margin-left: 13%;
        }
        """)
        head.append(style)
    elif format_ == 'xml':
        query_el = output.new_tag('query')
        query_el['term'] = query
        head.append(query_el)
        head = query_el
    for section in text.section_titles:
        sec = output.new_tag('section')
        section = re.sub(r'\s+', ' ', section)
        if format_ == 'html':
            heading = output.new_tag('h2')
            heading.append(section)
            sec.append(heading)
            head.append(sec)
            list_ = output.new_tag('ol')
            head.append(list_)
        else:
            heading = output.new_tag('title')
            heading.append(section)
            sec.append(heading)
        if section in matches_by_section:
            for key in sorted(matches_by_section[section]):
                el = BeautifulSoup(matches_by_section[section][key],
                                   parser=parser)
                if format_ == 'xml':
                    sec.append(el.match)
                else:
                    el.p.em.name = 'b'
                    element = output.new_tag('li')
                    element.append(el.p)
                    list_.append(element)
        if format_ == 'xml':
            head.append(sec)
    if additional_terms:
        if len(additional_terms) > 1:
            with open('document_search/' +
                      f'{id_}_{year}_incr_decr_custom' +
                      f' document_search.{format_}',
                      'w') as ofile:
                ofile.write(output.prettify())
        else:
            with open('document_search/' +
                      f'{id_}_{year}_incr_decr_document_search.{format_}',
                      'w') as ofile:
                ofile.write(output.prettify())
    else:
        with open('document_search/' +
                  f'{id_}_{year}_document_search.{format_}',
                  'w') as ofile:
            ofile.write(output.prettify())
    print(len(matches_by_section))
    return len(text.section_titles), len(matches)


if __name__ == "__main__":
    # analyzer = BasicTokenizer(do_lower_case=False) |\
    #    analysis.LowercaseFilter()
    analyzer = StandardAnalyzer(stoplist=[])
    schema = Schema(doc_title=TEXT(stored=True, analyzer=analyzer,
                                   lang='se'),
                    sec_title=TEXT(stored=True, analyzer=analyzer,
                                   lang='se'),
                    target=TEXT(stored=True, phrase=True,
                                analyzer=analyzer, lang='se'),
                    left_context=TEXT(stored=True, phrase=False,
                                      analyzer=analyzer, lang='se'),
                    right_context=TEXT(stored=True, phrase=False,
                                       analyzer=analyzer, lang='se'),
                    sent_nb=NUMERIC(stored=True, sortable=True))
    # ix = index.open_dir("yet_another_ix", indexname="big_index")
    # ix = index.open_dir("test_index", indexname="test")
    # ix = index.open_dir("bigger_index", indexname="big_index")
    # ix = index.open_dir("big_bt_index", indexname="bt_index")
    # query_list = [wf for term in expanded_dict.values() for wf in term]
    # print_to_file(query_list)
    # To create new index
    # create_index('bigger_index', schema=schema,
    # ixname='big_index', random_files=True)
    paths = ['documents/ft_GIB33.html', 'documents/ft_GKB3145d3.html',
             'documents/ft_GLB394.html', 'documents/ft_GOB345d1.html',
             'documents/ft_GRB350d3.html', 'documents/ft_GVB386.html',
             'documents/ft_GYB362.html', 'documents/ft_H1B314.html',
             'documents/ft_H4B391.html', 'documents/ft_H7B312.html']
    create_index('new_schema_document_ix', schema=schema, filenames=paths)
    query_list = [wf for term in expanded_dict.values() for wf in term]
    decr_terms = [wf for term in {**decr_dict, **incr_dict}.values()
                  for wf in term]
    ix = index.open_dir('new_schema_document_ix', indexname='test')
    years = ['1995', '1997', '1998', '2001', '2004', '2007', '2010',
             '2013', '2016', '2019']
    ids = ['GIB33', 'GKB3145d3', 'GLB394', 'GOB345d1', 'GRB350d3',
           'GVB386', 'GYB362', 'H1B314', 'H4B391', 'H7B312']
    match_counter = 0
    sec_counter = 0
    format_ = 'xml'
    for i, id_ in enumerate(ids):
        print(years[i], id_)
        secs, matches = query_document(ix, id_, keywords=query_list,
                                       year=years[i], format_=format_,
                                       field='target', context_size=2)

        secs, matches = query_document(ix, id_, keywords=query_list,
                                       year=years[i], format_=format_,
                                       additional_terms=[decr_terms],
                                       field='target')
        secs, matches = query_document(ix, id_, keywords=query_list,
                                       year=years[i], format_=format_,
                                       additional_terms=[decr_terms, topics[1]],
                                       field='target')
        
        print(secs, matches)
        sec_counter += secs
        match_counter += matches
    print(sec_counter, match_counter)


def compare_files(name):
    """helper to compare use of Highlighter and MyHighlighter"""
    with open(f'document_search/{name}') as ifile:
        new_f = BeautifulSoup(ifile.read())
    with open(f'old_doc_search/{name}') as ifile:
        old_f = BeautifulSoup(ifile.read())
    assert new_f == old_f


def find_causality_regions(filename):
    """situate sections with causality in a document"""
    with open(filename) as ifile:
        soup = BeautifulSoup(ifile.read())
    for i, section in enumerate(soup.find_all('section')):
        causal_sents = section.find_all('match')
        if causal_sents:
            print(f'{i}: {section.title.text.strip()} {len(causal_sents)}')
    print(len(soup.find_all('section')))

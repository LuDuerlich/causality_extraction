import re
import sys
sys.path.append("/Users/luidu652/Documents/causality_extraction/")
from data_extraction import Text
# sys.path.append("/Users/luidu652/Documents/causality_extraction/whoosh/src/")
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, NGRAM
from whoosh.qparser import QueryParser
from whoosh.highlight import *
from whoosh import index, highlight, query

schema = Schema(doc_title=TEXT(stored=True),
                sec_title=TEXT(stored=True),
                body=TEXT(stored=True, phrase=True))


def strip_tags(string):
    return re.sub("<h1>|</h1>", "", string)


class MyHighlighter(Highlighter):
    def __init__(self, fragmenter=None, scorer=None, formatter=None,
                  always_retokenize=False, order=FIRST,
                 analyzer=StandardAnalyzer(minsize=1, stoplist=[])):
        super().__init__(fragmenter, scorer, formatter,
                         always_retokenize, order)
        self.analyzer = analyzer

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
            fragments = self.fragmenter.fragment_matches(text, tokens)
        else:
            # Retokenize the text
            analyzer = results.searcher.schema[fieldname].analyzer
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
            fragments = self.fragmenter.fragment_tokens(text, tokens)
        fragments = top_fragments(fragments, top, self.scorer, self.order,
                                  minscore=minscore)
        output = self.formatter.format(fragments)
        return output


class ContextFragment(Fragment):

    def __init__(self, text, matches, startchar=0, endchar=-1,
                 sent_boundaries=(0, -1)):
        super().__init__(text, matches,  startchar, endchar)
        self.sent_boundaries = sent_boundaries


class MyFormatter(Formatter):
    """Joke's on you, it does not format"""

    def __init__(self, between="..."):
        """
        :param between: the text to add between fragments.
        """

        self.between = between

    def format_fragment(self, fragment, replace=False):
        """Returns a formatted version of the given text, using the "token"
        objects in the given :class:`Fragment`.

        :param fragment: a :class:`Fragment` object representing a list of
            matches in the text.
        :param replace: if True, the original text corresponding to each
            match will be replaced with the value of the token object's
            ``text`` attribute.
        """

        text = fragment.text
        index = fragment.sent_boundaries[0]
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
        return "".join(output)

    def format(self, fragments, replace=False):
        """Returns a formatted version of the given text, using a list of
        :class:`Fragment` objects.
        """

        formatted = list(filter(None, [self.format_fragment(f, replace=replace)
                                       for f in fragments]))
        return formatted

    def format_token(self, text, token, replace=False):
        return f"<b>{highlight.get_text(text, token, replace)}</b>"


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
        if last_tks:
            for i in match_sent_id:
                if match_sent_id:
                    # account for matches at the beginning of the document
                    start_s = min(i, len(sents))
                    current_startchar = sents[max(start_s - context, 0)][0]
                else:
                    startchar = sents[max(len(sents) - context, 0)][0]
                current_endchar = sents[min(i+context, len(sents) - 1)][-1]
                yield mksentfrag(text, last_tks.pop(0),
                                 startchar=sents[i-context][0],
                                 endchar=min(current_endchar, endchar),
                                 match_s_boundaries=sents[i])


def create_index(path="test_index/", ixname="test"):
    if os.path.exists(path) and os.listdir(path):
        if input(
                "index directory not empty. delete content?(Y/n) > "
        ).casefold() == "y":
            os.system(f"rm -r {path}*")
    ix = index.create_in(path, schema, indexname=ixname)

    writer = ix.writer()
    for k in ["H2B34", "H2B340", "H2B341", "H2B341", "H2B342",
              "H2B343", "H2B344", "H2B345", "H2B346", "H2B347",
              "H2B348", "H2B349", "H2B35"]:
        text = Text("")
        text.from_html(f"documents/s_{k}.html")
        for section in text:
            writer.add_document(doc_title=re.sub(r'\s+',
                                                 ' ',
                                                 text.title.text) + f" {k}",
                                sec_title=re.sub(r'\s+',
                                                 ' ', section.title),
                                body=re.sub(r'\s+', ' ',
                                            "\n".join(section.text)))

    writer.commit()


ix = index.open_dir("test_index", indexname="test")


def print_to_file():
    qp = QueryParser("body", schema=ix.schema)
    sentf = MySentenceFragmenter(maxchars=1000, context_size=3)
    formatter = MyFormatter(between="\n...")
    highlighter = Highlighter(fragmenter=sentf, scorer=BasicFragmentScorer(),
                              formatter=formatter)
    punct = re.compile(r"[!.?]")
    with ix.searcher() as s,\
         open("example_queries.xml", "w") as output:
        print("<xml>", file=output)
        for _query in ["orsak",
                       "\"bidrar till\"",
                       "\"på grund av\""]:
            print(f"<query term='{_query}'>", file=output)
            parsed_query = qp.parse(_query)
            r = s.search(parsed_query, terms=True, limit=None)
            for matched_s in r:
                matched_s.results.order = highlight.FIRST
                print(f"<match doc='{strip_tags(matched_s['doc_title'])}' " +
                      f"section='{matched_s['sec_title']}'>",
                      file=output)
                hits = highlighter.highlight_hit(
                    matched_s, "body",
                    top=len(matched_s.results),
                    strict_phrase=_query.startswith('"'))
                for i, hit in enumerate(hits):
                    print(f"<hit hit_nb='{i}'>", file=output)
                    print(hit, file=output)
                    print("</hit>", file=output)
                print("</match>", file=output)
            print("</query>", file=output)
        print("</xml>", file=output)

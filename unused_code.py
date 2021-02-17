from bs4 import BeautifulSoup
from html.entities import html5
from custom_whoosh import mksentfrag
import re
import unicodedata
from whoosh.highlight import *
from whoosh import analysis

htmlchars2ents = {char: entity for entity, char in html5.items()}
xmlchars2ents = {'"': 'quot;',
                 "'": 'apos;',
                 "<": 'lt;',
                 ">": 'gt;'}


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


def compare_files(file_a, file_b):
    """helper to compare use of Highlighter and CustomHighlighter"""
    # with open(f'document_search/{name}') as ifile:
    with open(file_a) as ifile:
        new_f = BeautifulSoup(ifile.read())
    # with open(f'old_doc_search/{name}') as ifile:
    with open(file_b) as ifile:
        old_f = BeautifulSoup(ifile.read())
    assert new_f.body == old_f.body

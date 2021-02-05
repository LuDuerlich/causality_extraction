from whoosh.highlight import *

def set_matched_filter_phrases(tokens, text, terms, phrases, analyzer):
    """
    Mark tokens to be highlighted as matched. Used for Strict Phrase highlighting.
    Phrase-aware: highlights only individual matches for individual query terms
                  and phrase matches for phrase terms.

    :param tokens:  Result tokens
    :param text:    Result text to scan for matched terms to highlight
    :param terms:   Individual query terms
    :param phrases: Query Phrases
    :return: yield each token with t.matched = True / False, indicating if the
             token should be highlighted
    """

    """
    Implementation note: Because the Token object follows a Singleton pattern,
    we can only read each one once. Because phrase matching requires rescanning,
    we require a rendered token list (the text parameter) instead. The function must 
    still yield Token objects at the end, so the text list is used as a way to build a list
    of Token indices (the matches set). The yield loop at the end uses this
    to properly set .matched on the yielded Token objects.
    """

    # Pass on the same tokenizer used during indexing of the field that is searched
    text = [token.text for token in analyzer(text)]
    matches = set()
    # Match phrases
    for phrase in phrases:
        i = 0
        n_phrase_words = len(phrase.words)
        slop = phrase.slop
        while i < len(text):
            if phrase.words[0] == text[i]:  # If first word matched
                if slop == 1:
                    # Simple substring match
                    if text[i + 1:i + n_phrase_words] == phrase.words[1:]:  # If rest of phrase matches
                        any(map(matches.add, range(i, i + n_phrase_words)))  # Collect matching indices
                        # Advance past match area.
                        # Choosing to ignore possible overlapping matches for efficiency due to low probability.
                        i += n_phrase_words
                    else:
                        i += 1
                else:
                    # Slop match
                    current_word_index = first_slop_match = last_slop_match = i
                    slop_matches = [first_slop_match]
                    for word in phrase.words[1:]:
                        try:
                            """
                            Find the *last* occurrence of word in the slop substring by reversing it and mapping the index back.
                            If multiple tokens match in the substring, picking the first one can overlook valid matches.
                            For example, phrase is: 'one two three'~2
                            Target substring is:    'one two two six three', which is a valid match.
                                                     [0] [1] [2] [3] [4]
                            
                            Looking for the first match will find [0], then [1] then fail since [3] is more than ~2 words away
                            Looking for the last match will find [0], then, given a choice between [1] or [2], will pick [2],
                            making [4] visible from there
                            """
                            text_sub = text[current_word_index + 1:current_word_index + 1 + slop][::-1]  # Substring to scan (reversed)
                            len_sub = len(text_sub)
                            next_word_index = len_sub - text_sub.index(word) - 1  # Map index back to unreversed list
                            last_slop_match = current_word_index + next_word_index + 1
                            slop_matches.append(last_slop_match)
                            current_word_index = last_slop_match
                        except ValueError:
                            # word not found in substring
                            i += 1
                            break
                    else:
                        i = last_slop_match
                        any(map(matches.add, slop_matches))  # Collect matching indices
            else:
                i += 1

    # Match individual terms
    for i, word in enumerate(text):
        for term in terms:
            if term.text == word:
                matches.add(i)
                break

    for i, t in enumerate(tokens):
        t.matched = i in matches 
        yield t


class CustomHighlighter(Highlighter):
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


class CustomFormatter(Formatter):
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


class CustomSentenceFragmenter(Fragmenter):
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

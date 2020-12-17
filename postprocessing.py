from bs4 import BeautifulSoup
import spacy


with open("hit_samplereconstructed.xml") as ifile:
    mark_up = BeautifulSoup(ifile.read())
hits = [match.text for match in mark_up.find_all('match')]


def test_language_models(m1=None, m2=None, m3=None):
    """compares two or more language models on the first
    match in hits. If no model names are given, default
    is to test English, Norwegian and Multilingual"""

    models = []
    docs = []
    if not m1 and not m2:
        m1 = "en_core_web_sm"
        m2 = "nb_core_news_sm"
        m3 = "xx_ent_wiki_sm"

    example = hits[0].strip()
    for m in [m1, m2, m3]:
        if m:
            model = spacy.load(m)
            sentencizer = model.create_pipe('sentencizer')
            model.add_pipe(sentencizer)
            models.append(model)
            docs.append(model(example))

    if len(docs) > 2:
        last = docs[2]
    else:
        last = None
    # To print all sentences
    # compare_segmentation(en, nb, ml, True)
    compare_segmentation(docs[0], docs[1], last)


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
        for b_s in b_gen:
            if c and c_gen:
                for c_s in c_gen:
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

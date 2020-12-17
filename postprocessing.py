from bs4 import BeautifulSoup
import spacy


with open("hit_samplereconstructed.xml") as ifile:
    mark_up = BeautifulSoup(ifile.read())
hits = [match.text for match in mark_up.find_all('match')]


def test_language_models():
    """compares English, Norwegian and Multilingual Models
    on the first match in hits"""
    en_model = spacy.load("en_core_web_sm")
    nb_model = spacy.load("nb_core_news_sm")
    ml_model = spacy.load("xx_ent_wiki_sm")
    for model in [en_model, nb_model, ml_model]:
        sentencizer = model.create_pipe('sentencizer')
        model.add_pipe(sentencizer)
    example = hits[0].strip()
    en = en_model(example)
    nb = nb_model(example)
    ml = ml_model(example)
    # To print all sentences
    # compare_segmentation(en, nb, ml, True)
    compare_segmentation(en, nb, ml)


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
            if c_gen:
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

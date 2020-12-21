import requests
import pickle
import re
import os
search_terms = ['"bero på"',
                '"bidra till"',
                '"leda till"',
                '"på grund av"',
                '"till följd av"',
                "följd",
                '"är ett resultat av"',
                "resultat",
                "resultera",
                "därför",
                "eftersom",
                "förklara",
                "förorsaka",
                "orsak",
                "orsaka",
                "påverka",
                "effekt",
                "medföra",
                "framkalla",
                "vålla",
                "rendera"]

annotated_search_terms = [('"bero på"', 0, "vb"),
                ('"bidra till"', 0, "vb"),
                ('"leda till"', 0, "vb"),
                ('"på grund av"', 1, "nn"),
                ('"till följd av"', 1, "nn"),
                ("följd", 0, "nn"),
                ('"vara ett resultat av"', 0, "vb"),
                ("resultat", 0, "nn"),
                ("resultera", 0, "vb"),
                ("därför", None, None),
                ("eftersom", None, None),
                ("förklara", 0, "vb"),
                ("förorsaka", 0, "vb"),
                ("orsak", 0, "nn"),
                ("orsaka", 0, "vb"),
                ("påverka", 0, "vb"),
                ("effekt", 0, "nn"),
                ("medföra",0, "vb"),
                ("framkalla", 0, "vb"),
                ("vålla", 0, "vb"),
                ("rendera", 0, "vb")]


def expand():
    dictionary = {"vara": ["var", "är", "varit", "vore", "vara"]}
    terms = {}
    for term, i, pos in annotated_search_terms:
        if term not in terms:
            terms[term] = set()
        if pos:
            words = term.strip("\"").split()
            if words[i] in dictionary:
                if len(words) > 1:
                    for form in dictionary[words[i]]:
                        if i + 1 == len(words):
                            terms[term].add(f'"{" ".join(words[:i] + [form])}"')
                        else:
                            terms[term].add(f'"{" ".join(words[:i] + [form] + words[min(i+1, len(words)-1):])}"')
                else:
                    terms[term] = terms[term].union(dictionary[words[i]])
            else:
                forms = requests.get(f"https://skrutten.csc.kth.se/granskaapi/inflect.php?word={words[i]}&tag={pos}")
                if forms:
                    forms = re.sub("((&lt;)+[a-z.*]*(&gt;)+|<br>|\d+)", "", forms.text.strip())
                    forms = [el.split()[0].strip() for el in forms.split("\n")[1:]
                             if el.split() and not el.startswith("all forms")]
                    dictionary[words[i]] = forms
                    for form in set(forms):
                        if len(words) > 1:
                            terms[term].add(f'"{" ".join(words[:i] + [form] + words[min(i+1, len(words)-1):])}"')
                        else:
                            terms[term].add(form)
        else:
            terms[term].add(term)
    return terms

if os.path.isfile("expanded_dict.pickle"):
    with open("expanded_dict.pickle", "rb") as ifile:
        expanded_dict = pickle.load(ifile)
else:
    expanded_dict = expand()


expanded_list = ['"berodde på"', '"berotts på"', '"beros på"', '"beroddes på"', '"berott på"', '"berodd på"', '"beror på"', '"berodda på"', '"berodds på"', '"bero på"', '"bidras till"', '"bidragits till"', '"bidragna till"', '"bidragit till"', '"bidrogs till"', '"bidragens till"', '"bidraget till"', '"bidragne till"', '"bidragande till"', '"bidrar till"', '"bidragen till"', '"bidrog till"', '"bidra till"', '"frågans om hur"', '"fråge om hur"', '"frågors om hur"', '"frågornas om hur"', '"frågor om hur"', '"frågan om hur"', '"fråga om hur"', '"frågorna om hur"', '"gavs ökad effektivitet"', '"ge ökad effektivitet"', '"given ökad effektivitet"', '"givne ökad effektivitet"', '"givet ökad effektivitet"', '"gav ökad effektivitet"', '"givens ökad effektivitet"', '"givande ökad effektivitet"', '"givna ökad effektivitet"', '"getts ökad effektivitet"', '"ges ökad effektivitet"', '"ger ökad effektivitet"', '"gett ökad effektivitet"', '"ges positiva effekter i form av"', '"ge positiva effekter i form av"', '"givna positiva effekter i form av"', '"ger positiva effekter i form av"', '"givens positiva effekter i form av"', '"givet positiva effekter i form av"', '"ge positiva effekter i form av"', '"given positiva effekter i form av"', '"ges positiva effekter i form av"', '"gavs positiva effekter i form av"', '"gett positiva effekter i form av"', '"givne positiva effekter i form av"', '"gav positiva effekter i form av"', '"getts positiva effekter i form av"', '"givande positiva effekter i form av"', '"ledads till"', '"letts till"', '"ledats till"', '"ledades till"', '"ledad till"', '"ledda till"', '"leda till"', '"ledande till"', '"ledds till"', '"ledd till"', '"leder till"', '"lett till"', '"ledat till"', '"led till"', '"ledade till"', '"ledes till"', '"ledas till"', '"ledar till"', '"ledde till"', '"leddes till"', '"på grund av"', '"som en lösning på"', '"syftet med"', '"till följd av"', 'bidras', 'bidra', 'bidragna', 'bidrar', 'bidragens', 'bidraget', 'bidra', 'bidragen', 'bidras', 'bidrogs', 'bidragit', 'bidragne', 'bidrog', 'bidragits', 'bidragande', 'därför', 'eftersom', 'förklarades', 'förklarads', 'förklarar', 'förklarat', 'förklarad', 'förklarats', 'förklara', 'förklarade', 'förklarande', 'förklaras', 'förorsakads', 'förorsakar', 'förorsakande', 'förorsakas', 'förorsakats', 'förorsaka', 'förorsakat', 'förorsakad', 'förorsakades', 'förorsakade', 'orsak', 'orsakens', 'orsakers', 'orsakernas', 'orsaker', 'orsaken', 'orsakerna', 'orsakas', 'orsakats', 'orsakar', 'orsakande', 'orsakad', 'orsaka', 'orsakades', 'orsakat', 'orsakads', 'orsakade', 'påverkat', 'påverkades', 'påverkads', 'påverkade', 'påverkats', 'påverkar', 'påverka', 'påverkas', 'påverkad', 'påverkande', 'resultatindikatorer', 'resultatindikatorens', 'resultatindikatorerna', 'resultatindikator', 'resultatindikators', 'resultatindikatorers', 'resultatindikatoren', 'resultatindikatorernas']

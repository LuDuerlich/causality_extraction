import requests
import pickle
import re
import os
search_terms = ['"bero på"',
                '"bidra till"',
                '"leda till"',
                '"på grund av"',
                '"till följd av"',
                '"är ett resultat av"',
                "resultera",
                "förorsaka",
                "orsaka",
                "påverka",
                "effekt",
                "medföra",
                "framkalla",
                "vålla"]

annotated_search_terms = [('"bero på"', 0, "vb"),
                ('"bidra till"', 0, "vb"),
                ('"leda till"', 0, "vb"),
                ('"på grund av"', 1, "nn"),
                ('"till följd av"', 1, "nn"),
                ('"vara ett resultat av"', 0, "vb"),
                ("resultera", 0, "vb"),
                ("förorsaka", 0, "vb"),
                ("orsaka", 0, "vb"),
                ("påverka", 0, "vb"),
                ("medföra",0, "vb"),
                ("framkalla", 0, "vb"),
                ("vålla", 0, "vb")]


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

expanded_dict = {'"bero på"': {'"berodd på"', '"beror på"', '"berodds på"', '"beros på"', '"berodda på"',
                               '"berott på"', '"berotts på"', '"beroddes på"', '"berodde på"', '"bero på"'},
                 '"bidra till"': {'"bidragna till"', '"bidragits till"', '"bidrog till"', '"bidra till"',
                                  '"bidragande till"', '"bidragit till"', '"bidrogs till"', '"bidras till"',
                                  '"bidrar till"', '"bidragne till"'},
                 '"leda till"': {'"ledda till"', '"led till"', '"ledar till"', '"leder till"', '"lett till"',
                                 '"ledades till"', '"ledds till"', '"ledes till"', '"leda till"', '"ledde till"',
                                 '"ledad till"', '"ledats till"', '"ledd till"', '"ledas till"', '"ledat till"',
                                 '"ledads till"', '"ledande till"', '"letts till"', '"ledade till"', '"leddes till"'},
                 '"på grund av"': {'"på grunders av"', '"på grundet av"', '"på grunder av"', '"på grunds av"',
                                   '"på grundets av"', '"på grunden av"', '"på grund av"', '"på grundens av"',
                                   '"på grunderna av"', '"på grundernas av"'},
                 '"till följd av"': {'"till följds av"', '"till följden av"', '"till följd av"', '"till följdens av"',
                                     '"till följder av"', '"till följdernas av"', '"till följderna av"', '"till följders av"'},
                 '"vara ett resultat av"': {'"vara ett resultat av"', '"är ett resultat av"', '"var ett resultat av"',
                                            '"vore ett resultat av"', '"varit ett resultat av"'},
                 'resultera': {'resulterats', 'resulterat', 'resulterar', 'resulterads', 'resulterade', 'resulterande',
                               'resulterad', 'resulteras', 'resultera', 'resulterades'},
                 'förorsaka': {'förorsakads', 'förorsakas', 'förorsakar', 'förorsakande', 'förorsakades', 'förorsakade',
                               'förorsakats', 'förorsakat', 'förorsakad', 'förorsaka'},
                 'orsaka': {'orsakades', 'orsakad', 'orsaka', 'orsakads', 'orsakande', 'orsakade', 'orsakat', 'orsakats',
                            'orsakar', 'orsakas'},
                 'påverka': {'påverkats', 'påverkads', 'påverkas', 'påverkade', 'påverkar', 'påverkad', 'påverkades',
                             'påverkande', 'påverka', 'påverkat'},
                 'medföra': {'medförts', 'medförd', 'medföra', 'medföras', 'medförda', 'medföres', 'medförds', 'medförde',
                             'medför', 'medfördes', 'medfört'},
                 'framkalla': {'framkallas', 'framkallade', 'framkallads', 'framkallades', 'framkallats', 'framkallat',
                               'framkallad', 'framkalla', 'framkallar', 'framkallande'},
                 'vålla': {'vållats', 'vållad', 'vållat', 'vållades', 'vållas', 'vållande', 'vållar', 'vållade',
                           'vålla', 'vållads'}}

import pytest
from dependency_paths import *


def test_paths():
    with open('pos_filtered_out_target_only_combined_topics.html') as ifile:
        samples = BeautifulSoup(ifile.read(), parser='html.parser')('p')
    gold_out = [(2, 'arbetslöshet', 'medfört'),
                (1, 'arbetsmarknad', 'medfört'),
                (2, 'arbetslöses', 'medföra'),
                (1, 'arbetslöshet', 'medföra'),
                (2, 'arbetslöshet', 'på'),
                (1, 'arbetslöshet', 'på'),
                (3, 'arbetslöshet', 'medför'),
                (5, 'arbetslöshet', 'medföra'),
                (3, 'arbetslös', 'på'),
                (3, 'arbetslös', 'på')]
    for i in range(10):
        caus, top = find_tokens(samples[i])
        shortest = None
        tokens = []
        for c in caus:
            if shortest == 1:
                break
            for t in top:
                path = _path(c, t)
                if path and (shortest is None or shortest > path):
                    shortest = path
                    tokens = [t, c]
                    if path == 1:
                        break
        assert gold_out[i] == (shortest, tokens[0].text, tokens[1].text)

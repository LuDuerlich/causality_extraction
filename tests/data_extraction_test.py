import re, sys
sys.path.append("/Users/luidu652/Documents/causality_extraction/")
from data_extraction import *

def test_hierarchy_heuristics():
    def make_test_sections(contents):
            sections = []
            for title, content in contents:
                new_section = Section(title)
                if content:
                    new_section.append(content)
                sections.append(new_section)
            assert len(sections) == len(contents),\
                'There should be as many sections as there are elements in content!'
            return sections

    # test title merging
    contents = [('1 Avsnitt', ''), ('utan nummer', 'text'),
                ('1.1 Avsnitt', 'text'), ('2 Avsnitt', 'text'),
                ('Bilaga 1', 'text')]
    sections = make_test_sections(contents)
    hierarchy_heuristics(sections)
    assert sections[0].title.endswith('utan nummer'),\
        'The first section should have a merged header'
    assert len(sections) == 4

    # test multi-column title merging
    contents = [('1 Avsnitt', ''), ('har en lång rubrik', ''),
                ('över tre råder', 'text'), ('2 Avsnitt', 'text')]
    sections = make_test_sections(contents)
    
    hierarchy_heuristics(sections)
    assert len(sections) == 2

    # test super- / sub-title merging
    contents = [('1 Avsnitt', 'text'), ('rubrik 1', 'text'),
                ('rubrik 2', 'text'), ('2 Avsnitt', 'text')]
    sections = make_test_sections(contents)

    hierarchy_heuristics(sections)
    assert sections[1].title.startswith(contents[0][0])
    assert sections[-1].title.startswith(contents[0][0]) == False

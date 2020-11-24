import json
import requests
import tqdm
import re
import os
import pickle
import bs4

# name format is sou_<year>_<id>.pdf or column 1 for html
# (check column 7 and 16 for sou with multiple parts at
# https://data.riksdagen.se/dokument/
# do we care about relations between SOUs in multiple parts?
# are the summaries useful?
# look for json and extract directly through document_url_text
# / document_url_html
# verify if we get all the documents
# hyphenation
# some have English summary?
# to keep or to discard headings / titles?
# summary separate from full text?
# remove date and place + SOU ID?
# work with two-column format -> one case of suggested change of wording
# i.e. large part of the texts are really similar
# to fix: GPB334, 'H3B333', GVB315'
# broken files https://data.riksdagen.se/dokument/GSB321.html
# ignore file : GQB399d2 - GQB399d7 (just tables / figures)
# bf in paragraphs not titles


class Section:
    """"""
    def __init__(self):
        self.title = ""
        self.text = []

    def __repr__(self):
        return f"Section object with title: '{self.title}',\
        length: {len(self)}"

    def __iter__(self):
        for text in self.text:
            yield text

    def __getitem__(self, i):
        return self.text[i]

    def __len__(self):
        return len(self.text)

    def __setitem__(self, i, v):
        self.text[i] = v

    def __eq__(self, other):
        if type(other) == Section:
            return self.title == other.title and\
                self.text == other.text
        return False

    def append(self, el):
        self.text.append(el)

    def extend(self, el):
        self.text.extend(el)


class Text(object):
    """simple text representation"""
    def __init__(self, title):
        self.title = title
        self.content = []

    def __repr__(self):
        return f"Text object with title: '{self.title}', length: {len(self)}"

    def __iter__(self):
        for text in self.content:
            yield text

    def __getitem__(self, i):
        return self.content[i]

    def __len__(self):
        return len(self.content)

    def __setitem__(self, i, v):
        self.content[i] = v

    def __bool__(self):
        return bool(self.title)

    def __eq__(self, other):
        if type(other) == Text:
            return self.title == other.title and self.content == other.content
        return False

    def append(self, el):
        self.content.append(el)


def extract_from_json():
    """extract document date, text and html locations from respective
    json file"""

    page = json.load(open("../data.riksdagen.se.json"))
    nb_pages = int(page["dokumentlista"]["@sidor"])

    documents = {}

    for i in tqdm.tqdm(range(nb_pages), desc="Extracting...", ncols=100):
        for document in page["dokumentlista"]["dokument"]:
            doc_id = document["id"]
            documents[doc_id] = {"titel": document["sokdata"]["titel"] + "/"
                                 + document["sokdata"]["undertitel"],
                                 "datum": document["datum"],
                                 "content": {match: document[match] for
                                             match in document.keys() if
                                             "dokument_url_" in match}
                                 }
            # if "filbilaga" in document and "fil" in document["filbilaga"]:
            #    documents[doc_id]["content"]["pdf"] =
            #    document["filbilaga"]["fil"]["url"]

        if "@nasta_sida" in page["dokumentlista"]:
            # apparently, the next page is not saved as json anymore
            page = json.loads(requests.get(
                page["dokumentlista"]["@nasta_sida"].replace("=htm&",
                                                             "=json&")).text)

    return documents


def _pages(tag):
    """select only divs that represent a full page"""
    return tag.name == "div" and tag["id"].startswith("page_")


def _is_pagenum(tag):
    """or SOU id"""
    match = re.match("(SOU \d+:)?\d+", tag.text)
    if match:
        return match.span()[1] - match.span()[0] == len(tag.text)
    return False


def _freestanding_p(tag):
    """verify that text is actual content and not part of header or
    table of contents"""
    # replace (tag.parent.name != "td") by not tag.a to keep
    # two-column instances (e.g. H8B336)
    return tag.name in "p" and not tag.a and not _is_pagenum(tag)


def _table_of_contents(tag):
    """filter out links that are not for navigation in the html or
    other paragraph elements"""
    return (tag.name == "a" and tag.has_attr("href") and
            tag["href"].startswith("#page"))\
            or ((tag.name in ["p", "span"] and
                 tag.text.strip().endswith(".."))
                or (tag.name == "p"
                    and type(tag.next_sibling) == bs4.element.Tag
                    and tag.next_sibling.text.endswith("..")))


def _tables_outside_toc(tag):
    """useful for finding headers and real tables"""
    return tag.name == "table" and not tag.find("a")


def _appendix(tag):
    """find appendices in table of content"""
    return tag.name == "td" and re.match("bilaga|bilagor",
                                         tag.text.casefold().strip())


def insert_whitespace(children):
    """correct for elements with multiple children with missing
    white space in between"""
    text = ""
    for i, child in enumerate(children[:-1]):
        if (not
            (type(child) == bs4.element.NavigableString and
             child.endswith(" ")) or
            (type(child) != bs4.element.NavigableString and
             child.text.endswith(" "))) and (not
            (type(children[i+1]) == bs4.element.NavigableString
              and children[i+1].startswith(" "))
         or (type(children[i+1]) != bs4.element.NavigableString
             and children[i+1].text.startswith(" "))):
        
             if type(child) == bs4.element.NavigableString:
                 text += f"{child} "
             elif type(child) == bs4.element.Tag:
                 text += f"{child.text} "
        else:
            if type(child) == bs4.element.NavigableString:
                text += f"{child}"
            elif type(child) == bs4.element.Tag:
                 text += f"{child.text}"

    if type(children[-1]) == bs4.element.NavigableString:
        text += f"{children[-1]} "
    elif type(children[-1]) == bs4.element.Tag:
        text += f"{children[-1].text} "
    return text


def extract_from_html(text, key=None):
    """extract text as list of sentences and return full text and summary"""
    # text segments to distinguish
    is_summary, is_english_summary, is_simple_summary, is_appendix =\
        False, False, False, False
    summary_section_nb, end_of_summary = 0, ""
    has_summary, has_english_summary, has_simple_summary, has_appendix =\
        False, True, True, False
    summary_title, en_summary_title, s_summary_title = "Sammanfattning",\
        "Summary", "Lättläst Sammanfattning"
    section_titles, seen_sections = [], []

    # return variables
    summary, english_summary, simple_summary, full_text = None, None,\
        None, None

    # text segments to exclude
    first_section = ""
    is_order_info, is_table_of_c = False, False

    soup = bs4.BeautifulSoup(
        requests.get(f'http:{text["content"]["dokument_url_html"]}').text)
    if soup.find("error"):
        global errors
        errors.add(key)
        return "", "", "", ""
    # there does not seem to be any difference between headings
    # and text in the markup. Hence, paragraphs can be a short title
    # or multiple complete sentences
    # trying divs over paragraphs
    # paragraphs = soup.find_all(_freestanding_p)
    pages = soup.find_all(_pages)
    
    links = soup.find_all(_table_of_contents)
    print(len(links))
    tables = soup.find_all(_tables_outside_toc)
    table_types = set(tables)
    copied_tables = "".join([str(t) for t in
                             table_types if tables.count(t) > 1])

    title_classes = re.findall("\.(ft\d+)\{font: +(bold|[2-9][0-9]px) [^}]+\}",
                               str(soup.style))
    title_classes = [el[0] if type(el) == tuple else
                     el for el in title_classes]

    def _determine_structure(text_part, element, text):
        """sort out whether an element is part of title or text body
        and add them to the respective Text object accordingly
        """
        nonlocal section_titles, seen_sections
        print("here", section_titles[:2], text, file=out)
        print(seen_sections, file=out)
        font_classes = re.findall('class="[^"]*(ft\d+)[^"]*"', str(element))
        # if we didn't want to treat unnumbered titles as such
        # if (font_class and font_class[1] in title_classes) or\
        # (font_class and font_class in bold and re.match("\d[.\d+]*",
        # element.text)):

        if text_part.content and text == text_part[-1].title\
           or element.text.strip() in seen_sections:
            pass
        elif not _is_pagenum(element):
            if (font_classes and
                len([el for el in font_classes if el in title_classes])
                 == len(font_classes)):
                print(f"_determine_structure '{font_classes}'", file=out)
                # is title
                new_section = Section()
                new_section.title = text.strip()
                text_part.append(new_section)
                if (re.sub("^\d+ *", "", new_section.title.casefold())
                    == section_titles[0].casefold()):
                    seen_sections.append(section_titles284.pop(0))
                else:
                    seen_sections.append(new_section.title)
                print(f"new section *{text.strip()}*", file=out)
            elif section_titles and\
                 text.strip().casefold().endswith(section_titles[0].casefold()):
                # is title
                new_section = Section()
                new_section.title = text.strip()
                text_part.append(new_section)
                print(f"new section 2 *{text.strip()}*", file=out)
                seen_sections.append(section_titles.pop(0))

            # elif re.match(r"[\d.]+ *[^.!?;:]+$", string.strip()):
            #     new_section = Section()
            #     new_section.title = text
            #     text_part.append(new_section)
            #     print("new section 3",file=out)
            elif text != text_part.title:
                # is text body
                if text_part.content:
                    text_part[-1].append(text.strip())

                else:
                    # text_part is empty
                    new_section = Section()
                    new_section.append(text.strip())
                    text_part.append(new_section)

    # appendix = soup.find_all(_appendix)[0].text.strip()
    # if appendix:
    #     has_appendix = True

    # find the name of the section following the summary
    # and determine whether there are simplified or English summaries
    # from the table of contents
    for i, hlink in enumerate(links):
        section_title = ""
        print(i, hlink, summary_title, en_summary_title,
              end_of_summary, file=out)
        if section_titles:
            sec_title = re.match(r"[^.]+ *\.\.+$", hlink.text)
            print("hello", hlink.text, sec_title, file=out)
            if sec_title:
                sec_title = sec_title[0].strip(". ")
                if sec_title != section_titles[-1]:
                    section_titles.append(sec_title)
        if not first_section and re.match(r"[A-Za-z]+", hlink.text)\
           and hlink.parent.name == "td":
            print(2, file=out)
            first_section = hlink.text.strip(". ").casefold()
        if re.match(r"\bsammanfattning\b", hlink.text.casefold().strip())\
           and not has_summary:
            print(3, file=out)
            summary_title = hlink.text.strip(". ")
            has_summary = True
            summary_section_nb = -1
            # print("Update summary_section_nb", summary_section_nb)
            if hlink.previous_sibling and\
               hasattr(hlink.previous_sibling, "text"):
                # print(f"text: '{links[i-1].text}'")
                match = re.match("[0-9.]+", hlink.previous_sibling.text)
                if match:
                    summary_section_nb = int(match.group().split(".")[0])
                    # print("ssn", summary_section_nb)

        if ("sammanfattning" in hlink.text.casefold() or
           ((has_english_summary or has_simple_summary) and
             end_of_summary)) or not end_of_summary:
            print(4, file=out)
            # print("this", hlink, summary_section_nb)
            i += 1
            # print("new loop", len(links), i)
            while i < len(links) and\
                  not re.search(r"[A-Za-z]+ ?\.+", links[i].text):
                # or links[i+1].text.strip().endswith("Sammanfattning"):
                # print(links[i], i, len(links))
                i += 1
            if i >= len(links):
                print("stop", i, len(links), summary_section_nb, file=out)
                break
            # print("stop", links[i], i, len(links), summary_section_nb)
            if links[i].text.strip(". ").casefold() not in\
               [summary_title.casefold(), s_summary_title.casefold(),
                en_summary_title.casefold()]:
                if summary_section_nb < 0 and not section_titles:
                    print("this one", hlink.text, links[i].text, i, file=out)
                    end_of_summary = links[i].text.strip(". ")
                    if not first_section:
                        first_section = end_of_summary
                    section_title = end_of_summary
                    # print(f'1 "{end_of_summary}", "{summary_section_nb}"')
                if (summary_section_nb and\
                    links[i-1].text.strip(". ").startswith(str(summary_section_nb+1))
                    and not section_titles):
                    end_of_summary = links[i].text.strip(". ")
                    print("that one", summary_section_nb,
                          links[i-1].text.strip(". "), summary_section_nb+1,
                          file=out)
                    print(f'2 "{end_of_summary}" "{links[i-1]}"\
                    "{summary_section_nb}"')
                    section_title = end_of_summary
                    if not first_section:
                        first_section = end_of_summary

            else:
                section_title = links[i].text.strip(". ")
            if re.match(r"\bsummary\b", section_title.casefold()):
                has_english_summary = True
                en_summary_title = section_title
                end_of_summary = ""
            elif "lättläst" in section_title.casefold() and "sammanfattning"\
                 in section_title.casefold():
                s_summary_title = section_title.strip(". ")
                has_simple_summary = True
                end_of_summary = ""

        if end_of_summary and not section_titles:

            section_titles.append(end_of_summary)
        print(6, file=out)

    print("section_titles", len(section_titles))
    print(section_titles, file=out)
    print(has_english_summary, has_simple_summary, f"'{end_of_summary}'",
          f"'{summary_title}'", f"'{en_summary_title}'",
          f"'{s_summary_title}'")

    def _is_end_of_summary(element):
        """helper function to separate summary
        from (simplified/English version or full report"""
        nonlocal seen_sections
        if end_of_summary and element.text.endswith(end_of_summary):
            if section_titles:
                seen_sections.append(section_titles.pop(0))
            return True
        elif has_english_summary and\
            re.match(r"\b" + en_summary_title.casefold()
                     + r"\b", element.text.casefold()):
            nonlocal is_english_summary, english_summary
            is_english_summary = True
            print("English summary", file=out)
            if not english_summary:
                english_summary = Text(element.text.strip(". "))
                seen_sections.append(english_summary.title)
        elif has_simple_summary and re.match(r"\b" + s_summary_title.casefold()
                                             + r"\b", element.text.casefold()):
            nonlocal is_simple_summary, simple_summary
            print("Simple summary", file=out)
            is_simple_summary = True
            simple_summary = Text(element.text.strip(". "))
            seen_sections.append(simple_summary.title)
        return False

    def _two_columns(table):
        """detect and merge two column format"""
        columns = {}
        print("TWO COLS", file=out)
        trs = table.find_all("tr")
        if len(trs) < 2:
            return
        for tr in trs:
            for i, td in enumerate(tr.find_all("td")):
                if i in columns:
                    columns[i] += " " + td.text
                else:
                    columns[i] = td.text
        return [columns[k] for k in sorted(columns.keys())]

    def _split_text_on_page(paragraphs):
        nonlocal is_summary, is_english_summary, is_simple_summary,\
            summary, full_text, english_summary, simple_summary
        for element in paragraphs:
            if type(element) == bs4.element.Tag:
                print(f"element: {element}", file=out)
                text = element.text

                # correct for p-elements with multiple children,
                # e.g. consecutive spans with missing white space
                children = list(element.children)
                if len(children) > 1:
                    text = insert_whitespace(children)
                if is_summary:
                    if is_english_summary:
                        _determine_structure(english_summary, element, text)
                    elif is_simple_summary:
                        _determine_structure(simple_summary, element, text)
                    else:
                        _determine_structure(summary, element, text)
                else:  # if not is_table_of_c and not is_order_info:
                    if full_text:
                        _determine_structure(full_text, element, text)
                    else:
                        full_text = Text(text)
                # if not is_table_of_c and p.parent.name == "td":
                #     global also_tables
                #     also_tables.add(key)

    print("TEXT EXTRACTION", file=out)
    # text extraction
    for page in pages:
        print(page, file=out)
        for i, p in enumerate(page.children):
            print("NEW PAGE SECTION", "*"*5, file=out)
            print(p, file=out)
            paragraphs = False
            if hasattr(p, "find_all"):
                if p.name == "p":
                    paragraphs = [p]
                else:
                    paragraphs = p.find_all(_freestanding_p)

            print("Paragraphs after:", paragraphs, file=out)
            if not paragraphs or not p:
                print("skip", file=out)
                continue
            print(i, f"*{i}**{p}*", is_summary, is_table_of_c, is_order_info,
                  is_english_summary, is_simple_summary,
                  first_section, end_of_summary, file=out)
            if p.text.rstrip().endswith("....") or\
               p.text.startswith("SOU och Ds kan köpas från "):
                print(f"ignoring *{p}*", file=out)
                break
            if p.name == "table":
                if not p.text.casefold().startswith("tablå")\
                   or "tablå" in p.previous_sibling.text.casefold():
                    print("potentially two columns", p.text, file=out)
                    text = _two_columns(p)
                    if text:
                        print(text)
                        if is_summary:
                            if is_english_summary:
                                english_summary[-1].extend(text)
                            elif is_simple_summary:
                                simple_summary[-1].extend(text)
                            else:
                                summary[-1].extend(text)
                        else:
                            full_text[-1].extend(text)
                            continue
                else:
                    continue
            elif p.has_attr("class") and\
                 f'{p["class"][-1]}">{p.text}</p></td>'\
                 in copied_tables and p["class"][-1] != "ft0":
                continue
                    # is header / footer

            elif (not is_table_of_c or "...." in page.text)\
                 and ("innehåll" in paragraphs[0].text.casefold() or
                      (len(paragraphs) > 1 and "innehåll" in
                       paragraphs[1].text.casefold())):
                is_table_of_c = True
                print(f"ignoring toc *{p}*", file=out)
                break
            elif re.match(r"\b" + re.sub(r"([[\]()])", r"\\\1",
                                         summary_title.casefold())
                          + r"\b", paragraphs[0].text.casefold())\
                          and not summary:
                is_summary = True
                summary = Text(paragraphs[0].text.strip(". "))
                seen_sections.append(summary.title)
                paragraphs.pop(0)
                if not paragraphs: continue
                is_table_of_c = False
                # continue

            elif is_summary and _is_end_of_summary(paragraphs[0]):
                is_summary = False

            if paragraphs[0].text.casefold().strip().endswith(first_section):
                is_table_of_c = False
            _split_text_on_page(paragraphs)
            # text = p.text

            # # correct for p-elements with multiple children,
            # # e.g. consecutive spans with missing white space
            # children = list(p.children)
            # if len(children) > 1:
            #     text = insert_whitespace(children)
            # if is_summary:
            #     if is_english_summary:
            #         _determine_structure(english_summary, p, text)
            #     elif is_simple_summary:
            #         _determine_structure(simple_summary, p, text)
            #     else:
            #         _determine_structure(summary, p, text)
            # elif not is_table_of_c and not is_order_info:
            #     if full_text:
            #         _determine_structure(full_text, p, text)
            #     else:
            #         full_text = Text(text)
            # # if not is_table_of_c and p.parent.name == "td":
            # #     global also_tables
            # #     also_tables.add(key)
    return full_text, summary, english_summary, simple_summary


def print_to_files(id_, text, summary, english_summary, simple_summary):
    if not os.path.exists("documents"):
        os.system("mkdir documents")
    for name, text_part in [("ft", text), ("s", summary),
                            ("ENs", english_summary), ("SEs", simple_summary)]:
        if text_part:
            with open(f"documents/{name}_{id_}.html", "w") as ofile:
                print(f"<h1>{text_part.title}</h1>", file=ofile)
                for section in text_part.content:
                    print(f"<h2>{section.title}</h2>", file=ofile)
                    for line in section.text:
                        print(f"<p>{line}</p>", file=ofile)


# docs = extract_from_json()
# with open("documents.pickle","rb") as ifile:
# docs = pickle.load(ifile)
# ft, s = extract_from_html(docs["H8B36"])
# len(ft) # 2028
# for key in list(docs.keys())[:10]:
# ft, s, es, ss =  extract_from_html(docs[key])
# print_to_files(key, ft, s, es, ss)
# verify that the full text begins where we want it to start
# (often Författningsförslag / Inledning)
# Text object with title: 'Sammanfattning', length: 15
# Text object with title: 'Ett nationellt sammanhållet
# system för kunskapsbaserad vård', length: 365

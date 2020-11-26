import json
import requests
import tqdm
import re
import os
import pickle
import bs4
import logging

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
    def __init__(self, title=""):
        self.title = title
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

    def __contains__(self, el):
        return el in [sec.title for sec in self.content]

    def append(self, el):
        self.content.append(el)

    def remove(self, el):
        self.content.remove(el)

    def index(self, el, reverse=False):
        if reverse:
            return [sec.title for sec in self.content][::-1].index(el) - 1
        return [sec.title for sec in self.content].index(el)

    @property
    def section_titles(self):
        return [sec.title for sec in self.content]


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
    return (tag.name == "div" and tag.has_attr("id") and
            tag["id"].startswith("page_"))


def _is_pagenum(tag):
    """or SOU id"""
    match = re.match(r"(SOU \d+:)?\d+", tag.text)
    if match:
        return match.span()[1] - match.span()[0] == len(tag.text)
    return False


def _freestanding_p(tag):
    """verify that text is actual content and not part of header or
    table of contents"""
    # replace (tag.parent.name != "td") by not tag.a to keep
    # two-column instances (e.g. H8B336)
    # return tag.name in ["p", "span"] and not tag.a and not _is_pagenum(tag)
    return tag.name in ["p"] and not tag.a and not _is_pagenum(tag)


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
    text_id = re.search(r"\d{4}:\d+", text["titel"])[0]
    logging.info(f"title: {text_id}")
    if soup.find("error"):
        logging.warning(r"""File contains errors, most likely due to problems\
        in the conversion to html.
        Refer to the HTML file for more information:
        {f'http:{text["content"]["dokument_url_html"]}'}""")
        return "", "", "", ""

    pages = soup.find_all(_pages)

    links = soup.find_all(_table_of_contents)
    logging.debug(f"Found {len(links)} links.")
    tables = soup.find_all(_tables_outside_toc)
    table_types = set(tables)
    copied_tables = "".join([str(t) for t in
                             table_types if tables.count(t) > 1])

    font_cl = re.findall(r"\.(ft\d+)\{font: +(\w+|\d+px) [^}]+\}",
                              str(soup.style))
    title_st = r"(bold|[2-9][0-9]px)"
    title_cl = [el[0] if type(el) == tuple else
                     el for el in font_cl if
                     re.search(title_st, el[1])]

    footer_st = r"(1[0-2]px|[0-9]px)"
    footer_cl = {x: y for x, y in font_cl
                      if re.match(footer_st, y)}

    def _determine_structure(text_part, element, text):
        """sort out whether an element is part of title or text body
        and add them to the respective Text object accordingly
        """
        nonlocal section_titles, seen_sections
        logging.debug(f"""segmenting {text}; the current section\
        titles are {section_titles[:2]}.
        The last 5 titles are {seen_sections[-5:]}""")
        
        font_cl = re.findall(r'class="[^"]*(ft\d+)[^"]*"', str(element))
        st = re.findall(r'style="[^"]*font:[^":]*(\b\d+px)[^"]*"',
                        str(element))
        if (text_part.content and text == text_part[-1].title)\
           or element.text.strip() in seen_sections[-5:]\
           or re.match(f"SOU *{text_id}", element.text.strip())\
           or re.search(f"SOU *{text_id}$", element.text.strip()):
            return
        elif not _is_pagenum(element):
            if (font_cl and
                len([f for f in font_cl if f in title_cl]) == len(font_cl)) or\
                (st and
                 len([s for s in st if re.search(title_st, s)]) == len(st)):
                logging.debug(f"comparing font class '{font_cl}' with title " +
                "style and known classes.")
                # is title
                text_part.append(Section(text.strip()))
                if (section_titles and
                    (re.sub(r"^\d+ *", "", text_part[-1].title.casefold())
                     == section_titles[0].casefold())):
                    seen_sections.append(section_titles.pop(0))
                else:
                    seen_sections.append(text_part[-1].title)
                logging.info(f"added new section '{text.strip()}'")
                logging.debug(f"""updated seen_sections to length\
                {len(seen_sections)} (cond 1)
                (the last 5 titles are {seen_sections}).""")
            elif (section_titles and text.strip().casefold()
                  .endswith(section_titles[0].casefold())):
                # is title
                text_part.append(Section(text.strip()))
                logging.info(f"added new section '{text.strip()}'")
                seen_sections.append(section_titles.pop(0))
                logging.debug(f"""updated seen_sections to length\
                {len(seen_sections)} (cond 2)
                (the last 5 titles are {seen_sections}).""")
            elif text != text_part.title:
                logging.debug("element is in text body")
                if text_part.content:
                    text_part[-1].append(text.strip())
                else:
                    logging.debug(f"text_part {text_part.title} is empty")
                    new_section = Section()
                    new_section.append(text.strip())
                    text_part.append(new_section)
                    logging.info(f"""added text to new section
                    {text_part.content[-1]}""")
            else:
                logging.warning("text is neither title nor text body!")

    # find the name of the section following the summary
    # and determine whether there are simplified or English summaries
    # from the table of contents
    logging.info("extracting links / table of contents")
    for i, hlink in enumerate(links):
        section_title = ""
        logging.debug(f"""current state: at index {i}:.{hlink}; 
        summary_title: {summary_title}, 
        end_of_summary: {end_of_summary}""")
        if section_titles:
            sec_title = re.match(r"[^.]+ *\.\.+$", hlink.text)
            logging.debug(f"section title candidate: '{sec_title}'")
            if sec_title:
                sec_title = sec_title[0].strip(". ")
                if sec_title != section_titles[-1]:
                    section_titles.append(sec_title)

        if not first_section and re.match(r"[A-Za-z]+", hlink.text)\
           and hlink.parent.name == "td":
            logging.debug("identified as first section in the document")
            first_section = hlink.text.strip(". ").casefold()

        if re.match(r"\bsammanfattning\b", hlink.text.casefold().strip())\
           and not has_summary:
            logging.debug("identified as summary title")
            summary_title = hlink.text.strip(". ")
            has_summary = True
            summary_section_nb = -1
            if hlink.previous_sibling and\
               hasattr(hlink.previous_sibling, "text"):
                match = re.match("[0-9.]+", hlink.previous_sibling.text)
                if match:
                    summary_section_nb = int(match.group().split(".")[0])

        if ("sammanfattning" in hlink.text.casefold() or
            ((has_english_summary or has_simple_summary) and
             end_of_summary)) or not end_of_summary:
            logging.debug("looking for the section following the summary ...")
            i += 1
            while (i < len(links) and not
                   re.search(r"[A-Za-z]+ ?\.+", links[i].text)):
                i += 1
            if i >= len(links):
                logging.debug(f"looped through all links and found " +
                              "no candidates (index {i}, " +
                              "list of length {len(links)}")
                break
            if (links[i].text.strip(". ").casefold() not in
                [summary_title.casefold(), s_summary_title.casefold(),
                 en_summary_title.casefold()]):
                if summary_section_nb < 0 and not section_titles:
                    logging.debug(f"end_of_summary set to: '{links[i].text}'")
                    end_of_summary = links[i].text.strip(". ")
                    if not first_section:
                        first_section = end_of_summary
                    section_title = end_of_summary
                if (summary_section_nb and
                    links[i-1].text.strip(". ")
                    .startswith(str(summary_section_nb+1))
                    and not section_titles):
                    end_of_summary = links[i].text.strip(". ")
                    logging.debug(f"""end_of_summary and summary_section_nb
                    identified as: '{links[i].text.lstrip('. ')}'; 
                    {summary_section_nb}""")
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
        logging.debug("-"*15)

    logging.info(f"indentified {len(section_titles)} section titles based " +
    "on table of contents")
    logging.debug(section_titles)
    logging.info(f"""extracted document information:
    contains En summary: \
    {f'Yes ({en_summary_title})' if has_english_summary else 'No'}
    contains Simple summary: \
    {f'Yes ({s_summary_title})' if has_simple_summary else 'No'}
    summary title: {summary_title}
    summary ends at: {end_of_summary}""")

    def _is_end_of_summary(element):
        """helper function to separate summary
        from (simplified/English version or full report"""
        nonlocal seen_sections
        if end_of_summary and element.text.endswith(end_of_summary):
            if section_titles and section_titles[0].casefold()\
               in element.text.casefold():
                logging.debug(f"found section title: " +
                              "{section_titles[0].casefold()}")
                seen_sections.append(section_titles.pop(0))
                logging.debug(f"update seen sections {seen_sections}")
            else:
                seen_sections.append(element.text)
            nonlocal full_text
            full_text.append(Section(element.text))
            return True
        elif has_english_summary and\
            re.match(r"\b" + en_summary_title.casefold()
                     + r"\b", element.text.casefold()):
            nonlocal is_english_summary, english_summary
            is_english_summary = True
            logging.info(f"found English summary")
            if not english_summary:
                english_summary = Text(element.text.strip(". "))
                seen_sections.append(english_summary.title)
        elif has_simple_summary and re.match(r"\b" + s_summary_title.casefold()
                                             + r"\b", element.text.casefold()):
            nonlocal is_simple_summary, simple_summary
            logging.info(f"found simple summary")
            is_simple_summary = True
            if not simple_summary:
                simple_summary = Text(element.text.strip(". "))
                seen_sections.append(simple_summary.title)
        return False

    def _footnotes(page):
        """find and remove footnotes"""
        def _footn_filter(tag):
            return (tag.name in ["span", "p"] and
                    (tag.has_attr("class") and tag["class"][-1] in
                     footer_cl) or
                    (tag.has_attr("style") and
                     re.search(footer_st, tag["style"])))

        candidates = page.find_all(_footn_filter)
        for c in candidates:
            if re.match(r"^\d+ *", c.text):
                if c.previous_sibling is None:
                    el = c.parent.extract()
                else:
                    el = c.extract()
                logging.debug(f"removed element {el}")

    def _two_columns(table):
        """detect and merge two column format"""
        columns = {}
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
                logging.debug(f"at element: {element}")
                if (re.match(r"(tabell|tablå|figur) *\d*",
                             element.text.casefold())
                    or (element.has_attr("class") and
                        f'{element["class"][-1]}">{element.text}</p></td>'
                        in copied_tables and
                        element["class"][-1] != "ft0")
                    or element.name == "table"):
                    element.extract()
                    logging.debug("ignoring header!")
                    continue
                elif element.parent.name in ["table", "td", "tr"]:
                    logging.debug("ignoring table!")
                    continue
                text = element.text

                # correct for p-elements with multiple children,
                # e.g. consecutive spans with missing white space
                children = element.contents
                if len(children) > 1:
                    text = insert_whitespace(children)
                if is_summary:
                    if is_english_summary:
                        _determine_structure(english_summary, element, text)
                    elif is_simple_summary:
                        _determine_structure(simple_summary, element, text)
                    else:
                        _determine_structure(summary, element, text)
                else:
                    if full_text:
                        _determine_structure(full_text, element, text)
                    else:
                        full_text = Text(text)

    logging.info("Starting text extraction ...\n")
    # text extraction
    _footnotes(soup)
    for page in pages:
        logging.debug(f"moving to new page")
        for i, p in enumerate(page.children):
            paragraphs = False
            if hasattr(p, "find_all"):
                if p.name == "p":
                    paragraphs = [p]
                else:
                    paragraphs = p.find_all(_freestanding_p)

            logging.debug(f"identified " +
                          "{len(paragraphs) if paragraphs else 'no'}" +
                          "paragraphs under current element")
            if not paragraphs or not p:
                continue
            logging.debug(f"""current state:
            is order info: {is_order_info}
            is table of contents: {is_table_of_c}
            is summary: {is_summary}
            is English summary: {is_english_summary}
            is simple summary: {is_simple_summary}
            first section: {first_section}
            end of summary: {end_of_summary}""")
            if p.text.rstrip().endswith("....") or\
               p.text.startswith("SOU och Ds kan köpas från "):
                logging.debug(f"skipping paragraph: {p}")
                break

            if p.name == "table":
                el = p.extract()
                logging.debug(f"ignoring element: {el}")
                continue
            elif (p.has_attr("class") and
                  f'{p["class"][-1]}">{p.text}</p></td>' in
                  copied_tables and p["class"][-1] != "ft0"):
                el = p.extract()
                logging.debug(f"ignoring header: {el}")
                continue

            elif (is_table_of_c or "...." in page.text)\
                and re.search(r"\binnehåll\b",
                              paragraphs[0].text.casefold())\
                or (len(paragraphs) > 1 and
                    re.search(r"\binnehåll\b",
                              paragraphs[1].text.casefold())):
                is_table_of_c = True
                logging.debug(f"ignoring table of contents '{p}'")
                break
            elif re.match(r"\b" + re.sub(r"([\[\]\(\)])", r"\\\1",
                                         summary_title.casefold())
                          + r"\b", paragraphs[0].text.casefold())\
                    and not summary:
                is_summary = True
                summary = Text(paragraphs[0].text.strip(". "))
                seen_sections.append(summary.title)
                paragraphs.pop(0)
                if not paragraphs:
                    continue
                is_table_of_c = False
                # continue

            elif is_summary and _is_end_of_summary(paragraphs[0]):
                is_summary = False

            if paragraphs[0].text.casefold().strip().endswith(first_section):
                is_table_of_c = False
            _split_text_on_page(paragraphs)

    # todo filter references and other lists
    # Litteraturförteckning Källförteckning Referenser Förkortningar
    def _filter(text, titles):
        """filter out sections with specific titles"""
        logging.info(f"Text before filtering: {text}")
        filter_terms = []
        sec_titles = text.section_titles
        for term in titles:
            if term in section_titles and term in text:
                start = text.index(term)
                stop = text.index(
                    section_titles[section_titles.index(term) + 1])
                logging.debug(f"discarding {stop - start}elements " +
                "from sections {start} to {stop} starting at title {term}")
                filter_terms.extend(sec_titles[start:stop])
            elif term in text:
                filter_terms.append(term)
        if filter_terms:
            contents = list(filter(lambda x: not
                                   re.match(rf"({'|'.join(filter_terms)})",
                                            x.title), text.content))
            text.content = contents
        logging.info(f"Text after filtering: {text}")

    _filter(full_text, ["Litteraturförteckning",
                        "Källförteckning",
                        "Referenser",
                        "Förkortningar",
                        f"Statens offentliga utredningar\
                        {text_id.split(':')[0]}"])
    # Todo match to section titles to also discard subsections
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


logging.basicConfig(filename="extraction.log",
                    filemode="w",
                    level=logging.INFO  # DEBUG)
                    )
# docs = extract_from_json()
with open("documents.pickle", "rb") as ifile:
    docs = pickle.load(ifile)
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


def run_example(key):
    global docs
    with open("output_examples", "w") as out:
        ft, s, es, ss = extract_from_html(docs[key])
    print_to_files(key, ft, s, es, ss)
    print(ft)

import json
import requests
import tqdm
import re
import os
import pickle
import bs4
import logging
from langdetect import detect_langs, lang_detect_exception

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
# English files GLB3132d1 & d2
# filter summaries by length
# bf in paragraphs not titles


def is_lang(lang, el):
    logging.debug(f"detecting language for {el}")
    url_or_email = r"https?://(\w+\./?)+\w+(/[\w.]*)*|[\w.]+@\w+\.\w+"
    el = re.sub(url_or_email, "", el).strip("()0123456789._-–…, ")
    logging.debug(f"detecting language for {el}")
    if len(el.split()) > 3:
        try:
            langs = {l.lang: l.prob for l in detect_langs(el)}
            logging.debug(langs)
            if max(langs.values()) < 0.5 or lang in langs.keys():
                return True
            return False
        except lang_detect_exception.LangDetectException:
            return False
    return True


class Section:
    """"""
    def __init__(self, title="", lang="sv"):
        self.title = title
        self.lang = lang
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
        if self.check_lang(el):
            self.text.append(el)

    def extend(self, el):
        self.text.extend(el)

    def pop(self, index):
        self.text.pop(index)

    def check_lang(self, el):
        if re.search(r"[^\d\W_…!?()&%€#£$§|]", el):
            if is_lang(self.lang, el):
                return True
            else:
                logging.warning(
                    "text does not match section " +
                    f"language {self.lang}: '{el}'")
                return False
        else:
            return True


class Text(object):
    """simple text representation"""
    def __init__(self, title, lang="sv"):
        self.title = title
        self.content = []
        self.lang = lang

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
        if self.content and not self.content[-1].text:
            if merge_title(self.content[-1], el):
                del self.content[-1]
        self.content.append(el)

    def remove(self, el):
        self.content.remove(el)

    def index(self, el, reverse=False):
        if reverse:
            return [sec.title for sec in self.content][::-1].index(el) - 1
        return [sec.title for sec in self.content].index(el)

    def check_lang(self, el):
        if re.search(r"[^\d\W_…!?()&%€#£$§|]", el):
            if is_lang(self.lang, el):
                return True
            else:
                logging.warning(
                    "section title does not match text " +
                    f"language {self.lang}: '{el}'")
                return False
        else:
            return True

    def from_html(self, path):
        with open(path) as ifile:
            doc = ifile.read()
        soup = bs4.BeautifulSoup(doc)
        if soup.h1:
            self.title = soup.h1.text
        current_el = soup.h1.next_element
        current_section = Section()
        while current_el.next_element:
            if current_el.name == "h2":
                if current_section.title or current_section.text:
                    self.content.append(current_section)
                    current_section = Section(current_el.text)
                else:
                    current_section.title = current_el.text
            if current_el.name == "p":
                current_section.text.append(current_el.text)
            current_el = current_el.next_element
        if current_section.title or current_section.text:
            self.content.append(current_section)

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


def merge_hyphenation(string):
    logging.debug("merging hyphenation for string of length "
                  + str(len(string)))
    candidates = re.findall(r"((\b\w+)- +(\b\w+\b))", string)
    for original, tok1, tok2 in candidates:
        if tok2.casefold() not in ["och", "eller"]:
            replacement = tok1 + tok2
            if tok2[0]*2 == tok1[-2:]:
                replacement = tok1 + tok2[1:]
            string = string.replace(original, replacement)
    logging.debug(f"new length: {len(string)}")
    return string


def merge_title(sec1, sec2):
    if sec1.title and sec1.title.split()[-1].islower() and\
       sec2.title and sec2.title[0] not in "0123456789":
        sec2.title = f"{sec1.title} {sec2.title}"
        logging.debug(f"removing {sec1}")
        return True
    return False


def merge_paragraphs(section):
    if hasattr(section, "text"):
        section = section.text
    i = 0
    logging.info(f"section length before merge {len(section)}")
    while (i + 1) < len(section):
        if not section[i].endswith('!;.?"'):
            if section[i + 1] and (not section[i + 1][0].isupper() and
                                   not section[i + 1].startswith("•")):
                section[i] += f" {section.pop(i+1).lstrip()}"
            else:
                i += 1
        else:
            i += 1
    for i, p in enumerate(section):
        section[i] = merge_hyphenation(p)
    logging.info(f"section length after merge {len(section)}")


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
        requests.get(f'http:{text["content"]["dokument_url_html"]}').text)  #,
        # parser="lxml")
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
    title_st = r"(bold|[2-9][0-9]px|1[6-9]px)"
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
                if text_part.content:
                    logging.debug("merging paragraphs in last section")
                    merge_paragraphs(text_part[-1])
                if text_part.check_lang(text.strip()):
                    text_part.append(Section(text.strip(),
                                             lang=text_part.lang))
                if (section_titles and text_part and text_part.content and
                    (re.sub(r"^\d+ *", "", text_part[-1].title.casefold())
                     == section_titles[0].casefold())):
                    seen_sections.append(section_titles.pop(0))
                elif text_part.content:
                    seen_sections.append(text_part[-1].title)
                logging.info(f"added new section '{text.strip()}'")
                logging.debug("updated seen_sections to length " +
                              f"{len(seen_sections)} (cond 1) " +
                              f"(the last 5 titles are {seen_sections}).")
            elif (section_titles and text.strip().casefold()
                  .endswith(section_titles[0].casefold()) and
                  text_part.check_lang(text.strip())):
                # is title
                text_part.append(Section(text.strip(), lang=text_part.lang))
                logging.info(f"added new section '{text.strip()}'")
                seen_sections.append(section_titles.pop(0))
                logging.debug("updated seen_sections to length " +
                              f"{len(seen_sections)} (cond 2) " +
                              f"(the last 5 titles are {seen_sections}).")
            elif text != text_part.title:
                logging.debug("element is in text body")
                if text_part.content:
                    text_part[-1].append(text.strip())
                elif text_part.check_lang(text.strip()):
                    logging.debug(f"text_part {text_part.title} is empty")
                    new_section = Section(lang=text_part.lang)
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
                logging.debug("looped through all links and found " +
                              "no candidates (index {i}, " +
                              f"list of length {len(links)}")
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
    if not end_of_summary:
        logging.debug(f"Setting default title for end_of_summary")
        end_of_summary = "Författningsförslag"
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
        nonlocal seen_sections, is_simple_summary, simple_summary,\
            is_english_summary, english_summary, summary, full_text
        if end_of_summary and re.search(rf"{end_of_summary.casefold()}\.?",
                                        element.text.casefold())\
                                        and summary and summary.content\
                                        and summary.content[0].text:
            if section_titles and section_titles[0].casefold()\
               in element.text.casefold():
                logging.debug("found section title: " +
                              f"{section_titles[0].casefold()}")
                seen_sections.append(section_titles.pop(0))
                logging.debug(f"update seen sections {seen_sections}")
            else:
                seen_sections.append(element.text)
            if full_text and full_text.check_lang(element.text.strip()):
                full_text.append(Section(element.text))
            return True
        elif has_english_summary and\
            re.match(r"\b" + en_summary_title.casefold()
                     + r"\b", element.text.casefold()):
            is_english_summary = True
            if simple_summary and simple_summary.content:
                merge_paragraphs(simple_summary[-1])
            elif summary and summary.content:
                merge_paragraphs(summary[-1])
            elif full_text and full_text.content:
                merge_paragraphs(full_text[-1])
            logging.info(f"found English summary")
            if not english_summary:
                english_summary = Text(element.text.strip(". "), lang="en")
                seen_sections.append(english_summary.title)
        elif has_simple_summary and re.match(r"\b" + s_summary_title.casefold()
                                             + r"\b", element.text.casefold()):
            if english_summary:
                merge_paragraphs(english_summary[-1])
            elif summary:
                merge_paragraphs(summary[-1])
            elif full_text:
                merge_paragraphs(full_text[-1])
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

            logging.debug("identified " +
                          f"{len(paragraphs) if paragraphs else 'no'}" +
                          " paragraphs under current element")
            if not paragraphs or not p:
                continue
            logging.debug(f"""current state:
            is order info: {is_order_info}
            is table of contents: {is_table_of_c}
            is summary: {is_summary}
            is English summary: {is_english_summary}
            is simple summary: {is_simple_summary}
            first section: '{first_section}'
            end of summary: '{end_of_summary}'""")
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
                if full_text and full_text.content:
                    merge_paragraphs(full_text[-1])
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
                stop_term_id = section_titles.index(term) + 1
                if stop_term_id < len(section_titles):
                    if (section_titles[stop_term_id] in text):
                        stop = text.index(
                            section_titles[stop_term_id])
                    elif start < (len(sec_titles) + 1):
                            stop = start + 1
                else:
                    stop = len(sec_titles) - 1
                logging.debug(f"discarding {stop - start} elements " +
                f"from sections {start} to {stop} starting at title {term}")
                filter_terms.extend(sec_titles[start:stop])
            elif term in text:
                previous_title = text.index(term) - 1
                if text[previous_title].title == section_titles[-10:]:
                    filter_terms.extend(sec_titles[previous_title + 1:])
                else:
                    filter_terms.append(term)
        if filter_terms:
            logging.debug(
                f"filtering out the following sections: {filter_terms}")
            term = re.sub(r"([\[\]\(\)])", r"\\\1",
                          '|'.join(filter_terms))
            term = rf'({term})'
            logging.debug(f"term: {term}")
            contents = list(
                filter(
                    lambda x: not re.match(term,
                                           x.title), text.content))
            text.content = contents
        logging.info(f"Text after filtering: {text}")
    if full_text:
        _filter(full_text, ["Innehållsförteckning",
                            "Innehåll",
                            "Litteratur",
                            "Litteraturförteckning",
                            "Tabellförteckning",
                            "Förteckning över tabeller och diagram",
                            "Källförteckning",
                            "Referenser",
                            "Förkortningar",
                            "Statens offentliga utredningar " +
                            text_id.split(':')[0].strip()])
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
                    # level=logging.DEBUG
                    level=logging.INFO
                    )
with open("documents.pickle", "rb") as ifile:
    docs = pickle.load(ifile)


def run_example(key):
    global docs
    ft, s, es, ss = extract_from_html(docs[key])
    print_to_files(key, ft, s, es, ss)
    print(ft)

# for key in ["H8B336", "H4B319", "GIB33"]:
#     run_example(key)

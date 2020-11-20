import json
import requests
import tqdm
import re
import os
import pickle
import bs4

# name format is sou_<year>_<id>.pdf or column 1 for html (check column 7 and 16 for sou with multiple parts at https://data.riksdagen.se/dokument/
# do we care about relations between SOUs in multiple parts?
# are the summaries useful? 
# look for json and extract directly through document_url_text / document_url_html
# verify if we get all the documents
# hyphenation
# some have English summary?
# to keep or to discard headings / titles?
# summary separate from full text?
# remove date and place + SOU ID?
# work with two-column format -> one case of suggested change of wording i.e. large part of the texts are really similar

class Section:
    """"""
    def __init__(self):
        self.title = ""
        self.text = []

    def __repr__(self):
        return f"Section object with title: '{self.title}', length: {len(self)}"

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
            return self.title == other.title and self.text == other.text
        return False

    
    def append(self, el):
        self.text.append(el)
        
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

        
    # Set this right
    def __bool__(self):
        return bool(self.title)

    
    def __eq__(self, other):
        if type(other) == Text:
            return self.title == other.title and self.content == other.content
        return False
        
    def append(self, el):
        self.content.append(el)
        
def retrieve_ids(sou_csv):
    """retrieves name of pdf and/or html document"""
    pass


def extract_from_json():
    """extract document date, text and html locations from respective json file"""
    
    page = json.load(open("../data.riksdagen.se.json"))
    nb_pages = int(page["dokumentlista"]["@sidor"])
    
    documents = {}
    
    for i in tqdm.tqdm (range(nb_pages), desc="Extracting...", ncols=100):
        for document in page["dokumentlista"]["dokument"]:
            doc_id = document["id"]
            documents[doc_id] = {"titel": document["sokdata"]["titel"] + "/"\
                                 + document["sokdata"]["undertitel"],
                                 "datum": document["datum"],
                                 "content": {match:document[match] for match in document.keys() if "dokument_url_" in match}
            }
            #if "filbilaga" in document and "fil" in document["filbilaga"]:
            #    documents[doc_id]["content"]["pdf"] = document["filbilaga"]["fil"]["url"]

            
        if "@nasta_sida" in page["dokumentlista"]:
            # apparently, the next page is not saved as json anymore
            page = json.loads(requests.get(page["dokumentlista"]["@nasta_sida"].replace("=htm&", "=json&")).text)

    return documents


def _freestanding_p(tag):
    """verify that text is actual content and not part of header or table of contents"""
    def _is_pagenum(tag):
        match = re.match("[0-9]+", tag.text)
        if match:
            return match.span()[1] - match.span()[0] == len(tag.text)
        return False
    #replace (tag.parent.name != "td") by not tag.a to keep two-column instances (e.g. H8B336)
    return tag.name == "p" and not tag.a and not _is_pagenum(tag)

def _table_of_contents(tag):
    """filter out links that are not for navigation in the html or other paragraph elements"""
    return (tag.name == "a" and tag["href"].startswith("#page"))\
        or ((tag.name in ["p","span"] and tag.text.strip().endswith(".."))\
            or (tag.name == "p" and type(tag.next_sibling) == bs4.element.Tag and tag.next_sibling.text.endswith("..")))


def _tables_outside_toc(tag):
    """useful for finding headers and real tables"""
    return tag.name == "table" and not tag.find("a")


def _appendix(tag):
    """find appendices in table of content"""
    return tag.name == "td" and re.match("bilaga|bilagor", tag.text.casefold().strip())


def insert_whitespace(children):
    """correct for elements with multiple children with missing white space in between"""
    text = ""
    for i, child in enumerate(children[:-1]):
        if (not (type(child) == bs4.element.NavigableString and child.endswith(" "))\
            or (type(child) != bs4.element.NavigableString and child.text.endswith(" "))) and\
            (not (type(children[i+1]) == bs4.element.NavigableString and children[i+1].startswith(" "))\
             or (type(children[i+1]) != bs4.element.NavigableString and children[i+1].text.startswith(" "))):
             if type(child) == bs4.element.NavigableString:
                 text += f"{child} "
             else:
                 text += f"{child.text} "
        else:
            if type(child) == bs4.element.NavigableString:
                 text += f"{child}"
            else:
                 text += f"{child.text}"
    if type(children[-1]) == bs4.element.NavigableString:
        text += f"{children[-1]} "
    else:
        text += f"{children[-1].text} "
    return text


def extract_from_html(text):
    """extract text as list of sentences and return full text and summary"""
    # text segments to distinguish
    is_summary, is_english_summary, is_simple_summary, is_appendix = False, False, False, False
    summary_section_nb, end_of_summary = 0, ""
    has_english_summary, has_simple_summary, has_appendix = True, True, False
    summary_title, en_summary_title = "Sammanfattning", "Summary"
    section_titles = []
    
    
    # return variables
    #summary, english_summary, simple_summary, full_text = [], [], [], []
    summary, english_summary, simple_summary, full_text = None, None, None, None

    # text segments to exclude
    first_section = ""
    is_order_info, is_table_of_c = False, False

    soup = bs4.BeautifulSoup(requests.get(f'http:{text["content"]["dokument_url_html"]}').text)
    # there does not seem to be any difference between headings and text in the markup
    # hence, paragraphs can be a short title or multiple complete sentences
    paragraphs = soup.find_all(_freestanding_p)  
    
    links = soup.find_all(_table_of_contents)
    print(len(links))
    tables = soup.find_all(_tables_outside_toc)
    table_types = set(tables)
    copied_tables = "".join([str(t) for t in table_types if tables.count(t) > 1])

    # find class names for section title assuming that they are all in bold font
    #title_classes = re.findall("\.(ft\d+)\{font: +[2-9][0-9]px [^}]+\}", str(soup.style))
    ##title_classes = [el[0] if type(el)==tuple else el for el in title_classes]
    #bold = re.findall("\.(ft\d+)\{font: bold [^}]+\}", str(soup.style))
    # create dict of types and compare to mean / max point size
    ##print([el for el in title_classes if el not in old])
    #text_font_size = re.findall("\.ft\d+\{font: +([0-9]+)px [^}]+\}", str(soup.style))
    #mean_font_size = str(math.ceil(sum([int(el) for el in text_font_size])/len(text_font_size))+1)
    #print(f"mean font size: {mean_font_size}")
    title_classes = re.findall("\.(ft\d+)\{font: +(bold|[2-9][0-9]px) [^}]+\}",
                               str(soup.style))
    title_classes = [el[0] if type(el)==tuple else el for el in title_classes]
    # keep track of the font size for each style
    # title_classes = {name: value for name, value in title_classes}
    # font_to_styles = {}
    # for name, font in title_classes.items():
    #     if font in font_to_styles:
    #         font_to_styles[font].append(name)
    #     else:
    #         font_to_styles[font] = [name]
    
    #old = re.findall("\.(ft\d+)\{font: bold [^}]+\}", str(soup.style))
    #print([el for el in title_classes if el not in old])
    #assert title_classes, "No section headings"
        
    def _determine_structure(text_part, element, string):
        #__structure(summary, p, text)
        """sort out whether an element is part of title or text body
        and add them to the respective Text object accordingly
        """
        nonlocal section_titles
        print("here", section_titles[:2], string, file=out)
        font_class = re.search('class="[^"]*(ft\d+)[^"]*"', str(element))
        # if we didn't want to treat unnumbered titles as such
        # if (font_class and font_class[1] in title_classes) or\
        #(font_class and font_class in bold and re.match("\d[.\d+]*", element.text)):
        if text_part.content and text == text_part[-1].title:
            pass
        else:
            if font_class and font_class[1] in title_classes:
                print(f"_determine_structure '{font_class[1]}'",file=out)
                # is title
                new_section = Section()
                new_section.title = text
                text_part.append(new_section)
                print("new section",file=out)
            elif string.strip().endswith(section_titles[0]):
                # is title
                new_section = Section()
                new_section.title = text
                text_part.append(new_section)
                print("new section 2",file=out)
                section_titles.pop(0)
            elif text != text_part.title:
                # is text body
                if text_part.content:
                    text_part[-1].append(text)

                else:
                    # text_part is empty
                    new_section = Section()
                    new_section.append(text)
                    text_part.append(new_section)
                
    # appendix = soup.find_all(_appendix)[0].text.strip()
    # if appendix:
    #     has_appendix = True
    
    # find the name of the section following the summary
    # and determine whether there are simplified or English summaries from the table of contents
    for i, hlink in enumerate(links):
        #print(i, hlink, summary_title, en_summary_title, end_of_summary, file=out)
        if section_titles:
            sec_title = re.match(r"[^.]+ *\.\.+$", hlink.text)
            print("hello", hlink.text, sec_title, file=out)
            if sec_title:
                sec_title = sec_title[0].strip(". ")
                if sec_title != section_titles[0]:
                    section_titles.append(sec_title)
        if not first_section and re.match(r"[A-Za-z]+", hlink.text) and hlink.parent.name == "td":
            print(2, file=out)
            first_section = hlink.text.strip(". ").casefold()
        if re.match(r"\bsammanfattning\b", hlink.text.casefold().strip()):
            print(3, file=out)
            summary_title = hlink.text.strip(". ")
            summary_section_nb = -1
            #print("Update summary_section_nb", summary_section_nb)
            if hlink.previous_sibling:
                #print(f"text: '{links[i-1].text}'")
                match = re.match("[0-9.]+", hlink.previous_sibling.text)
                if match:
                    summary_section_nb = int(match.group().split(".")[0])
                    #print("ssn", summary_section_nb)

        if ("sammanfattning" in hlink.text.casefold() or\
           ((has_english_summary or has_simple_summary) and end_of_summary)) or\
           not end_of_summary:
            print(4, file=out)
            #print("this", hlink, summary_section_nb)
            i += 1
            #print("new loop", len(links), i)
            while i < len(links) and not re.search(r"[A-Za-z]+ ?\.+", links[i].text):
                # or links[i+1].text.strip().endswith("Sammanfattning"):
                #print(links[i], i, len(links))
                i += 1
            if i >= len(links):
                print("stop", i, len(links), summary_section_nb, file=out)
                break
            #print("stop", links[i], i, len(links), summary_section_nb)
            if summary_section_nb < 0 and links[i].text.strip(". ").casefold() not in ["sammanfattning", "summary"]\
               and not section_titles:
                print("this one", hlink.text, links[i].text, i, file=out)
                end_of_summary = links[i].text.strip(". ")
                if not first_section:
                    first_section = end_of_summary
                section_title = end_of_summary
                #print(f'1 "{end_of_summary}", "{summary_section_nb}"')
            if (summary_section_nb and\
                links[i-1].text.strip(". ").startswith(str(summary_section_nb+1))\
                and not section_titles):
                end_of_summary = links[i].text.strip(". ")
                print("that one", summary_section_nb, links[i-1].text.strip(". "), summary_section_nb+1, file=out)
                print(f'2 "{end_of_summary}" "{links[i-1]}" "{summary_section_nb}"')
                section_title = end_of_summary
                if not first_section:
                    first_section = end_of_summary

            else:
                section_title = links[i].text.strip(". ")
            print("second block", file=out)
            if re.match(r"\bsummary\b", section_title.casefold()):
                has_english_summary = True
                en_summary_title = section_title
            elif section_title.casefold() == "lättläst sammanfattning":
                has_simple_summary = True
            #elif summary_section_nb < 0 and end_of_summary:
                #print("stop", hlink, i, f"'{end_of_summary}'")
                #break
        if end_of_summary and not section_titles:
            
            section_titles.append(end_of_summary)
        print(6, file=out)

    print("section_titles", len(section_titles))
    print(section_titles, file=out)
    print(has_english_summary, has_simple_summary, f"'{end_of_summary}'", f"'{summary_title}'")
    def _is_end_of_summary(element):
        """helper function to separate summary from (simplified/English version or full report"""
        if end_of_summary and element.text.endswith(end_of_summary):
            section_titles.pop(0)
            return True
        elif has_english_summary and re.match(r"\b" + en_summary_title.casefold() + r"\b", element.text.casefold()):
            nonlocal is_english_summary, english_summary
            is_english_summary = True
            if not english_summary:
                english_summary = Text(element.text)
        elif has_simple_summary and element.text.casefold().endswith("lättläst sammanfattning"):
            nonlocal is_simple_summary, simple_summary
            is_simple_summary = True
            simple_summary = Text(element.text)

            

        return False
    
    # text extraction
    for i, p in enumerate(paragraphs):
        #print(i, p, is_summary, is_table_of_c, is_order_info, is_english_summary, is_simple_summary, first_section, end_of_summary, file=out)
        if p.text.startswith("SOU och Ds kan köpas från "):
            is_order_info = True
        elif is_order_info:
            if p.text.startswith("ISSN"):
                is_order_info = False
                continue
        elif not is_table_of_c and p.text.casefold() == "innehåll":
            is_table_of_c = True

        elif re.match(r"\b" + summary_title.casefold() + r"\b", p.text.casefold()) and not summary:
            is_summary = True
            summary = Text(p.text)            
            is_table_of_c = False
            continue

        elif p.has_attr("class") and f'{p["class"][-1]}">{p.text}</p></td>' in copied_tables and p["class"][-1] != "ft0":
            print("holla", file=out)
            continue
        elif is_summary and _is_end_of_summary(p):
            is_summary = False

        if p.text.casefold().strip().endswith(first_section):

            is_table_of_c = False

        text = p.text
        
        # correct for p-elements with multiple children, e.g. consecutive spans with missing white space
        children = list(p.children)
        if len(children) > 1:
            text = insert_whitespace(children)
        if is_summary:
            if is_english_summary:
                #english_summary.append(text)
                _determine_structure(english_summary, p, text)
            elif is_simple_summary:
                #simple_summary.append(text)
                _determine_structure(simple_summary, p, text)
            else:
                _determine_structure(summary, p, text)
                #summary.append(text)
        elif not is_table_of_c and not is_order_info:
            if full_text:
                # is content
                #print("full text", full_text == True, full_text, file=out)
                _determine_structure(full_text, p, text)
            else:
                #print("No full text", full_text == True, full_text, file=out)
                full_text = Text(text)
            #full_text.append(text)
        if not is_table_of_c and p.parent.name == "td":
            global also_tables
            also_tables.add(soup.head.title)
    return full_text, summary, english_summary, simple_summary


def print_to_files(id_, text, summary, english_summary, simple_summary):
    if not os.path.exists("documents"):
        os.system("mkdir documents")
    for name, text_part in [("ft", text), ("s", summary), ("ENs", english_summary), ("SEs", simple_summary)]:
        if text_part:
            with open(f"documents/{name}_{id_}", "w") as ofile:
                print(f"<h1>{text_part.title}</h1>", file=ofile)
                for section in text_part.content:
                    print(f"<h2>{section.title}</h2>", file=ofile)
                    for line in section.text:
                        print(f"<p>{line}</p>", file=ofile)
                    
#docs = extract_from_json()
# with open("documents.pickle","rb") as ifile:
# docs = pickle.load(ifile)
#ft, s = extract_from_html(docs["H8B36"])
#len(ft) # 2028
#for key in list(docs.keys())[:10]:
#ft, s, es, ss =  extract_from_html(docs[key])
#print_to_files(key, ft, s, es, ss)
# verify that the full text begins where we want it to start (often Författningsförslag / Inledning)

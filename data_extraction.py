import json
import requests
import tqdm
import re
import os
import pickle
from bs4 import BeautifulSoup

# name format is sou_<year>_<id>.pdf or column 1 for html (check column 7 and 16 for sou with multiple parts at https://data.riksdagen.se/dokument/
# do we care about relations between SOUs in multiple parts?
# are the summaries useful? 
# look for json and extract directly through document_url_text / document_url_html
# verify if we get all the documents
# hyphenation
# some have English summary?

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
    #replace (tag.parent.name != "td") by not tag.a to keep two-column instances
    return tag.name == "p" and not tag.a and not _is_pagenum(tag)


def extract_from_html(text):
    """extract text as list of sentences and return full text and summary"""
    # text segments to distinguish
    is_summary, is_english_summary, is_simple_summary = False, False, False
    end_of_summary = ""
    has_english_summary, has_simple_summary = False, False

    # return variables
    summary, english_summary, simple_summary, full_text = [], [], [], []

    # text segments to exclude
    first_section = ""
    is_order_info, is_table_of_c = False, False

    soup = BeautifulSoup(requests.get(f'http:{text["content"]["dokument_url_html"]}').text)
    # there does not seem to be any difference between headings and text in the markup
    # hence, paragraphs can be a short title or multiple complete sentences
    paragraphs = soup.find_all(_freestanding_p)
    # to keep or to discard headings / titles?
    # summary separate from full text?
    # remove date and place + SOU ID?
    # work with two-column format -> one case of suggested change of wording i.e. large part of the texts are really similar
   
    
    links = soup.find_all("a")

    # find the name of the section following the summary
    for i,hlink in enumerate(links):
        if not first_section and re.match(r"[A-Za-z]", hlink.text):
            first_section = hlink.text.strip(". ").casefold()
        if "Sammanfattning".casefold() in hlink.text.casefold() or\
           (has_english_summary or has_simple_summary):
            while not re.match(r"[A-Za-z]", links[i+1].text):# or links[i+1].text.strip().endswith("Sammanfattning"):
                i += 1
            end_of_summary = links[i+1].text.strip(".").strip()
            print(f'"{end_of_summary}"')
            if end_of_summary.casefold() == "summary":
                has_english_summary = True
            elif end_of_summary.casefold() == "lättläst sammanfattning":
                has_simple_summary = True
            else:
                break
    print(end_of_summary, has_english_summary, has_simple_summary)

    
    def _is_end_of_summary(element):
        """helper function to separate summary from (simplified/English version or full report"""
        if element.text.endswith(end_of_summary):
            return True
        elif has_english_summary and element.text.casefold().endswith("summary"):
            nonlocal is_english_summary
            is_english_summary = True
        elif has_simple_summary and element.text.casefold().endswith("lättläst sammanfattning"):
            nonlocal is_simple_summary
            is_simple_summary = True
        return False

    
    for i, p in enumerate(paragraphs):
        if p.text.startswith("SOU och Ds kan köpas från Norstedts Juridiks kundservice."):
            is_order_info = True
        elif is_order_info:
            if p.text.startswith("ISSN"):
                is_order_info = False
                continue      
        elif p.text.casefold() == "sammanfattning" and not summary:
            is_summary = True
        elif _is_end_of_summary(p):
            is_summary = False
        elif p.text.casefold() == "innehåll":
            is_table_of_c = True
        if p.text.casefold() == first_section:
            is_table_of_c = False
        text = p.text
        # correct for consecutive spans with missing white space
        if p.span:
            spans = p.find_all("span")
            if len(spans) > 1:
                text = ""
                for i, span in enumerate(spans[:-1]):
                    if not span.text.endswith(" ") and not spans[i+1].text.startswith(" "):
                        text += f"{span.text} "
                text += spans[-1].text
        if is_summary:
            if is_english_summary:
                english_summary.append(text)
            elif is_simple_summary:
                simple_summary.append(text)
            else:
                summary.append(text)
        elif not is_table_of_c and not is_order_info: #?
            full_text.append(text)
    return full_text, summary, english_summary, simple_summary


def print_to_files(id_, text, summary, english_summary, simple_summary):
    if not os.path.exists("documents"):
        os.system("mkdir documents")
    for name, content in [("ft", text), ("s", summary), ("ENs", english_summary), ("SEs", simple_summary)]:
        if content:
            with open(f"documents/{name}_{id_}", "w") as ofile:
                for line in content:
                    print(line, file=ofile)
#docs = extract_from_json()
#ft, s = extract_from_html(docs["H8B36"])
#len(ft) # 2028
#for key in list(docs.keys())[:10]:
#ft, s, es, ss =  extract_from_html(docs[key])
#print_to_files(key, ft, s, es, ss)

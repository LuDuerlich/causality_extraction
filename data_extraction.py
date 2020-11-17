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

def _internal_link(tag):
    """filter out links that are not for navigation in the html"""
    return tag.name == "a"and tag["href"].startswith("#page")
    
def extract_from_html(text):
    """extract text as list of sentences and return full text and summary"""
    # text segments to distinguish
    is_summary, is_english_summary, is_simple_summary = False, False, False
    summary_section_nb, end_of_summary = 0, ""
    has_english_summary, has_simple_summary = False, False

    # return variables
    summary, english_summary, simple_summary, full_text = [], [], [], []

    # text segments to exclude
    first_section = ""
    is_order_info, is_table_of_c = False, False

    soup = bs4.BeautifulSoup(requests.get(f'http:{text["content"]["dokument_url_html"]}').text)
    # there does not seem to be any difference between headings and text in the markup
    # hence, paragraphs can be a short title or multiple complete sentences
    paragraphs = soup.find_all(_freestanding_p)
    # to keep or to discard headings / titles?
    # summary separate from full text?
    # remove date and place + SOU ID?
    # work with two-column format -> one case of suggested change of wording i.e. large part of the texts are really similar
   
    
    links = soup.find_all(_internal_link)

    # find the name of the section following the summary
    for i,hlink in enumerate(links):
        if not first_section and re.match(r"[A-Za-z]+", hlink.text) and hlink.parent.name == "td":
            first_section = hlink.text.strip(". ").casefold()
        if re.match("sammanfattning", hlink.text.casefold().strip()):
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
            #print("this", hlink, summary_section_nb)
            i += 1
            #print("new loop", len(links), i)
            while i < len(links) and not re.search(r"[A-Za-z]+ ?\.+", links[i].text):
                # or links[i+1].text.strip().endswith("Sammanfattning"):
                #print(links[i], i, len(links))
                i += 1
            if i >= len(links):
                print("No summary")
                break
            #print("stop", links[i], i, len(links), summary_section_nb)
            if summary_section_nb < 0:
                #print(hlink.text, links[i].text, i)
                end_of_summary = links[i].text.strip(". ")
                section_title = end_of_summary
                print(f'1 "{end_of_summary}", "{summary_section_nb}"')
            if (summary_section_nb and links[i-1].text.strip(". ").startswith(str(summary_section_nb+1))):
                end_of_summary = links[i].text.strip(". ")
                #print(summary_section_nb, links[i-1].text.strip(". "), summary_section_nb+1)
                print(f'2 "{end_of_summary}" "{links[i-1]}" "{summary_section_nb}"')
                section_title = end_of_summary
            else:
                section_title = links[i].text.strip(". ")
            if section_title.casefold() == "summary":
                has_english_summary = True
            elif section_title.casefold() == "lättläst sammanfattning":
                has_simple_summary = True
            elif summary_section_nb < 0:
                break
                

    def _is_end_of_summary(element):
        """helper function to separate summary from (simplified/English version or full report"""
        if end_of_summary and element.text.endswith(end_of_summary):
            return True
        elif has_english_summary and element.text.casefold().endswith("summary"):
            nonlocal is_english_summary
            is_english_summary = True
        elif has_simple_summary and element.text.casefold().endswith("lättläst sammanfattning"):
            nonlocal is_simple_summary
            is_simple_summary = True
        return False

    
    for i, p in enumerate(paragraphs):
        #print(i, p, is_summary, is_table_of_c, is_english_summary, is_simple_summary, first_section)
        if p.text.startswith("SOU och Ds kan köpas från Norstedts Juridiks kundservice."):
            is_order_info = True
        elif is_order_info:
            if p.text.startswith("ISSN"):
                is_order_info = False
                continue      
        elif p.text.casefold() == "sammanfattning" and not summary:
            is_summary = True
            is_table_of_c = False
        elif is_summary and _is_end_of_summary(p):
            is_summary = False
            print(f"stop summary '{p}'")
        elif p.text.casefold() == "innehåll":
            is_table_of_c = True
        if p.text.casefold().strip().endswith(first_section):
            is_table_of_c = False
        text = p.text
        # correct for p-elements with multiple children, e.g. consecutive spans with missing white space
        children = list(p.children)
        if len(children) > 1:
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

        if is_summary:
            if is_english_summary:
                english_summary.append(text)
            elif is_simple_summary:
                simple_summary.append(text)
            else:
                summary.append(text)
        elif not is_table_of_c and not is_order_info: #?
            full_text.append(text)
        if not is_table_of_c and p.parent.name == "td":
            global also_tables
            also_tables.add(soup.head.title)
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
# with open("documents.pickle","rb") as ifile:
# docs = pickle.load(ifile)
#ft, s = extract_from_html(docs["H8B36"])
#len(ft) # 2028
#for key in list(docs.keys())[:10]:
#ft, s, es, ss =  extract_from_html(docs[key])
#print_to_files(key, ft, s, es, ss)
# verify that the full text begins where we want it to start (often Författningsförslag / Inledning)

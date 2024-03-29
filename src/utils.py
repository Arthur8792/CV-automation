"""
Implementation of some useful functions for the pipeline.
"""
import json
from PyPDF2 import PdfReader 
from docxtpl import DocxTemplate

def pdf_to_txt(pdf_file: str)->str:
    """
    Extracts text from the PDF file in parameters.

    Args:
        pdf_file : path of the PDF file
    Returns:
        The text contained in the PDF
    """
    # Read PDF file
    reader = PdfReader(pdf_file)
    content = ""
    # For each page, add the text to the content variable
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text = page.extract_text()
        content +=extracted_text
    return content

def json_to_dico(json_file: str)->dict[str,str|list[str]]:
    """
    Reads json file in parameter and store in contents in a dictionary.

    Args:
        json_file : path of the JSON file
    Returns:
        the dictionary which contains the elements from the json file.
    """
    dico = {} 
    with open(json_file,'r', encoding='utf-8') as file:
        dico = json.load(file)
    return dico

def dico_to_json(dico: dict[str,str|list[str]], json_file: str)->None:
    """
    Writes the contents of the dictionary in parameter in a json file.

    Args:
        dico : the dictionary to store in JSON file.
        json_file : path of the JSON file 
    Returns:
        None
    """
    with open(json_file,'w', encoding='utf-8') as file:
        json.dump(dico, file, ensure_ascii=True)

def fill_docx_template(dico: dict[str,str], template_file: str, docx_file: str)->None:
    """
    Fills the docx file template with the elements in the dictionary.

    Args:
        dico : the dictionary which contains elements to fill the template.
        template_file : path of the docx template.
        docx_file : path of the file which will be generated with filled template.
    Returns:
        None
    """
    template = DocxTemplate(template_file)
    template.render(dico)
    template.save(docx_file)
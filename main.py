from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader 

OPENAI_MODEL = 'gpt-3.5-turbo'
OPENAI_API_KEY = ''

# Get OpenAI API key : 
with open('api_key.txt','r') as api_key_file:
    OPENAI_API_KEY = api_key_file.readline().strip('\n')

def pdf_to_txt(pdf_file: str)->None:
    """
    Converts a pdf file to a txt file.

    Args:
        pdf_file : path of the PDF file.
    """
    reader = PdfReader(pdf_file)
    with open("result.txt","w",encoding='utf-8') as file:
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_text = page.extract_text()
            file.write(extracted_text)

def ask_question(model: str, api_key: str, question: str)->str:
    """
    Creates the langchain LLM model using the OpenAI API key.

    Args:
        model: the LLM to use
        api_key : the OpenAI API key
        question : the question to ask
    Returns:
        The response to the question.
    """
    llm = ChatOpenAI(model=model, openai_api_key = api_key)
    response = llm.invoke(question)
    return response.content

def main():
    # Entry PDF file
    pdf_file = 'original-cv/Exemple1.pdf'
    # Convert from PDF to txt
    pdf_to_txt(pdf_file)
    question = "Introduce yourself"
    response = ask_question(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, question=question)
    print(response)

if __name__ == '__main__':
    main()

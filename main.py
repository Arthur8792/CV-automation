from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
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
    with open("cv_content.txt","w",encoding='utf-8') as file:
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_text = page.extract_text()
            file.write(extracted_text)

def main():

    # Entry PDF file
    pdf_file = 'original-cv/Exemple1.pdf'

    # Convert from PDF to txt
    pdf_to_txt(pdf_file)

    # Extract text contents of the CV
    cv_file = open("cv_content.txt","r")
    cv_content = cv_file.read()

    ## LLM PART

    # Create LLM model
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

    # Create prompts
    prompt1 = PromptTemplate(
        input_variables=['CV'],
        template = "Donne-moi la formation la plus récente du CV suivant : {CV}"
    )

    prompt2 = PromptTemplate(
        input_variables=['CV'],
        template = "Liste-moi ses compétences techniques, en détaillant les langages, les outils collaboratifs, et les outils métiers du CV suivant : {CV}"
    )

    # Create chains : 
    chain1 = LLMChain(llm = llm, prompt=prompt1, output_key="derniere-formation")
    chain2 = LLMChain(llm = llm, prompt=prompt2, output_key="competences-techniques")

    # Create sequence of these chains :
    sequential_chain = SequentialChain(
        chains = [chain1, chain2],
        input_variables = ["CV"],
        output_variables = ["derniere-formation","competences-techniques"]
    )

    result = sequential_chain.invoke({"CV":cv_content})

    ## END LLM PART

    # Close CV file : 
    cv_file.close()

    # Write result in TXT file : 
    with open("result.txt",'w', encoding='utf-8') as result_file:
        result_file.write("Dernière formation : " + result["derniere-formation"] + "\n")
        result_file.write("Compétences techniques : " + result["competences-techniques"] + "\n")

if __name__ == '__main__':
    main()

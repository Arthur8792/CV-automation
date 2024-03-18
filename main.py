import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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
    
    dico_prompts = {}

    # Get prompts from JSON : 
    with open("prompts.json",'r') as json_file:
        dico_prompts = json.load(json_file)
    
    chains = []
    output_variables = []

    for prompt in dico_prompts["prompts"]:
        input_variables = prompt["input_variables"]
        template = prompt["content"]
        output_key = prompt["output_key"]

        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=template
        )

        chain = LLMChain(llm = llm, prompt=prompt_template, output_key = output_key)

        chains.append(chain)
        output_variables.append(output_key)

    # Create sequence of these chains :
    sequential_chain = SequentialChain(
        chains = chains,
        input_variables = ["CV"],
        output_variables = output_variables
    )

    result = sequential_chain.invoke({"CV":cv_content})

    ## END LLM PART

    # Close CV file : 
    cv_file.close()

    # Store results in a dictionary : 
    dico_result = {}
    for output in output_variables:
        if output not in dico_result:
            dico_result[output] = result[output]
    
    # Save results in a JSON file
    with open("result.json",'w', encoding='utf-8') as result_file:
        json.dump(dico_result, result_file, ensure_ascii=False)

if __name__ == '__main__':
    main()

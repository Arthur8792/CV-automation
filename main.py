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
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

    # Create initial prompt
    initial_prompt = PromptTemplate(
        input_variables = ['CV', 'chat_history', 'human_input'],
        template = "Lis attentivement le CV suivant : {CV}. Je vais ensuite te poser des questions dessus."
    )

    # Create prompts
    prompt1 = PromptTemplate(
        input_variables = ['chat_history', 'human_input'],
        template = "{chat_history} \n Donne-moi la formation la plus récente de ce CV \n {human_input}"
    )

    prompt2 = PromptTemplate(
        input_variables=['chat_history', 'human_input'],
        template = "{chat_history} \n Liste-moi les compétences techniques, en détaillant les langages, les outils collaboratifs, et les outils métiers \n {human_input}"
    )

    # Create chains : 
    initial_chain = LLMChain(llm = llm, prompt = initial_prompt, memory = memory)
    chain1 = LLMChain(llm = llm, prompt=prompt1, output_key="derniere-formation", memory = memory)
    chain2 = LLMChain(llm = llm, prompt=prompt2, output_key="competences-techniques", memory = memory)

    # Create sequence of these chains :
    sequential_chain = SequentialChain(
        chains = [initial_chain, chain1, chain2],
        input_variables = ["CV","human_input"],
        output_variables = ["derniere-formation","competences-techniques"]
    )

    result = sequential_chain.invoke({"CV":cv_content, "human_input":""})

    print(result)
    ## END LLM PART

    # Close CV file : 
    cv_file.close()

    # Write result in TXT file : 
    with open("result.txt",'w', encoding='utf-8') as result_file:
        result_file.write("Dernière formation : " + result["derniere-formation"] + "\n")
        result_file.write("Compétences techniques : " + result["competences-techniques"] + "\n")

if __name__ == '__main__':
    main()

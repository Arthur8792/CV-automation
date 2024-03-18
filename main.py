from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI

from src.utils.utils import pdf_to_txt, json_to_dico, dico_to_json 

OPENAI_MODEL = 'gpt-3.5-turbo'
OPENAI_API_KEY = ''

# Get OpenAI API key : 
with open('api_key.txt','r') as api_key_file:
    OPENAI_API_KEY = api_key_file.readline().strip('\n')


def main():

    # FIles paths
    pdf_file = 'data/cv/Exemple1.pdf'
    prompts_file = 'data/prompts/prompts.json'
    result_file = 'data/results/results.json'

    # Convert from PDF to txt
    cv_content = pdf_to_txt(pdf_file)

    ## LLM PART

    # Create LLM model
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    
    dico_prompts = json_to_dico(prompts_file)
    
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

    # Store results in a dictionary : 
    dico_result = {}
    for output in output_variables:
        if output not in dico_result:
            dico_result[output] = result[output]
    
    dico_to_json(dico_result, result_file)

if __name__ == '__main__':
    main()

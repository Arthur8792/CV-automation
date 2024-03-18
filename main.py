from langchain_openai import ChatOpenAI
from src.utils import pdf_to_txt, json_to_dico, dico_to_json 
from src.sequential_chains import create_sequential_chain

OPENAI_MODEL = 'gpt-3.5-turbo'
OPENAI_API_KEY = ''

def main():
    """
    Main entry of the pipeline.
    """
    # Get OpenAI API key : 
    with open('api_key.txt','r') as api_key_file:
        OPENAI_API_KEY = api_key_file.readline().strip('\n')
    
    # Files paths
    pdf_file = 'data/cv/Exemple1.pdf'
    prompts_file = 'data/prompts/prompts.json'
    result_file = 'data/results/results.json'

    # Convert from PDF to txt
    cv_content = pdf_to_txt(pdf_file)

    # Create LLM model
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    
    # Extract prompts from JSON file.
    dico_prompts = json_to_dico(prompts_file)
    
    # Create the sequential chain : 
    sequential_chain, output_variables = create_sequential_chain(llm, dico_prompts)

    # Run the pipeline
    result = sequential_chain.invoke({"CV":cv_content})

    # Store results in a dictionary : 
    dico_result = {}
    for output in output_variables:
        if output not in dico_result:
            dico_result[output] = result[output]
    
    # Store the result in a JSON file
    dico_to_json(dico_result, result_file)

if __name__ == '__main__':
    main()

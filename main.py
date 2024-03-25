from langchain_openai import ChatOpenAI
from src.utils import pdf_to_txt, json_to_dico, dico_to_json, fill_docx_template
from src.sequential_chains import create_sequential_chain

OPENAI_MODEL = 'gpt-4-1106-preview'
OPENAI_API_KEY_FILE = 'api_key_4.txt'
OPENAI_API_KEY = ''

def main():
    """
    Main entry of the pipeline.
    """
    # Get OpenAI API key : 
    with open(OPENAI_API_KEY_FILE,'r') as api_key_file:
        OPENAI_API_KEY = api_key_file.readline().strip('\n')
    
    # Files paths
    pdf_file = 'data/cv/ex5.pdf'
    prompts_file = 'data/prompts/prompts.json'
    prompts_experience_file = 'data/prompts/prompts-experiences.json'
    result_file = 'data/results/results.json'
    result_file_experience = 'data/results/results-experiences.json'
    template_file = 'data/templates/Template-CV.docx'
    cv_word = 'data/results/cv-silamir.docx'

    # Convert from PDF to txt
    cv_content = pdf_to_txt(pdf_file)

    # Create LLM model
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    
    # Extract prompts from JSON file.
    dico_prompts = json_to_dico(prompts_file)
    dico_prompts_experiences = json_to_dico(prompts_experience_file)
    
    # Create the sequential chain : 
    sequential_chain, output_variables = create_sequential_chain(llm, dico_prompts)
    sequential_chain_experiences, output_variables_experiences = create_sequential_chain(llm, dico_prompts_experiences)

    # Run the general pipeline
    print("Getting general information about CV...")
    result = sequential_chain.invoke({"CV":cv_content})
    print("Get general information !")

    # Store results in a dictionary : 
    dico_result = {}
    for output in output_variables:
        if output != "noms_clients":
            result[output] = result[output].replace("&", "&amp;")
            dico_result[output] = result[output]

    # Part experiences : 
    clients = result["noms_clients"]
    liste_clients = clients.strip("][").split(', ')

    # For each client,  : 
    print("Getting information for each professional experience...")
    dico_result_experiences = {"experiences":[]}
    # counter_monitoring = 0
    for client_name in liste_clients:
        # counter_monitoring += 1
        # if counter_monitoring>3:
        #     break
        result_experience = sequential_chain_experiences.invoke({"CV":cv_content,"client_name":client_name})
        # Store result in dictionary
        # Store client name
        dico_client = {}
        dico_client["client_name"] = client_name
        for output in output_variables_experiences:
            result_experience[output] = result_experience[output].replace("&", "&amp;")
            dico_client[output] = result_experience[output]
        dico_result_experiences['experiences'].append(dico_client)
    print("Get information about each experience ! ")
    # Store the result in a JSON file
    dico_to_json(dico_result, result_file)
    dico_to_json(dico_result_experiences, result_file_experience)

    # # Fill the Word template
    dico_fill, dico_fill_experience = json_to_dico(result_file), json_to_dico(result_file_experience)
    # Gather in 1 dictionary
    dico_fill["experiences"] = dico_fill_experience["experiences"]
    fill_docx_template(dico_fill, template_file, cv_word)

if __name__ == '__main__':
    main()


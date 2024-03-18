"""
Functions associated to langchain sequential chain.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

def create_sequential_chain(llm: ChatOpenAI, dico_prompts: dict[str, str|list[str]])->tuple[SequentialChain,list[str]]:
    """
    Creates the Langchain sequential chain from the prompts in the dictionary.

    Args:
        dico_prompts : the dictionary which contains the prompts.
    Returns:
        The Langchain Sequential Chain
        The output variables
    """
    # Initialize chains and output variables : 
    chains, output_variables = [],[]

    # For each prompt, create a LLM chain.
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

    return sequential_chain, output_variables
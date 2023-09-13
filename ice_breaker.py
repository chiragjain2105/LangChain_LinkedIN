from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedIn import scrape_linkedin_profile
from Agents.linkedin_lookup_agents import lookup as linkedin_lookup_agent
from output_parsers import person_intel_parser, PersonIntel
from typing import Tuple

def ice_break(name: str) -> Tuple[PersonIntel,str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    summary_template = """
        given the LinkedIN information {information} about a person I want you to create:
        1. a short summary
        2. two interesting facts about them
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # linkedin_data = scrape_linkedin_profile(
    #     linkedin_profile_url="https://www.linkedin.com/in/harrison-chase-961287118/"
    # )

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    result = chain.run(information=linkedin_data)
    print(result)
    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")
    # print(linkedin_data.json())


if __name__ == "__main__":
    print("Hello LangChain!")
    result = ice_break(name="Chirag Jain NITJ'23")
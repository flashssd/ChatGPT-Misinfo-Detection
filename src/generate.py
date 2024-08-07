import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Dict


def generate_responses(
    tweets: pd.DataFrame,
    identity: int,
    iteration: int,
    api_key: str,
    model: str,
    temperature: float,
) -> pd.DataFrame:
    # Define the prompt templates
    template = "You are a helpful assistant."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Initialize the OpenAI model
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=model,
        temperature=temperature,
    )

    if identity == 0:

        human_template = (
            "You will be asked to judge whether a tweet contains misinformation or not. "
            "The definition of misinformation is the following:\n\n"
            "False or inaccurate information, especially that which is deliberately intended to deceive.\n\n"
            "Do you think the following tweet contains misinformation?\n\n"
            "Yes: the tweet contains misinformation\n"
            "No: the tweet does NOT contain misinformation\n\n"
            "Be sure to show your reasoning in this form: ‘Choice:__### Reason:__‘ "
            "(make sure to use ### as the delimiter).\n\n"
            "Tweet begins:\n{tweet}"
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(llm=llm, prompt=chat_prompt)

        output = []
        for tweet in tweets["tweet"]:
            char_dict: Dict[str, str] = {"tweet": tweet}
            for i in range(iteration):
                char_dict[f"response_{i+1}"] = chain.run(char_dict)
            output.append(char_dict)

    elif identity == 1:

        human_template = (
            "Suppose you are {political_belief}.\n\n"
            "So based on your above character description, you will be asked to judge whether "
            "a tweet contains misinformation or not. The definition of misinformation is the following:\n\n"
            "False or inaccurate information, especially that which is deliberately intended to deceive.\n\n"
            "Do you think the following tweet contains misinformation?\n\n"
            "Yes: the tweet contains misinformation\n"
            "No: the tweet does NOT contain misinformation\n\n"
            "Be sure to align your reasoning with your identity description above in this form: ‘Choice:__### Reason:__‘ "
            "(make sure to use ### as the delimiter).\n\n"
            "Tweet begins:\n{tweet}"
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(llm=llm, prompt=chat_prompt)

        # Generate responses
        output = []
        for tweet in tweets["tweet"]:
            for belief in ["Liberal", "Conservative"]:
                char_dict: Dict[str, str] = {"political_belief": belief, "tweet": tweet}
                for i in range(iteration):
                    char_dict[f"response_{i+1}"] = chain.run(char_dict)
                output.append(char_dict)

    else:

        human_template = (
            "Suppose you are a person with {education} degree and live in {place}. "
            "You are {political_belief}, {religion}, and {personality}.\n\n"
            "So based on your above character description, you will be asked to judge whether "
            "a tweet contains misinformation or not. The definition of misinformation is the following:\n\n"
            "False or inaccurate information, especially that which is deliberately intended to deceive.\n\n"
            "Do you think the following tweet contains misinformation?\n\n"
            "Yes: the tweet contains misinformation\n"
            "No: the tweet does NOT contain misinformation\n\n"
            "Be sure to align your reasoning with your identity description above in this form: ‘Choice:__### Reason:__‘ "
            "(make sure to use ### as the delimiter).\n\n"
            "Tweet begins:\n{tweet}"
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(llm=llm, prompt=chat_prompt)

        # Generate responses
        output = []
        for tweet in tweets["tweet"]:
            for edu in ["high school", "undergraduate", "graduate"]:
                for place in ["rural", "urban"]:
                    for personality in ["narcissistic", "empathetic"]:
                        for religion in ["religious", "atheistic"]:
                            for belief in ["Liberal", "Conservative"]:
                                char_dict: Dict[str, str] = {
                                    "education": edu,
                                    "place": place,
                                    "political_belief": belief,
                                    "religion": religion,
                                    "personality": personality,
                                    "tweet": tweet,
                                }
                                for i in range(iteration):
                                    char_dict[f"response_{i+1}"] = chain.run(char_dict)
                                output.append(char_dict)

    df = pd.DataFrame(output)

    return df

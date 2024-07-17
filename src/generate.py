import pandas as pd
import fire
import warnings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from postprocessing import postprocess
from common import excel_write
from analysis import analyze


def main(
    model: str = "gpt-3.5-turbo-0613",
    api_key: str | None = None,
    temperature: float = 0.7,
    iteration: int = 30,
    identity: int = 2,
):
    assert api_key is not None, "API key is not provided."
    assert identity in [0, 1, 2], "invalid identity option"

    if temperature == 0 and iteration > 1:
        warnings.warn(
            (
                "When the temperature is set to 0, there is no variability in the output for "
                "the same prompt. Therefore, the iteration count is set to 1."
            )
        )
        iteration = 1

    # Load the data
    df = pd.read_excel(
        "data/MTurk_Empathy_Data.xlsx", header=None, sheet_name="Tweet_Text_Coding"
    )

    # Select the first two rows.
    texts = df.iloc[0].tolist()[1::5]
    # tags = df.iloc[1].tolist()[1::5]
    # tweets = pd.DataFrame({"tweet": texts, "origin_tag": tags})
    tweets = pd.DataFrame({"tweet": texts})
    tweets["Tweet_ID"] = tweets.index.values + 1

    def tweet_class(tweet_id):
        if tweet_id in [3, 4, 5, 6, 19, 20, 21, 22, 31, 32, 33, 34]:
            return "Misinformation"
        elif tweet_id in [7, 8, 23, 24, 35, 36]:
            return "Correction"
        elif tweet_id in [1, 2, 9, 18, 29, 30]:
            return "Neutral"
        elif tweet_id in [12, 13, 16, 17, 27, 28]:
            return "Unaligned Sentiment"
        else:
            return "Aligned Sentiment"

    tweets["Tweet_classification"] = tweets["Tweet_ID"].apply(tweet_class)

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
        sheet_name = "No_Identity"

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
            char_dict = {"tweet": tweet}
            for i in range(iteration):
                char_dict[f"response_{i+1}"] = chain.run(char_dict)
            output.append(char_dict)

        df = pd.DataFrame(output)

    elif identity == 1:
        sheet_name = "Poli_Only"

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
                char_dict = {"political_belief": belief, "tweet": tweet}
                for i in range(iteration):
                    char_dict[f"response_{i+1}"] = chain.run(char_dict)
                output.append(char_dict)

        df = pd.DataFrame(output)

    else:
        sheet_name = "All_Identities"

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
            for edu in ["high school", "undergraduate degree", "graduate"]:
                for place in ["rural", "urban"]:
                    for personality in ["narcissistic", "empathetic"]:
                        for religion in ["religious", "atheistic"]:
                            for belief in ["Liberal", "Conservative"]:
                                char_dict = {
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

    postprocess(df, identity, iteration)
    df = df.merge(tweets, on="tweet", how="left").drop("Tweet_ID", axis=1)

    output_file = "result/coded_results.xlsx"
    # Check if the file exists
    excel_write(df, output_file, sheet_name)
    analyze(df, iteration, identity)


if __name__ == "__main__":
    fire.Fire(main)

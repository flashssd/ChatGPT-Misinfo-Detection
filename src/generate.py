import pandas as pd
import fire
import warnings
from typing import Dict
from langchain.chat_models import ChatOpenAI 
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def main(
    output_file: str = "result/coded_results.xlsx",
    model: str = "gpt-3.5-turbo",
    api_key: str | None = None,
    temperature: float = 0.7,
    iteration: int = 30,
    identity: int = 2
):
    assert api_key is not None, "API key is not provided."
    assert identity in [0, 1, 2], "invalid identity option"
    
    if temperature == 0 and iteration > 1:
        warnings.warn(("When the temperature is set to 0, there is no variability in the output for "
                        "the same prompt. Therefore, the iteration count is set to 1."))
        iteration = 1
        
    
    # Load the data
    df = pd.read_excel('data/MTurk_Empathy_Data.xlsx', header=None, sheet_name='Tweet_Text_Coding')

    # Select the first two rows.
    texts = df.iloc[0].tolist()[1::5]
    tags = df.iloc[1].tolist()[1::5]
    tweets = pd.DataFrame({'text':texts, 'origin_tag':tags})

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
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=llm, prompt=chat_prompt)
        
        output = []
        for tweet in tweets['text']:
            char_dict = {'tweet': tweet}
            for i in range(iteration):
                char_dict[f'response_{i+1}'] = chain.run(char_dict)
            output.append(char_dict)
                
        df = pd.DataFrame(output)
        
        for i in range(iteration):
            df[f'Choice_{i+1}'] = df[f'response_{i+1}'].apply(lambda x: 1 if 'Choice: Yes' in x else 0)

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
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=llm, prompt=chat_prompt)
        
        # Generate responses
        output = []
        for tweet in tweets['text']:
            for belief in ['Liberal', 'Conservative']: 
                char_dict = {
                    'political_belief': belief,
                    'tweet': tweet
                }
                for i in range(iteration):
                    char_dict[f'response_{i+1}'] = chain.run(char_dict)
                output.append(char_dict)

        df = pd.DataFrame(output)
        
        liberal = ['liberal', 'liberalism', 'liberalist', 'liberalistic', 'liberally', 'liberals']
        conservative = ['conservatism', 'conservative', 'conservatively', 'conservativeness']
        political = liberal + conservative
        
        def mention_poli(text: str) -> int:
            text = text.lower()
            for poli in political:
                if poli in text:
                    return 1
            return 0

        for i in range(iteration):
            df[f'Choice_{i+1}'] = df[f'response_{i+1}'].apply(lambda x: 1 if 'Choice: Yes' in x else 0)
            df[f'poli_presence_{i+1}'] = df[f'response_{i+1}'].apply(mention_poli)
        
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
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=llm, prompt=chat_prompt)

        # Generate responses
        output = []
        for tweet in tweets['text']:
            for edu in ['high school', 'undergraduate degree', 'graduate']: 
                for place in ['rural', 'urban']:
                    for personality in ['narcissistic', 'empathetic']: 
                        for religion in ['religious', 'atheistic']:
                            for belief in ['Liberal', 'Conservative']: 
                                char_dict = {
                                    'education': edu, 
                                    'place': place, 
                                    'political_belief': belief,
                                    'religion': religion, 
                                    'personality': personality, 
                                    'tweet': tweet
                                }
                                for i in range(iteration):
                                    char_dict[f'response_{i+1}'] = chain.run(char_dict)
                                output.append(char_dict)

        df = pd.DataFrame(output)

        #check the presence of the variables
        undergraduate = ['undergrad', 'undergrads', 'undergraduate', 'undergraduated', 
                    'undergraduates', 'undergraduating','undergraduation']

        graduate = ['graduand', 'graduands', 'graduate', 'graduated', 'graduates',
                    'graduating', 'graduation']

        rural = ['rural', 'rurality', 'rurally']

        urban = ['urban', 'urbanely', 'urbanite', 'urbanity', 'urbanization', 'urbanize', 'urbanized']

        liberal = ['liberal', 'liberalism', 'liberalist', 'liberalistic', 'liberally', 'liberals']

        conservative = ['conservatism', 'conservative', 'conservatively', 'conservativeness']

        religious = ['religion', 'religious', 'religiously']

        atheistic = ['atheism', 'atheist', 'atheistic', 'atheistically']

        narcissistic = ['narcissism', 'narcissistic', 'narcissistically']

        empathetic = ['empath', 'empathetic', 'empathetically', 'empathise', 'empathised',
                    'empathising', 'empathize', 'empathized','empathizing']


        education = undergraduate + graduate + ['high school']
        place = rural + urban
        political = liberal + conservative
        religion = religious + atheistic
        personal = narcissistic + empathetic

        def mention_variable(text: str) -> Dict[str, int]:
            text = text.lower()
            result_dict = {key: 0 for key in ['Edu', 'Place', 'Political', 'Relig', 'Personal']}

            for edu in education:
                if edu in text:
                    result_dict['Edu'] = 1
                    break
            for pl in place:
                if pl in text:
                    result_dict['Place'] = 1
                    break
            for poli in political:
                if poli in text:
                    result_dict['Political'] = 1
                    break
            for relig in religion:
                if relig in text:
                    result_dict['Relig'] = 1
                    break
            for person in personal:
                if person in text:
                    result_dict['Personal'] = 1
                    break

            return result_dict

        for i in range(iteration):
            df[f'Choice_{i+1}'] = df[f'response_{i+1}'].apply(lambda x: 1 if 'Choice: Yes' in x else 0)
            df[f'variable_presence_{i+1}'] = df[f'response_{i+1}'].apply(mention_variable)

    df.to_excel(output_file, index=None)

if __name__ == "__main__":
    fire.Fire(main)

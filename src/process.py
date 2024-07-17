from typing import Dict
import pandas as pd


def preprocess() -> pd.DataFrame:
    # Load the data
    df = pd.read_excel(
        "data/MTurk_Empathy_Data.xlsx", header=None, sheet_name="Tweet_Text_Coding"
    )

    # Select the first two rows.
    texts = df.iloc[0].tolist()[1::5]
    tweets = pd.DataFrame({"tweet": texts})
    tweets["Tweet_ID"] = tweets.index.values + 1

    def tweet_class(tweet_id: int) -> str:
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

    return tweets


def postprocess(
    df: pd.DataFrame, identity: int, iteration: int, tweets: pd.DataFrame
) -> pd.DataFrame:
    if identity == 0:
        for i in range(iteration):
            df[f"Choice_{i+1}"] = df[f"response_{i+1}"].apply(
                lambda x: 1 if "Choice: Yes" in x else 0
            )

    elif identity == 1:
        liberal = [
            "liberal",
            "liberalism",
            "liberalist",
            "liberalistic",
            "liberally",
            "liberals",
        ]
        conservative = [
            "conservatism",
            "conservative",
            "conservatively",
            "conservativeness",
        ]
        political = liberal + conservative

        def mention_poli(text: str) -> int:
            text = text.lower()
            for poli in political:
                if poli in text:
                    return 1
            return 0

        for i in range(iteration):
            df[f"Choice_{i+1}"] = df[f"response_{i+1}"].apply(
                lambda x: 1 if "Choice: Yes" in x else 0
            )
            df[f"poli_presence_{i+1}"] = df[f"response_{i+1}"].apply(mention_poli)

    else:
        # check the presence of the variables
        undergraduate = [
            "undergrad",
            "undergrads",
            "undergraduate",
            "undergraduated",
            "undergraduates",
            "undergraduating",
            "undergraduation",
        ]

        graduate = [
            "graduand",
            "graduands",
            "graduate",
            "graduated",
            "graduates",
            "graduating",
            "graduation",
        ]

        rural = ["rural", "rurality", "rurally"]

        urban = [
            "urban",
            "urbanely",
            "urbanite",
            "urbanity",
            "urbanization",
            "urbanize",
            "urbanized",
        ]

        liberal = [
            "liberal",
            "liberalism",
            "liberalist",
            "liberalistic",
            "liberally",
            "liberals",
        ]

        conservative = [
            "conservatism",
            "conservative",
            "conservatively",
            "conservativeness",
        ]

        religious = ["religion", "religious", "religiously"]

        atheistic = ["atheism", "atheist", "atheistic", "atheistically"]

        narcissistic = ["narcissism", "narcissistic", "narcissistically"]

        empathetic = [
            "empath",
            "empathetic",
            "empathetically",
            "empathise",
            "empathised",
            "empathising",
            "empathize",
            "empathized",
            "empathizing",
        ]

        education = undergraduate + graduate + ["high school"]
        place = rural + urban
        political = liberal + conservative
        religion = religious + atheistic
        personal = narcissistic + empathetic

        def mention_variable(text: str) -> Dict[str, int]:
            text = text.lower()
            result_dict = {
                key: 0 for key in ["Edu", "Place", "Political", "Relig", "Personal"]
            }

            for edu in education:
                if edu in text:
                    result_dict["Edu"] = 1
                    break
            for pl in place:
                if pl in text:
                    result_dict["Place"] = 1
                    break
            for poli in political:
                if poli in text:
                    result_dict["Political"] = 1
                    break
            for relig in religion:
                if relig in text:
                    result_dict["Relig"] = 1
                    break
            for person in personal:
                if person in text:
                    result_dict["Personal"] = 1
                    break

            return result_dict

        for i in range(iteration):
            df[f"Choice_{i+1}"] = df[f"response_{i+1}"].apply(
                lambda x: 1 if "Choice: Yes" in x else 0
            )
            df[f"variable_presence_{i+1}"] = df[f"response_{i+1}"].apply(
                mention_variable
            )

    df = df.merge(tweets, on="tweet", how="left").drop("Tweet_ID", axis=1)
    return df

from typing import Dict


def postprocess(df, identity, iteration):
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

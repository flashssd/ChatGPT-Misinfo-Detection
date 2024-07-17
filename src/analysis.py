import pandas as pd
from common import excel_write


def analyze(df, iteration, identity):
    df["TotalChoice"] = 0
    for i in range(iteration):
        df["TotalChoice"] += df[f"Choice_{i+1}"]

    df["TotalChoice"] = df["TotalChoice"] / iteration

    # no identity
    if identity == 0:
        output_file = "result/analysis_no_identity.xlsx"
        weights = df[["tweet", "TotalChoice"]]
        excel_write(weights, output_file, "weights")

    # poli only
    elif identity == 1:
        output_file = "result/analysis_poli_only.xlsx"
        weights = df.groupby("tweet")["TotalChoice"].agg(["std", "mean"]).reset_index()
        excel_write(weights, output_file, "weights")

        weights_tweet_class = (
            df.groupby("Tweet_classification")["TotalChoice"]
            .agg(["std", "mean"])
            .reset_index()
        )
        excel_write(weights_tweet_class, output_file, "weights_tweet_class")

        weights_poli = (
            df.groupby("political_belief")["TotalChoice"]
            .agg(["std", "mean"])
            .reset_index()
        )
        excel_write(weights_poli, output_file, "weights_poli")

    # all identities
    else:
        output_file = "result/analysis_all_identities.xlsx"
        identities = [
            "education",
            "place",
            "political_belief",
            "religion",
            "personality",
        ]
        for identity in identities:
            globals()[f"weights_{identity}"] = (
                df.groupby(["tweet", identity])["TotalChoice"]
                .agg(["std", "mean"])
                .reset_index()
            )
            excel_write(
                globals()[f"weights_{identity}"], output_file, f"weights_{identity}"
            )

            globals()[f"weights_{identity}_class"] = (
                df.groupby(["tweet", "Tweet_classification"])["TotalChoice"]
                .agg(["std", "mean"])
                .reset_index()
            )
            excel_write(
                globals()[f"weights_{identity}_class"],
                output_file,
                f"weights_{identity}_class",
            )

        identities_abbrev = ["Edu", "Place", "Political", "Relig", "Personal"]
        for identity in identities_abbrev:
            df[f"Total_{identity}"] = 0
            for i in range(iteration):
                df[f"Total_{identity}"] += df[f"variable_presence_{i+1}"].apply(
                    lambda x: eval(x)[identity]
                )
            df[f"Total_{identity}"] /= iteration

        df = df[
            ["tweet", "Tweet_classification"]
            + identities
            + [f"Total_{identity}" for identity in identities_abbrev]
        ]
        Mention_Class = (
            df.groupby("Tweet_classification")[
                [f"Total_{identity}" for identity in identities_abbrev]
            ]
            .mean()
            .reset_index()
        )
        excel_write(Mention_Class, output_file, "Mention_Class")

        for i in range(5):
            globals()[f"{identities_abbrev[i]}_by_class"] = (
                df.groupby(identities[i])[f"Total_{identities_abbrev[i]}"]
                .mean()
                .reset_index()
            )

            excel_write(
                globals()[f"{identities_abbrev[i]}_by_class"],
                output_file,
                f"{identities_abbrev[i]}_by_class",
            )

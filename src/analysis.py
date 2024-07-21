import pandas as pd
from common import excel_write


def analyze(df: pd.DataFrame, iteration: int, identity: int) -> None:
    df["TotalChoice"] = 0
    for i in range(iteration):
        df["TotalChoice"] += df[f"Choice_{i+1}"]

    df["TotalChoice"] = df["TotalChoice"] / iteration

    # no identity
    if identity == 0:
        output_file = "result/analysis_no_identity.xlsx"
        # average yes by tweets
        weights = df[["tweet", "TotalChoice"]]
        excel_write(weights, output_file, "weights")
        # average yes by tweet class
        weights_tweet_class = (
            df.groupby("Tweet_classification")["TotalChoice"]
            .agg(["std", "mean"])
            .reset_index()
        )
        excel_write(weights_tweet_class, output_file, "weights_tweet_class")

    # poli only
    elif identity == 1:
        output_file = "result/analysis_poli_only.xlsx"
        # average yes by tweets
        weights = df.groupby("tweet")["TotalChoice"].agg(["std", "mean"]).reset_index()
        excel_write(weights, output_file, "weights")
        # average yes by tweet class
        weights_tweet_class = (
            df.groupby("Tweet_classification")["TotalChoice"]
            .agg(["std", "mean"])
            .reset_index()
        )
        excel_write(weights_tweet_class, output_file, "weights_tweet_class")
        # average yes by poli belief
        weights_poli = (
            df.groupby("political_belief")["TotalChoice"]
            .agg(["std", "mean"])
            .reset_index()
        )
        excel_write(weights_poli, output_file, "weights_poli")

    # all identities
    else:
        output_file = "result/analysis_all_identities.xlsx"

        # average yes by tweets
        weights = df.groupby("tweet")["TotalChoice"].agg(["std", "mean"]).reset_index()
        excel_write(weights, output_file, "weights")
        # average yes by tweet class
        weights_tweet_class = (
            df.groupby("Tweet_classification")["TotalChoice"]
            .agg(["std", "mean"])
            .reset_index()
        )
        excel_write(weights_tweet_class, output_file, "weights_tweet_class")

        identities = [
            "education",
            "place",
            "political_belief",
            "religion",
            "personality",
        ]
        for ident in identities:
            # average yes by tweet and identity
            globals()[f"weights_{ident}_tweet"] = (
                df.groupby(["tweet", ident])["TotalChoice"]
                .agg(["std", "mean"])
                .reset_index()
            )
            excel_write(
                globals()[f"weights_{ident}_tweet"],
                output_file,
                f"weights_{ident}_tweet",
            )

            # average yes by tweet class and identity
            globals()[f"weights_{ident}_class"] = (
                df.groupby([ident, "Tweet_classification"])["TotalChoice"]
                .agg(["std", "mean"])
                .reset_index()
            )
            excel_write(
                globals()[f"weights_{ident}_class"],
                output_file,
                f"weights_{ident}_class",
            )

        identities_abbrev = ["Edu", "Place", "Political", "Relig", "Personal"]
        for ident_abbr in identities_abbrev:
            df[f"Total_{ident_abbr}"] = 0
            for i in range(iteration):
                df[f"Total_{ident_abbr}"] += df[f"variable_presence_{i+1}"].apply(
                    lambda x: eval(x)[ident_abbr] if type(x) == str else x[ident_abbr]
                )
            df[f"Total_{ident_abbr}"] /= iteration

        df = df[
            ["tweet", "Tweet_classification"]
            + identities
            + [f"Total_{ident_abbr}" for ident_abbr in identities_abbrev]
        ]
        # average identity mention by tweet class
        Mention_Class = (
            df.groupby("Tweet_classification")[
                [f"Total_{ident_abbr}" for ident_abbr in identities_abbrev]
            ]
            .mean()
            .reset_index()
        )
        excel_write(Mention_Class, output_file, "Mention_Class")

        for i in range(5):
            # average identity mention by identity categories
            globals()[f"{identities_abbrev[i]}_Mention"] = (
                df.groupby(identities[i])[f"Total_{identities_abbrev[i]}"]
                .mean()
                .reset_index()
            )

            excel_write(
                globals()[f"{identities_abbrev[i]}_Mention"],
                output_file,
                f"{identities_abbrev[i]}_Mention",
            )


if __name__ == "__main__":
    df = pd.read_excel("result/coded_results.xlsx", sheet_name="No_Identity")
    iteration = 30
    identity = 0
    analyze(df, iteration, identity)

    df = pd.read_excel("result/coded_results.xlsx", sheet_name="Poli_Only")
    iteration = 30
    identity = 1
    analyze(df, iteration, identity)

    df = pd.read_excel("result/coded_results.xlsx", sheet_name="All_Identities")
    iteration = 30
    identity = 2
    analyze(df, iteration, identity)

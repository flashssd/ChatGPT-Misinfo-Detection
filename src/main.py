import fire
import warnings
import pandas as pd
from process import postprocess
from generate import generate_responses
from common import excel_write
from analysis import analyze


def main(
    model: str = "gpt-3.5-turbo-0613",
    api_key: str | None = None,
    temperature: float = 0.7,
    iteration: int = 30,
    identity: int = 2,
) -> None:
    assert api_key is not None, "API key is not provided."
    assert identity in [0, 1, 2], "Invalid identity option"

    if temperature == 0 and iteration > 1:
        warnings.warn(
            (
                "When the temperature is set to 0, there is no variability in the output for "
                "the same prompt. Therefore, the iteration count is set to 1."
            )
        )
        iteration = 1

    tweets = pd.read_excel("data/tweets_36.xlsx")
    df = generate_responses(tweets, identity, iteration, api_key, model, temperature)
    df = postprocess(df, identity, iteration, tweets)

    output_file = "result/coded_results.xlsx"

    if identity == 0:
        sheet_name = "No_Identity"
    elif identity == 1:
        sheet_name = "Poli_Only"
    else:
        sheet_name = "All_Identities"

    excel_write(df, output_file, sheet_name)

    analyze(df, iteration, identity)


if __name__ == "__main__":
    fire.Fire(main)

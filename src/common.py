import pandas as pd


def excel_write(df: pd.DataFrame, output_file: str, sheet_name: str) -> None:
    # Check if the file exists
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a") as writer:
            if sheet_name in writer.book.sheetnames:
                del writer.book[sheet_name]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

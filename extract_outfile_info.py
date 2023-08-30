import asyncio
import asyncssh
from shh_keys import keys  # sensitive and are personal to user
import traceback
import re
import pandas as pd
from io import StringIO


def extract_info_from_filename(filename):
    # Split the filename from its extension
    name, _ = filename.rsplit(".", 1)

    # Split by underscores to get each variable-value pair
    parts = name.split("_")

    # Use regex to capture pairs of [variable]-[value]
    pattern = re.compile(r"(\w+)-([\w.]+)")

    info_dict = {}
    for part in parts:
        match = pattern.match(part)
        if match:
            key, value = match.groups()
            info_dict[key] = value

    return pd.DataFrame([info_dict])


async def fetch_csv_data_from_all_dirs_via_ssh(
    hostname, username, password, base_directory_path, pattern
):
    async with asyncssh.connect(hostname, username=username, password=password) as conn:
        dir_result = await conn.run(f"find {base_directory_path} -type d")

        if dir_result.stderr:
            print(dir_result.stderr)
            return []

        directories = dir_result.stdout.splitlines()
        print(f"{len(directories)} directories found in base directory.")

        all_dataframes = []
        for directory in directories:
            print(f"Found: {directory}")
            cmd_result = await conn.run(f"ls {directory}/{pattern}")

            if cmd_result.stderr:
                continue

            filenames = cmd_result.stdout.splitlines()
            print(f"{len(filenames)} files found in directory.")

            for file in filenames:
                file_content = await conn.run(f"cat {file}")
                if not file_content.stderr:
                    df = pd.read_csv(StringIO(file_content.stdout))
                    id_df = extract_info_from_filename(file)

                    # Extract the folder from the full path
                    folder = file.split("/")[-2]
                    id_df["dataset"] = folder[:7]

                    all_dataframes.append(pd.concat([id_df, df], axis=1))
                else:
                    print(f"Error reading {file}: {file_content.stderr}")

        return all_dataframes


# Usage
pattern = "table_of_metrics*.csv"

try:
    dataframes = asyncio.run(
        fetch_csv_data_from_all_dirs_via_ssh(
            keys["hostname"],
            keys["username"],
            keys["password"],
            keys["basepath"],
            pattern,
        )
    )
    id_cols = ["dataset", "topic", "nsamples", "margin", "temp", "dr", "dl"]
    df_out = pd.concat(dataframes, axis=0).sort_values(by=id_cols)
    print(df_out.columns)
    print(df_out.shape)

    df_out.to_csv(
        keys["savepath"] + "experimental_results.csv",
        index=False,
    )

except Exception as e:
    print("An error occurred:", str(e))
    traceback.print_exc()

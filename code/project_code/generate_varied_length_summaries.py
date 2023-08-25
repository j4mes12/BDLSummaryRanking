import lorem
import numpy as np
import os


def main():
    mini, maxi, step = 200, 800, 25
    output_file = os.path.join("project_code", "data", "summary_candidates")
    summaries = []

    for n_words in np.arange(mini, maxi, step):
        summary = ""
        while len(summary) < n_words:
            summary += " " + lorem.sentence()

        summaries.append(summary)

    file_name = f"lorem_summaries_min-{mini}_max-{maxi}_step-{step}.csv"

    np.savetxt(
        os.path.join(output_file, file_name),
        summaries,
        fmt="%s",
        delimiter="#####",
    )


if __name__ == "__main__":
    main()

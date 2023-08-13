from summariser.vector.vector_generator import Vectoriser
from summariser.utils.corpus_reader import CorpusReader
import resources as res
from summariser.utils.writer import append_to_file
import os


def writeSample(actions, reward, path):
    if "heuristic" in path:
        str = "\nactions:"
        for act in actions:
            str += repr(act) + ","
        str = str[:-1]
        str += "\nutility:" + repr(reward)
        append_to_file(str, path)
    else:
        assert "rouge" in path
        str = "\n"
        for j, model_name in enumerate(reward):
            str += "\nmodel {}:{}".format(j, model_name)
            str += "\nactions:"
            for act in actions:
                str += repr(act) + ","
            str = str[:-1]
            str += (
                "\n"
                + f"R1:{reward[model_name][0]};"
                + f"R2:{reward[model_name][1]};"
                + f"R3:{reward[model_name][2]};"
                + f"R4:{reward[model_name][3]};"
                + f"RL:{reward[model_name][4]};"
                + f"RSU:{reward[model_name][5]}"
            )

        append_to_file(str, path)


if __name__ == "__main__":
    dataset = "DUC2001"
    language = "english"
    summary_len = 100

    summary_num = 10100
    base_dir = os.path.join(res.SUMMARY_DB_DIR, dataset)

    reader = CorpusReader(res.PROCESSED_PATH)
    data = reader.get_data(dataset, summary_len)

    topic_count = 0
    start = 0
    end = 100
    print(f"dataset: {dataset}; start: {start}; end: {end - 1}")

    for topic, docs, models in data:
        topic_count += 1
        dir_path = os.path.join(base_dir, topic)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        vec = Vectoriser(docs, summary_len)
        if start <= topic_count < end:
            print(f"-----Generate samples for topic {topic_count}: {topic}-----")
            (
                action_list,
                heuristic_rewards,
                rouge_rewards,
            ) = vec.sampleRandomReviews(summary_num, True, True, models)

            assert len(action_list) == len(heuristic_rewards) == len(rouge_rewards)

            for action_index in range(len(action_list)):
                writeSample(
                    action_list[action_index],
                    heuristic_rewards[action_index],
                    os.path.join(dir_path, "heuristic"),
                )
                writeSample(
                    action_list[action_index],
                    rouge_rewards[action_index],
                    os.path.join(dir_path, "rouge"),
                )

    print(
        f"dataset {dataset}; "
        + f"total topic num: {topic_count}; "
        + f"start: {start}; "
        + f"end: {end - 1}"
    )

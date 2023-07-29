import csv
import codecs
import os
import resources as res
from summariser.utils.misc import normaliseList


def read_csv(filename):
    with open(filename, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        rows = [row for row in reader]
        return rows


def read_file(filename):
    with codecs.open(filename, "r", "utf-8", errors="ignore") as fp:
        return fp.read()


def readSampleSummaries(dataset, topic, feature_type):
    summaries, ref_values = readSummaries(dataset, topic, "rouge")
    if feature_type == "april":
        summaries, heu_values = readSummaries(dataset, topic, "heuristic")
    else:
        summaries, heu_values = readSummaries(dataset, topic, "supert")

    return summaries, ref_values, heu_values


def readSummaries(dataset, topic, reward_type="rouge"):
    path = os.path.join(res.SUMMARY_DB_DIR, dataset, topic, reward_type)
    summary_list = []
    reward_value_list = []
    model_names = []
    sample_num = 9999

    with open(path, "r") as ff:
        if reward_type == "heuristic" or reward_type == "supert":
            count = 0
            while count < sample_num:
                line = ff.readline()
                if line == "":
                    break
                if "actions" not in line:
                    continue
                else:
                    count += 1
                    actions_str = line.split(":")[1]
                    actions = [int(act) for act in actions_str.split(",")]
                    summary_list.append(actions)
                    reward_value = float(ff.readline().split(":")[1])
                    reward_value_list.append(reward_value)
        elif reward_type == "rouge":
            flag = False
            reward_value = []
            idx = -1
            count = 0
            while count < sample_num:
                line = ff.readline()
                if line == "":
                    break
                if "model" not in line and not flag:
                    continue
                elif "model" in line:
                    idx = int(line.split(":")[0].split(" ")[1])
                    name = line.split(":")[1].strip()
                    if name not in model_names:
                        model_names.append(name)
                    flag = True
                elif "model" not in line and flag:
                    if "action" in line and idx == 0:
                        actions_str = line.split(":")[1].split(",")
                        actions = []
                        for act in actions_str:
                            actions.append(int(act))
                        summary_list.append(actions)
                    elif "R1" in line:
                        scores = line.split(";")
                        R1 = float(scores[0].split(":")[1])
                        R2 = float(scores[1].split(":")[1])
                        # R3 = float(scores[2].split(':')[1])
                        # R4 = float(scores[3].split(':')[1])
                        # RL = float(scores[4].split(':')[1])
                        RSU = float(scores[5].split(":")[1])
                        vv = (
                            R1 / 0.47 + R2 / 0.22 + RSU / 0.18
                        )  # (R1/0.48 + R2/0.212 + RSU/0.195)  #
                        # used the commented out values
                        reward_value.append(vv)
                    elif "action" not in line and "R1" not in line:
                        flag = False
                        reward_value_list.append(reward_value)
                        reward_value = []
                        count += 1

    # normalise
    norm_value_dic = {}
    if reward_type == "heuristic" or reward_type == "supert":
        norm_value_dic = normaliseList(reward_value_list)
    else:
        assert reward_type == "rouge"
        for ii in range(len(reward_value_list[0])):
            temp_list = [x[ii] for x in reward_value_list]
            norm_value_dic[model_names[ii]] = normaliseList(temp_list)

    return summary_list, norm_value_dic

import nibable
import os
import numpy as np


def file_path(data_dir, input_identifier="image", ground_truth_identifier="label", random=True):
    input_list = [f for f in sorted(os.listdir(data_dir)) if input_identifier in f]
    ground_truth_list = [f for f in sorted(os.listdir(data_dir)) if ground_truth_identifier in f]

    if len(input_list) != len(ground_truth_list):
        raise RuntimeError("Directory {} contains {} i!nput items,but {} ground truth item".format(data_dir, len(input_list), len(ground_truth_list)))
    combined_list = zip(input_list,ground_truth_list)
    if random:
        permutation = np.random.permutation(len(combined_list))
    else:
        permutation = np.asarray(range(len(combined_list)))

    for i in permutation:
        yield(combined_list[i])


import pandas as pd
import re
import os, sys
from skimage import io, util
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


path_to_data = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\generated_problems\csv_dir\info_28_03_2020_00_21_34.csv"
data_gen_path = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator"

def switch_path_to_win(path):
    img_relative_path = re.match("[\s\S]+Data_generator\/([\s\S]+)", path)[1]
    problem_image_path = os.path.join(data_gen_path, img_relative_path)
    return problem_image_path


if __name__ == '__main__':
    df = pd.read_csv(path_to_data)
    df['image'] = df['image'].apply(switch_path_to_win)
    images_list_1 = []
    for idx, row in df[df['from_id'] == 2].iterrows():
        print(row['plan length'])
        print(f"reading image: {row['image']}")
        image = io.imread(row['image'])
        image = util.invert(image)
        image = np.asarray(image)
        image = image / 255.0  # Normalize the data =
        images_list_1.append(image)

    images_list_2 = []
    for idx, row in df[df['from_id'] == 1].iterrows():
        print(row['plan length'])
        print(f"reading image: {row['image']}")
        image = io.imread(row['image'])
        image = util.invert(image)
        image = np.asarray(image)
        image = image / 255.0  # Normalize the data =
        images_list_2.append(image)

    diff = images_list_1[0] - images_list_2[9]
    print("Is there difference: {}".format(diff.any()))

    # for idx in range(0, len(images_list) - 1):
    #     diff_mat = images_list[idx] - images_list[idx+1]
    #     print(f"max val: {np.max(diff_mat)}")
    #     print(f"Is there a difference: {diff_mat.any()}")



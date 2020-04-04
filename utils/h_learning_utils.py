from skimage import io, util
import pandas as pd
import os
import re
import torch


def test__best(exp_dict, test_loader):
    """ TEST BEST MODEL """
    print(f'\nBest Model Test Initiated.')
    net = torch.load(exp_dict['path_to_model'])
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            for i, idx in enumerate(torch.argmax(outputs, dim=1)):
                total += 1
                correct += int(labels[i, idx])
        acc = correct / total
    print(f'\nBest Model Accuracy: %{round(acc * 100)}.')
    print(' ')



if __name__ == '__main__':
    data_gen_path = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator"
    data_path = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\generated_problems\csv_dir\info_28_03_2020_00_21_34.csv"
    df = pd.read_csv(data_path)

    df = df[df['plan length'] == 5]
    data_arr_list = []
    for index, row in df.iterrows():
        img_relative_path = re.match("[\s\S]+Data_generator\/([\s\S]+)", row['image'])[1]
        problem_image_path = os.path.join(data_gen_path, img_relative_path)
        image = io.imread(problem_image_path)
        data_arr_list.append(image)

    a = 3


























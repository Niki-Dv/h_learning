import pandas as pd


TRAIN_CSV_PATH = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\Final_data\Rovers_27755_balanced.csv"

if __name__ == '__main__':
    df = pd.read_csv(TRAIN_CSV_PATH)
    balanced_df = pd.DataFrame()
    for length, length_group in df.groupby(['plan length']):
        num_samples = length_group.shape[0]
        to_choose = num_samples if num_samples < 350 else 350
        balanced_df = pd.concat([balanced_df, length_group.sample(n=to_choose)])

    balanced_df.to_csv(TRAIN_CSV_PATH, index=False)








import pandas as pd

FF_COL = '--search "astar(ff)"'
NET_COL = "Net Heur"
ACTUALL_COL = 'plan length'


if __name__ == '__main__':
    csv_path = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\generated_problems\test\csv_dir\Heur_info_06_06_2020_16_03_57.csv"
    df = pd.read_csv(csv_path)
    diff_star_vs_ff = df[ACTUALL_COL] - df[FF_COL]
    diff_star_vs_net = df[ACTUALL_COL] - df[NET_COL]
    print(f"Average distance for ff: {diff_star_vs_ff.abs().mean()}")
    print(f"Average distance for net: {diff_star_vs_net.abs().mean()}")

import pandas as pd

FF_COL = '--search "astar(ff)"'
NET_COL = "Net Heur"
ACTUALL_COL = 'plan length'


if __name__ == '__main__':
    csv_path = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\Final_data\Heurstic_final\test_heur_Rovers_2_27755.csv"
    df = pd.read_csv(csv_path)
    heurstic_to_compare = ['--search "astar(add)"', '--search "astar(blind)"', '--search "astar(lmcut)"', '--search "astar(cegar)"', '--search "astar(ff)"']
    for heurstic in heurstic_to_compare:
        curr_df = df.copy()
        curr_df = curr_df[curr_df[heurstic] != -1]
        diff_star_vs_ff = df[ACTUALL_COL] - df[heurstic]
        print(f"Average distance for {heurstic}: {diff_star_vs_ff.abs().mean()}")

    diff_star_vs_net = df[ACTUALL_COL] - df[NET_COL]
    print(f"Average distance for net: {diff_star_vs_net.abs().mean()}")
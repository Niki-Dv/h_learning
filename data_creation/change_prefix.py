#!/usr/bin/env python3

import argparse
import pandas as pd
import os, sys

COLUMNS_TO_CHANGE =  ['problem', 'plan', 'table']
windows_pre = r"C:/Users/NikiDavarashvili/OneDrive - Technion/Desktop/Project/"
ubunto_pre = r"/media/sf_Project/"
###############################################################################################
def change_prefix_ubu_2_windows(csv_path):
    df = pd.read_csv(csv_path)

    def change_pre(path):
        after_pre = path.partition(ubunto_pre)[-1]
        if after_pre == '':
            print("Bad Name!!!")
            return path

        return os.path.join(windows_pre, after_pre)

    for column in COLUMNS_TO_CHANGE:
        df[column] = df[column].apply(change_pre)

    df.to_csv(csv_path)

###############################################################################################
def change_prefix_windows_2_ubu(csv_path):
    df = pd.read_csv(csv_path)
    def change_pre(path):
        after_pre = path.partition(windows_pre)[-1]
        if after_pre == '':
            print("Bad Name!!!")
            return path

        return os.path.join(ubunto_pre, after_pre)

    for column in COLUMNS_TO_CHANGE:
        df[column] = df[column].apply(change_pre)

    df.to_csv(csv_path)


###############################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Changes prefix from windows to ubunto or '
                                                 f'the opposite:\n give 1 for window to ubunto '
                                                 f'and 2 for ubunto to windows')
    parser.add_argument('-m', '--mode', required=True, help='type of change')
    parser.add_argument('-p', '--path', required=True, help='path to csv')
    args = parser.parse_args()
    if args.mode == "2":
        change_prefix_ubu_2_windows(args.path)
    elif args.mode == "1":
        change_prefix_windows_2_ubu(args.path)
    else:
        print(f"Bad mode was given: {args.mode}. has to be 1 or 2")
        sys.exit(-1)

    print("Done changing prefix")


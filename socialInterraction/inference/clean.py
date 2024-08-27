import os 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d

PRE_POST = 'post'
DATAPATH='' # ADD THE GLOBAL DATA PATH HERE


def find_center(x, y, w, h):
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def clean(df):
    leng = len(df)
    # print(leng)

    # convert to 0
    threshold_y = 1080-214
    indices = list(df.index[df['y']>threshold_y])
    if len(indices)>0:
        # print(indices)
        df.loc[indices,:]=0

    for j in range(2):
        for i in range (1, leng-2,1):
            if df.iloc[i]['x']==0 or df.iloc[i+1]['x']==0:
                continue
            dist = calculate_distance(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i+1]['x'], df.iloc[i+1]['y'])
            if dist >100:
                df.at[i+1,'x'] = 0
                df.at[i+1,'y'] = 0


    return df

if __name__ == '__main__':
    csv_path = f'{DATAPATH}/{PRE_POST}_intervention/tracked'
    save_csv_path = f'{DATAPATH}/{PRE_POST}_intervention/clean_tracked_centres'
    os.makedirs(save_csv_path, exist_ok=True)
    plot_save_path = f'{DATAPATH}/{PRE_POST}_intervention/path_vis'
    os.makedirs(plot_save_path, exist_ok=True)
    frame_rate = 24
    width = 1920
    height=1080
    delta_remove = 214

    for fish_name in os.listdir(csv_path):
        if fish_name == '.DS_Store':
            continue
        print(fish_name)

        fish_data = pd.read_csv(os.path.join(csv_path,fish_name))
        fish_data = fish_data.dropna(axis=0)
        num_frames = len(fish_data['frame'])
        time = num_frames/frame_rate
        # print(num_frames)
        # print(time)
        
        
        df=pd.DataFrame()
        df['cent'] = fish_data.apply(lambda row: (row['x'] + row['w'] / 2, row['y'] + row['h'] / 2), axis=1)
        df[['x', 'y']] = pd.DataFrame(df['cent'].tolist(), index=df.index)
        df = df.drop(columns=['cent'])

        fish_data['x'] = df['x']
        fish_data['y'] = df['y']
        fish_data = fish_data.drop(columns=['w','h'])

        fish_data = clean(fish_data)

        fish_data['x'] = fish_data['x'].replace(0, pd.NA).ffill()
        fish_data['y'] = fish_data['y'].replace(0, pd.NA).ffill()
        indices = list(fish_data.index[fish_data['y']==0])
        print(indices)

        fish_data['x'] = ((width-fish_data['x'])/width)*12
        fish_data['y'] = ((height-fish_data['y']) - delta_remove)
        fish_data['y'] = ((fish_data['y'])/(fish_data['y'].max()-fish_data['y'].min()))*3.9

        


        fish_data.to_csv(f'{save_csv_path}/{fish_name.split(".")[0]}.csv', index=False)


        X = fish_data['x']
        Y = fish_data['y']+0.05

        sigma = 1.5

        X = gaussian_filter1d(X, sigma)
        Y = gaussian_filter1d(Y, sigma)


        name_for_plot = "R-"+fish_name.split('-')[0].split('r')[1].title()
        side = fish_name.split('-')[1].split('.')[0]

        fig, ax1 = plt.subplots(figsize=(13,5))

        plt.plot(X,Y, '#3C486B')
        plt.title(f"Fish {name_for_plot} at the {side.title()}", fontsize=16, fontweight='bold')

        ax1.set_xlabel(f'Chamber wall length-wise (in cm)', fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'Chamber wall width-wise (in cm)', fontsize=14, fontweight='bold')

        plt.axvline(4, color='red', linestyle = "--", linewidth=0.75)
        plt.axvline(8, color='red', linestyle = "--", linewidth=0.75)

        plt.xlim(0,12)
        plt.ylim(0,4.1)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{plot_save_path}/{fish_name.split(".")[0]}.png', dpi=300)
        # break
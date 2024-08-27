import os 
import pandas as pd
import matplotlib.pyplot as plt
import math
import tqdm

FRAMERATE = 24
DATAPATH = "" # ADD THE GLOBAL DATA PATH HERE

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def distance(df):
    leng = len(df)
    num_frames = leng
    time = num_frames/FRAMERATE
    # print(num_frames)
    # print(time)
    # print(df)

    dist_df = pd.DataFrame(columns=['frame', 'dist'])
    dist_df['frame'] = df['frame']
    dist_df['dist'] = 0
    for i in range(1, leng, 1):
        if df.iloc[i]['x']==0 or df.iloc[i-1]['x']==0:
            continue
        dist = calculate_distance(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i-1]['x'], df.iloc[i-1]['y'])
        dist_df.at[i,'dist'] = round(dist,3)
    return dist_df

def zone(df, shoal):
    leng = len(df)
    num_frames = leng
    time = num_frames/FRAMERATE
    # print(num_frames)
    # print(time)
    # print(df)
    zone_df = pd.DataFrame(columns=['frame', 'zone'])
    zone_df['frame'] = df['frame']
    zone_df['zone'] = 0
    for i in range(0, leng, 1):
        if df.iloc[i]['x'] >=0 and df.iloc[i]['x'] <=4:
            zone_df.at[i,'zone'] = 1
        elif df.iloc[i]['x'] >4 and df.iloc[i]['x'] <=8:
            zone_df.at[i,'zone'] = 2
        elif df.iloc[i]['x'] >8 and df.iloc[i]['x'] <=12:
            zone_df.at[i,'zone'] = 3
    
    # mapping zone to shoaal side
    if shoal == "left":
        zone_df['zone'] = zone_df['zone'].map({1: "Shoal", 2: "Middle", 3: "Control"})
    else:
        zone_df['zone'] = zone_df['zone'].map({1: "Control", 2: "Middle", 3: "Shoal"})
    summary = zone_df.groupby('zone').count()
    perc_summary = round((summary/num_frames),3)
    summary = round(summary/FRAMERATE,3)

    df = pd.DataFrame(columns=['zone', 'time', 'perc'])
    # increment index by 1
    df.index = df.index + 1
    df['zone'] = summary.index
    df['time'] = summary['frame'].values
    df['perc'] = perc_summary['frame'].values
    # print(df)

    return df


def wall_hugging(df):
    leng = len(df)
    num_frames = leng
    time = num_frames/FRAMERATE
    # print(num_frames)
    # print(time)
    # print(df)
    wall_hugging_df = pd.DataFrame(columns=['frame', 'wall_hugging'])
    wall_hugging_df['frame'] = df['frame']
    wall_hugging_df['wall_hugging'] = 0
    for i in range(0, leng, 1):
        if (df.iloc[i]['x'] >2 and df.iloc[i]['x']<10) and (df.iloc[i]['y'] <1 or df.iloc[i]['y']>11):
            wall_hugging_df.at[i,'wall_hugging'] = 1
        else:
            wall_hugging_df.at[i,'wall_hugging'] = 0
    
    summary = wall_hugging_df['wall_hugging'].sum()
    return round(summary/FRAMERATE,3), round(summary/num_frames,3)


def freezing(df):
    leng = len(df)
    num_frames = leng
    time = num_frames/FRAMERATE

    freezing_df = 0

    for i in range(1, leng-FRAMERATE*5, 1):
        # distance travelled in 5 seconds for every consecutive frame_pair
        dist = 0
        for j in range(i, i+FRAMERATE*5, 1):
            if df.iloc[j]['x']==0 or df.iloc[j-1]['x']==0:
                continue
            dist += calculate_distance(df.iloc[j]['x'], df.iloc[j]['y'], df.iloc[j-1]['x'], df.iloc[j-1]['y'])
        if dist < 1:
            freezing_df = 1
    summary = freezing_df['freezing'].sum()
    return freezing_df

def lets_anova(df_pre,df_post):
    pass

if __name__ == "__main__":
    for pre_post in ['pre', 'post']:
        basepath = f'{DATAPATH}/{pre_post}_intervention'
        csv_path = os.path.join(basepath, 'clean_tracked_centres')
        save_csv_path = os.path.join(basepath, 'metrics', 'distances')
        os.makedirs(save_csv_path, exist_ok=True)
        fishes_df = pd.DataFrame(columns=['fish_name', "shoal_side", 'total_distance', 'average_speed', 'wall_hugging_time', 'wall_hugging_perc', "Shoal_time", "Shoal_perc", "Middle_time", "Middle_perc", "Control_time", "Control_perc", "Freezing_time"])
        fish_list = os.listdir(csv_path)
        try:
            fish_list.remove('.DS_Store')
        except:
            pass
        fishes_df['fish_name'] = [i.split('.')[0] for i in fish_list]

        for fish_name in tqdm.tqdm(fish_list, desc="Calculating metrics", unit="fish"):
            distance_df = pd.DataFrame(columns=['frame', 'dist'])
            if fish_name == '.DS_Store':
                continue
            fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'shoal_side'] = fish_name.split('-')[1].split('.')[0]

            # print(fish_name)
            data=pd.read_csv(os.path.join(csv_path,fish_name))
            distance_df = distance(data)
            # print(distance_df)
            distance_df.to_csv(os.path.join(save_csv_path, fish_name), index=False)
            
            total_distance = round(distance_df['dist'].sum(),3)
            fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'total_distance'] = total_distance
            average_speed = round(total_distance/(len(distance_df)/FRAMERATE),3)
            fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'average_speed'] = average_speed
            
            hugging_info = wall_hugging(data)
            fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'wall_hugging_time'] = hugging_info[0]
            fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'wall_hugging_perc'] = hugging_info[1]

            zone_info = zone(data, fish_name.split('-')[1].split('.')[0])
            # print(zone_info)
            try:
                # appending time and perc of "Shoal" zone
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Shoal_time'] = zone_info.loc[zone_info['zone']=='Shoal', 'time'].values[0]
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Shoal_perc'] = zone_info.loc[zone_info['zone']=='Shoal', 'perc'].values[0]
            except:
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Shoal_time'] = 0
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Shoal_perc'] = 0
            try:
                # appending time and perc of "Middle" zone
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Middle_time'] = zone_info.loc[zone_info['zone']=='Middle', 'time'].values[0]
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Middle_perc'] = zone_info.loc[zone_info['zone']=='Middle', 'perc'].values[0]
            except:
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Middle_time'] = 0
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Middle_perc'] = 0
            try:
                # appending time and perc of "Control" zone
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Control_time'] = zone_info.loc[zone_info['zone']=='Control', 'time'].values[0]
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Control_perc'] = zone_info.loc[zone_info['zone']=='Control', 'perc'].values[0]
            except:
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Control_time'] = 0
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Control_perc'] = 0

            freezing_info = freezing(data)
            try:
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Freezing_time'] = freezing_info
            except:
                fishes_df.loc[fishes_df['fish_name']==fish_name.split('.')[0], 'Freezing_time'] = 0
                


            # break
            
        # rounding off the values to 2 decimal places
        fishes_df = fishes_df.round(3)
        fishes_df = fishes_df.sort_values(by=['fish_name'])
        print(fishes_df)
        fishes_df.to_csv(os.path.join(basepath, 'metrics', 'metrics.csv'), index=False)
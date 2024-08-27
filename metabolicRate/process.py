import os 
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, shapiro, mannwhitneyu, pearsonr
import matplotlib.pyplot as plt
import sys

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})


def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def reject(df, reject_list):
    for i in reject_list:
        df = df[~df['time'].str.contains(i, case=False)]
    return df

def basic_clean(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.loc[:, ~df.columns.str.contains('time', case=False)]

    df = df.iloc[:90, :]
    df = df.dropna(axis=1)
    return df

def meaner(df):
    df_mean = df.mean(axis=1).values
    sem_df = df.sem(axis=1).values

    df_mean = [round(x, 2) for x in df_mean]
    sem_df = [round(x, 2) for x in sem_df]

    df = pd.DataFrame({'time': df.index.values, 'mean': df_mean, 'sem': sem_df})
    return df


def item_wise_ttest(pre, post):
    pre = pre.loc[:, ~pre.columns.str.contains('^Unnamed')]
    pre = pre.loc[:, ~pre.columns.str.contains('time', case=False)]
    
    post = post.loc[:, ~post.columns.str.contains('^Unnamed')]
    post = post.loc[:, ~post.columns.str.contains('time', case=False)]

    pre = pre.T 
    post = post.T

    ts =[]
    ps = []
    normal = True

    for i in range(0, len(pre.columns)):
        pre_col = pre.iloc[:, i]
        post_col = post.iloc[:, i]
        t, p = shapiro(pre_col)
        if p < 0.05:
            normal=False
            break
        t, p = shapiro(post_col)
        if p < 0.05:
            normal=False
            break

    for i in range(0, len(pre.columns)):
        pre_col = pre.iloc[:, i]
        post_col = post.iloc[:, i]
        if normal:
            t, p = ttest_rel(pre_col, post_col)
        else:
            t, p = mannwhitneyu(pre_col, post_col)

        ts.append(t)
        ps.append(p)
    
    ts = [round(x, 2) for x in ts]
    ps = [round(x, 3) for x in ps]
    df = pd.DataFrame({ 't': ts, 'p': ps})
    return df, "T" if normal else "M"

def plotter(pre, post, val, test, outdir, experiment, N):
    ttest_val = val['p'].values 
    ttest_val = [1 if x < 0.05 else 0 for x in ttest_val]
    ttest_val = np.array(ttest_val)

    
    x = np.arange(len(ttest_val))
    y = np.zeros(len(ttest_val))
    X_green = x[ttest_val == 1]
    y_green = y[ttest_val == 1]
    X_red = x[ttest_val == 0]
    y_red = y[ttest_val == 0]
    

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(pre['time'], pre['mean'], label='Day 1' if 'control' in experiment.lower() else 'Pre Int.', color = '#555065' if 'control' in experiment.lower() else '#de7e40')
    ax.plot(post['time'], post['mean'], label='Day 2' if 'control' in experiment.lower() else 'Post Int.', color = '#1ac1b9' if 'control' in experiment.lower() else '#58473e')

    ax.set_xlim(0, 90.0001)
    ax.set_ylim(-0.5, 10)

    ax.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dissolved oxygen (mg/L)', fontsize=14, fontweight='bold')
    # ax.set_title(f'Oxygen Depletion for \n {experiment}', fontsize=16, fontweight='bold')


    ax.set_xticks(np.arange(0, 90.0001, 10))
    ax.set_yticks(np.arange(0, 10.0001, 1))

    ax.fill_between(pre['time'], pre['mean'] - pre['sem'], pre['mean'] + pre['sem'], alpha=0.2, color = '#555065' if 'control' in experiment.lower() else '#de7e40')
    ax.fill_between(post['time'], post['mean'] - post['sem'], post['mean'] + post['sem'], alpha=0.2, color = '#1ac1b9' if 'control' in experiment.lower() else '#58473e')

    # ax.scatter([], [], color='#FFFFFF', label=' ',s=0.001)
    # ax.scatter(X_green, y_green, marker='s', color='green', label='p < 0.05', s=3)
    # ax.scatter(X_red, y_red, marker='s', color='red', label='p > 0.05', s=3)
    
    ax.scatter([], [], color='#FFFFFF', label=' ',s=0.001)
    ax.scatter([], [], color='#FFFFFF', label=f'N={N}',s=0.001)
    
    ax.scatter([], [], color='#FFFFFF', label=' ',s=0.001)
    ax.scatter([], [], color='#FFFFFF', label=f'Cohen\'s d:',s=0.001)
    ax.scatter([], [], color='#FFFFFF', label=f'• d= {round(cohens_d(pre["mean"], post["mean"]), 3)}',s=0.001)
    print('Cohens D')
    print(f'• d= {round(cohens_d(pre["mean"], post["mean"]), 3)}')
    ax.scatter([], [], color='#FFFFFF', label=' ',s=0.001)
    if experiment == 'Social Isolation':
        ax.scatter([], [], color='#FFFFFF', label=f'Mann-Whitney\nU Test:' if test == 'M' else 'T-test',s=0.001)
        ax.scatter([], [], color='#FFFFFF', label='* p<=0.05',s=0.001)
        ax.scatter([], [], color='#FFFFFF', label='** p<=0.001',s=0.001)

        points = [31,50]

        for point in points:
            ax.plot([point, point], [pre['mean'][point] - pre['sem'][point]*3, post['mean'][point] + post['sem'][point]*3], color='black', linewidth=1, linestyle='--')
            # ax.plot([point, point-1], [pre['mean'][point] + pre['sem'][point], pre['mean'][point] + pre['sem'][point]], color='black', linewidth=1, linestyle='--')
            # ax.plot([point, point-1], [post['mean'][point] - post['sem'][point], post['mean'][point] - post['sem'][point]], color='black', linewidth=1, linestyle='--')
            # upper_point = post['mean'][point] - post['sem'][point] + (pre['mean'][point] + pre['sem'][point] - post['mean'][point] - post['sem'][point])/3
            marker = r"$*$" if point == 31 else r"$**$"
            ax.annotate(f'{marker}', (point - (1 if point==31 else 2), post['mean'][point] + post['sem'][point]*3), fontsize=8, fontweight='bold')

    if experiment == 'Water Control':
        pearson = pearsonr(pre['mean'], post['mean'])
        ax.scatter([], [], color='#FFFFFF', label=f'Pearson\nCorrelation:',s=0.001)
        ax.scatter([], [], color='#FFFFFF', label=f'• r= {round(pearson[0], 3)}',s=0.001)
        ax.scatter([], [], color='#FFFFFF', label=r'• p<=0.05',s=0.001)
        print(pearson)
        print('Pearson Correlation')
        

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()

    plt.savefig(os.path.join(outdir, f'{experiment}.png'), dpi=300)
    plt.close()
    pass

def cumulate(pre, post):
    pre_col = pre.columns
    post_col = post.columns

    pre = pre[pre_col]
    post = post[post_col]

    return pre, post

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base', type=str, help='base directory')
    parser.add_argument('--experiment', type=str, help='experiment name')
    parser.add_argument('--outdir', type=str, help='output directory')
    parser.add_argument('--reject', type=list, help='reject list', default=None)
    args = parser.parse_args()

    base = args.base
    experiment = args.experiment

    exp_dict = {
        'rep': 'Water Control',
        'dmso': 'DMSO Control',
        'drug': 'Pharmacological Intervention',
        'social': 'Social Isolation',
    }
    indir = os.path.join(base, experiment + "-data", "clean")
    outdir = os.path.join(args.outdir, "plots_without_title")
    os.makedirs(outdir, exist_ok=True)

    # print(indir)
    # print(outdir)
    # print(experiment)

    pre = pd.read_csv(os.path.join(indir, 'pre_data.csv'))
    post = pd.read_csv(os.path.join(indir, 'post_data.csv'))
    
    
    pre = basic_clean(pre)
    post = basic_clean(post)



    vals, test = item_wise_ttest(pre, post)

    N = len(post.columns)
    pre = meaner(pre)
    post = meaner(post)

    # if args.reject is not None:
    #     pre = reject(pre, args.reject)
    #     post = reject(post, args.reject)

    # print(vals)
    # print(test)
    # print(pre)
    # print(post)



    experiment = exp_dict[experiment]
    print(experiment)
    plotter(pre, post, vals, test, outdir, experiment, N)
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
from scipy.stats import ttest_rel
import pandas as pd 
import matplotlib.pyplot as plt
from numpy import mean, var, sqrt
import numpy as np
import os
from statsmodels.stats.anova import AnovaRM 
import sys
from matplotlib import rc
from scipy.stats import shapiro, mannwhitneyu, chi2_contingency
from statsmodels.formula.api import ols
import statsmodels.api as sm

# change default font to times new roman
rc('font',**{'family':'serif','serif':['Times New Roman']})

master_dict = {
    "test_name": "",
    "variable_1": "",
    "variable_2": "",
    "value_ID": "",
    "value": 0.0,
    "value_unit": "",
}

DATAPATH = "" # ADD THE GLOBAL DATA PATH HERE
PLOTPATH = f"{DATAPATH}/plots"
os.makedirs(PLOTPATH, exist_ok=True)

def cohen_d(x, y):
    """Calculate Cohen's d.

    Args:
        x (list): First set of data.
        y (list): Second set of data.

    Returns:
        float: Cohen's d.
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*var(x)+(ny-1)*var(y)) / dof)

def combine_df(df):
    # split fish name at "-" and take only the frst part
    df[["fish_name", "fish_num"]] = df.fish_name.str.split("-", expand=True)
    df = df.drop(columns=["fish_num", "shoal_side"])
    # group by fish name and mean
    df_mean = df.groupby(["fish_name"]).mean()
    df_sem = df.groupby(["fish_name"]).sem()
    # change sem col names to _sem
    df_sem.columns = [str(col) + '_sem' for col in df_sem.columns]
    # combine sem with mean
    df = pd.concat([df_mean, df_sem], axis=1)
    # reset index
    df['fish_name'] = df.index
    df.index = range(len(df))



    return df

def sig(pre_df, post_df):
    total_distance_sig = significance_test(pre_df["total_distance"], post_df["total_distance"])
    total_distance_sig = round(total_distance_sig, 3)
    print("Shoal side: {}".format(total_distance_sig))

    average_speed_sig = significance_test(pre_df["average_speed"], post_df["average_speed"])
    average_speed_sig = round(average_speed_sig, 3)
    print("Shoal side: {}".format(average_speed_sig))
    
    wall_hugging_time_sig = significance_test(pre_df["wall_hugging_time"], post_df["wall_hugging_time"])
    wall_hugging_time_sig = round(wall_hugging_time_sig, 3)
    print("Shoal side: {}".format(wall_hugging_time_sig))
    
    Shoal_time_sig = significance_test(pre_df["Shoal_time"], post_df["Shoal_time"])
    Shoal_time_sig = round(Shoal_time_sig, 3)
    print("Shoal side: {}".format(Shoal_time_sig))
    
    Middle_time_sig = significance_test(pre_df["Middle_time"], post_df["Middle_time"])
    Middle_time_sig = round(Middle_time_sig, 3)
    print("Shoal side: {}".format(Middle_time_sig))
    
    Control_time_sig = significance_test(pre_df["Control_time"], post_df["Control_time"])
    Control_time_sig = round(Control_time_sig, 3)
    print("Shoal side: {}".format(Control_time_sig))


def significance_test(data1, data2):
    """Perform a significance test on two sets of data.

    Args:
        data1 (list): First set of data.
        data2 (list): Second set of data.

    Returns:
        bool: True if the two sets of data are significantly different.
    """
    return round(ttest_rel(data1, data2).pvalue,3)

def simplify(df_pre, df_post):
    df_pre = df_pre[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    df_post = df_post[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    # split fish name into two columns
    df_pre[["fish_name", "fish_num"]] = df_pre.fish_name.str.split("-", expand=True)
    df_post[["fish_name", "fish_num"]] = df_post.fish_name.str.split("-", expand=True)

    # group by fish name
    df_pre = df_pre.sort_values(by=["fish_name"]).drop(columns=["fish_num"])
    df_post = df_post.sort_values(by=["fish_name"]).drop(columns=["fish_num"])


    df_pre = df_pre.groupby(["fish_name"]).mean()
    df_pre['intervention'] = "pre"
    df_post = df_post.groupby(["fish_name"]).mean()
    df_post['intervention'] = "post"

    fishes = df_pre.index.values.tolist()
    df_pre["fish_name"] = fishes
    df_post["fish_name"] = fishes
    df_pre.index = range(len(df_pre))
    df_concat = pd.concat([df_pre, df_post], axis=0)




    return df_concat

def plot_detailed_zonal(df_pre, df_post):
    df_pre = combine_df(df_pre)
    df_post = combine_df(df_post)
    df_pre = df_pre[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    df_post = df_post[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    # print(df_pre)
    # print(df_post)

    fig, ax = plt.subplots(figsize=(5, 5))


    # plot with x axis as zones and y axis as time
    ax.boxplot([df_pre["Shoal_time"], df_pre["Middle_time"], df_pre["Control_time"]],
                positions=[1,3,5],
                patch_artist=True,
                boxprops=dict(facecolor="#a79baa"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                )
    ax.boxplot([df_post["Shoal_time"], df_post["Middle_time"], df_post["Control_time"]],
                positions=[2,4,6],
                patch_artist=True,
                boxprops=dict(facecolor="#f8e6ca"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                )
    
    max = df_pre["Shoal_time"].max() if df_pre["Shoal_time"].max() > df_post["Middle_time"].max() else df_post["Middle_time"].max()
    ax.plot([1, 2], [max + max/10, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.plot([1, 1], [max + max/10 - 5, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.plot([2, 2], [max + max/10 - 5, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.text(1.5, max + max/10, "*", ha="center", va="center", color="black")

    max = df_pre["Middle_time"].max() if df_pre["Middle_time"].max() > df_post["Middle_time"].max() else df_post["Middle_time"].max()
    ax.plot([3, 4], [max + max/10, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.plot([3, 3], [max + max/10 - 5, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.plot([4, 4], [max + max/10 - 5, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.text(3.5, max + max/10, "*", ha="center", va="center", color="black")

    max = df_pre["Control_time"].max() if df_pre["Control_time"].max() > df_post["Control_time"].max() else df_post["Control_time"].max()
    ax.plot([5, 6], [max + max/10, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.plot([5, 5], [max + max/10 - 5, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.plot([6, 6], [max + max/10 - 5, max + max/10], color="black", linestyle="--", linewidth=0.5)
    ax.text(5.5, max + max/10, "*", ha="center", va="center", color="black")

    ax.set_ylabel("Time (s)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Zones", fontsize=16, fontweight='bold')
    ax.set_title("Time Spent in Zones", fontsize=16, fontweight='bold')


    ax.set_xticks([1.5, 3.5, 5.5])
    ax.set_xticklabels(["Shoal", "Middle", "Control"], fontsize=16)

    ax.set_ylim(bottom=0, top=350)


    # custom legend
    ax.plot([], [], ' ', label="Pre Int.", color="#a79baa", marker="s")
    ax.plot([], [], ' ', label="Post Int.", color="#f8e6ca", marker="s")
    # ax.plot([], [], ' ', label=" ", color="#FFFFFF", marker="s")
    ax.plot([], [], ' ', label="N=26", color="#FFFFFF", marker="s")
    # ax.plot([], [], ' ', label=" ", color="#FFFFFF", marker="s")
    ax.plot([], [], ' ', label="* p<0.05", color="#FFFFFF", marker="s")
    ax.legend()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(PLOTPATH + "/zonal_boxplot.png", dpi=300)



def plot_zonal(df_pre, df_post):

    df_simp = simplify(df_pre, df_post)
    # making Shoal_time, Middle_time, Control_time into one column
    df_simp = pd.melt(df_simp, id_vars=['fish_name', 'intervention'], value_vars=['Shoal_time',  'Middle_time',  'Control_time'], var_name='zone', value_name='time')
    print(df_simp)
    anova = AnovaRM(df_simp, depvar='time', subject='fish_name', within=['zone', 'intervention'])
    res = anova.fit()
    print(res)


    df_pre = df_pre[["Shoal_time", "Middle_time", "Control_time"]]
    df_post = df_post[["Shoal_time", "Middle_time", "Control_time"]]



    df_pre_sem = df_pre.sem()
    df_post_sem = df_post.sem()

    df_pre = df_pre.mean()
    df_post = df_post.mean()


    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(x=[1,2,3,4,5,6],
            height = (df_pre["Shoal_time"], df_post["Shoal_time"], df_pre["Middle_time"], df_post["Middle_time"], df_pre["Control_time"], df_post["Control_time"]),
            width=0.75,
            yerr=(df_pre_sem["Shoal_time"], df_post_sem["Shoal_time"], df_pre_sem["Middle_time"], df_post_sem["Middle_time"], df_pre_sem["Control_time"], df_post_sem["Control_time"]),
            capsize=3,
            color=["#a79baa", "#f8e6ca", "#a79baa", "#f8e6ca", "#a79baa", "#f8e6ca"],
            ecolor="black",
            edgecolor="black",
            )
    
    # set 1:2 to a common axis label on x axis
    ax.set_xticks([1.5, 3.5, 5.5])
    ax.set_xticklabels(["Shoal", "Middle", "Control"], fontsize=16)
    
    ax.set_ylabel("Time (s)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Zones", fontsize=16, fontweight='bold')

    ax.set_title("Time Spent in Zones", fontsize=16, fontweight='bold')

    # custom legend
    ax.plot([], [], ' ', label="Pre Int.", color="#a79baa", marker="s")
    ax.plot([], [], ' ', label="Post Int.", color="#f8e6ca", marker="s")
    ax.plot([], [], ' ', label="N=26", color="#FFFFFF", marker="s")
    ax.legend()

           
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(PLOTPATH + "/zonal_barplot.png", dpi=300)

    pass


def plot_stats(df_pre, df_post):
    values_Df = pd.DataFrame(columns=master_dict.keys())

    df_pre["combined"] = df_pre["Middle_time"] + df_pre["Control_time"]
    df_post["combined"] = df_post["Middle_time"] + df_post["Control_time"]

    df_pre = combine_df(df_pre)
    df_post = combine_df(df_post)

    print(df_pre)
    print(df_post)

    list_of_cols = ["Shoal_time", "Middle_time", "Control_time","total_distance",  "average_speed",  "wall_hugging_time", "Freezing_time", "combined", "ratio"] 
    unlimited_cols = ["total_distance",  "average_speed",  "wall_hugging_time", "Freezing_time", "combined", "ratio"]
    fig_titles = ["Time Spent in Shoal Zone", "Time Spent in Middle Zone", "Time Spent in Shoal-averse Zone", "Total Distance Travelled", "Average Speed", "Time Spent Wall Hugging", "Time Spent Freezing", "Time Spent in Middle + Shoal-averse Zones", "Proportion of Time Spent in the Shoal Zone"]
    yaxis = ["Time (s)", "Time (s)", "Time (s)", "Distance (cm)", "Speed (cm/s)", "Time (s)", "Time (s)", "Time (s)", "Proportion (unitless)"]
    xaxis = ["Intervention Stage", "Intervention Stage", "Intervention Stage", "Intervention Stage", "Intervention Stage", "Intervention Stage", "Intervention Stage", "Intervention Stage", "Intervention Stage"]

    stats_df = pd.DataFrame(columns=["mean", "sem", "var", "std", "intervention","measure"])
    signif_test_Df = pd.DataFrame(columns=["var1", "var2", "stat", "pvalue", "intervention", "test"])

    for col in list_of_cols:
        # print(col)
        # print(significance_test(df_pre[col], df_post[col]))
        cohend = round(cohen_d(df_pre[col], df_post[col]),3)
        shap_pre = shapiro(df_pre[col])
        shap_post = shapiro(df_post[col])

        # print(f"Cohen's d: {cohend}")
        # print(f"Shapiro-Wilk test: {shap_pre}, {shap_post}")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot([df_pre[col]],
                    positions=[1],
                    patch_artist=True,
                    boxprops=dict(facecolor="#a79baa"),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    meanprops=dict(color="black", marker="--"),
                    )
        ax.boxplot([df_post[col]],
                    positions=[2],
                    patch_artist=True,
                    boxprops=dict(facecolor="#f8e6ca"),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    meanprops=dict(color="black", marker="--"),
                    )
        
        pre_stats = {}
        post_stats = {}
        pre_stats["mean"] = df_pre[col].mean()
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Mean"
        df_cpy["variable_1"] = f'Pre intervention for {col}'
        df_cpy["value_ID"] = "Mean"
        df_cpy["value"] = df_pre[col].mean()
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)
        
        pre_stats["sem"] = df_pre[col].sem()
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Standard Error"
        df_cpy["variable_1"] = f'Pre intervention for {col}'
        df_cpy["value_ID"] = "SEM"
        df_cpy["value"] = df_pre[col].sem()
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)
        
        post_stats["mean"] = df_post[col].mean()
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Mean"
        df_cpy["variable_1"] = f'Post intervention for {col}'
        df_cpy["value_ID"] = "Mean"
        df_cpy["value"] = df_post[col].mean()
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

        post_stats["sem"] = df_post[col].sem()
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Standard Error"
        df_cpy["variable_1"] = f'Post intervention for {col}'
        df_cpy["value_ID"] = "SEM"
        df_cpy["value"] = df_post[col].sem()
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

        pre_stats["var"] = df_pre[col].var()
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Variance"
        df_cpy["variable_1"] = f'Pre intervention for {col}'
        df_cpy["value_ID"] = "Variance"
        df_cpy["value"] = df_pre[col].var()
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

        post_stats["var"] = df_post[col].var()
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Variance"
        df_cpy["variable_1"] = f'Post intervention for {col}'
        df_cpy["value_ID"] = "Variance"
        df_cpy["value"] = df_post[col].var()
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

        pre_stats["std"] = df_pre[col].std()
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Standard Deviation"
        df_cpy["variable_1"] = f'Pre intervention for {col}'
        df_cpy["value_ID"] = "Standard Deviation"
        df_cpy["value"] = df_pre[col].std()
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

        post_stats["std"] = df_post[col].std()
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Standard Deviation"
        df_cpy["variable_1"] = f'Post intervention for {col}'
        df_cpy["value_ID"] = "Standard Deviation"
        df_cpy["value"] = df_post[col].std()
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

        pre_stats["intervention"] = "pre"
        post_stats["intervention"] = "post"
        pre_stats["measure"] = col
        post_stats["measure"] = col

        pre_stats = pd.DataFrame.from_dict(pre_stats, orient="index", columns=["Pre"])
        post_stats = pd.DataFrame.from_dict(post_stats, orient="index", columns=["Post"])
        pre_stats = pre_stats.transpose()
        post_stats = post_stats.transpose()
        print(pre_stats)

        stats_df = pd.concat([stats_df, pre_stats, post_stats], axis=0)


        ax.set_xlabel(xaxis[list_of_cols.index(col)], fontsize=16, fontweight='bold')
        ax.set_ylabel(yaxis[list_of_cols.index(col)], fontsize=16, fontweight='bold')
        # plt.suptitle(fig_titles[list_of_cols.index(col)], fontsize=16, fontweight='bold')
        
        ax.set_xticks([1,2])
        ax.set_xticklabels(["Pre", "Post"], fontsize=16)

        # custom legend
        ax.plot([], [], ' ', label="Pre Int.", color="#a79baa", marker="s")
        ax.plot([], [], ' ', label="Post Int.", color="#f8e6ca", marker="s")

        ax.plot([], [], ' ', label="N=26      ", color="#FFFFFF", marker="s")


        ax.plot([], [], ' ', label=" ", color="#FFFFFF", marker="s")
        ax.plot([], [], ' ', label="Cohen's d:", color="#FFFFFF", marker="s")
        ax.plot([], [], ' ', label="• d={}".format(cohend), color="#FFFFFF", marker="s")
        test_2 = {"var1":f"{col}_pre", "var2":f"{col}_post", "stat":cohend, "pvalue":0, "intervention":"pre", "test":"cohend"}
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Cohen's d"
        df_cpy["variable_1"] = f'Pre intervention for {col}'
        df_cpy["variable_2"] = f'Post intervention for {col}'
        df_cpy["value_ID"] = "Cohen's d"
        df_cpy["value"] = cohend
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

        ax.plot([], [], ' ', label=" ", color="#FFFFFF", marker="s")
        ax.plot([], [], ' ', label="Shapiro-Wilk (Pre):\n• W={}\n• p{}0.05".format(round(shap_pre[0],3), '<=' if round(shap_pre[1],3)<=0.05 else '>'), color="#FFFFFF", marker="s")
        test_3 = {"var1":f"{col}", "var2":f"{col}", "stat":shap_pre[0], "pvalue":shap_pre[1], "intervention":"pre", "test":"shapiro"}
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Shapiro-Wilk"
        df_cpy["variable_1"] = f'Pre intervention for {col}'
        df_cpy["value_ID"] = "W"
        df_cpy["value"] = shap_pre[0]
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Shapiro-Wilk"
        df_cpy["variable_1"] = f'Pre intervention for {col}'
        df_cpy["value_ID"] = "p-value"
        df_cpy["value"] = shap_pre[1]
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)


        ax.plot([], [], ' ', label=" ", color="#FFFFFF", marker="s")
        ax.plot([], [], ' ', label="Shapiro-Wilk (Post):\n• W={}\n• p{}0.05".format(round(shap_post[0],3), '<=' if round(shap_post[1],3)<=0.05 else '>'), color="#FFFFFF", marker="s")
        test_4 = {"var1":f"{col}", "var2":f"{col}", "stat":shap_post[0], "pvalue":shap_post[1], "intervention":"post", "test":"shapiro"}
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Shapiro-Wilk"
        df_cpy["variable_1"] = f'Post intervention for {col}'
        df_cpy["value_ID"] = "W"
        df_cpy["value"] = shap_post[0]
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)
        df_cpy = master_dict.copy()
        df_cpy["test_name"] = "Shapiro-Wilk"
        df_cpy["variable_1"] = f'Post intervention for {col}'
        df_cpy["value_ID"] = "p-value"
        df_cpy["value"] = shap_post[1]
        values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)


        test_2 = pd.DataFrame.from_dict(test_2, orient="index", columns=["Pre"]).T
        test_3 = pd.DataFrame.from_dict(test_3, orient="index", columns=["Pre"]).T
        test_4 = pd.DataFrame.from_dict(test_4, orient="index", columns=["Pre"]).T

        signif_test_Df = pd.concat([signif_test_Df, test_2, test_3, test_4], axis=0)




        ax.plot([], [], ' ', label=" ", color="#FFFFFF", marker="s")
        if shap_pre[1] < 0.05 or shap_post[1] < 0.05:
            man1 = mannwhitneyu(df_pre[col], df_post[col])
            ax.plot([], [], ' ', label="Mann-Whitney U: \n• U={} \n• p{}0.05".format(round(man1[0],3), '<=' if round(man1[1],3)<=0.05 else '>'), color="#FFFFFF", marker="s")
            df_cpy = master_dict.copy()
            df_cpy["test_name"] = "Mann-Whitney U"
            df_cpy["variable_1"] = f'Pre intervention for {col}'
            df_cpy["variable_2"] = f'Post intervention for {col}'
            df_cpy["value_ID"] = "U"
            df_cpy["value"] = man1[0]
            values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)
            df_cpy = master_dict.copy()
            df_cpy["test_name"] = "Mann-Whitney U"
            df_cpy["variable_1"] = f'Pre intervention for {col}'
            df_cpy["variable_2"] = f'Post intervention for {col}'
            df_cpy["value_ID"] = "p-value"
            df_cpy["value"] = man1[1]
            values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

            test_5 = {"var1":f"{col}_pre", "var2":f"{col}_post", "stat":man1[0], "pvalue":man1[1], "intervention":col, "test":"mannwhitneyu"}
            test_5 = pd.DataFrame.from_dict(test_5, orient="index", columns=["Pre"]).T
            signif_test_Df = pd.concat([signif_test_Df, test_5], axis=0)
            print(man1)
            if man1[1] < 0.05:
                # add link between two box plots and asterisks
                max = df_pre[col].max() if df_pre[col].max() > df_post[col].max() else df_post[col].max()
                diff = max/10
                ax.plot([1, 2], [max + diff, max + diff], color="black", linestyle="--", linewidth=0.5)
                ax.plot([1, 1], [max + diff/1.5, max+diff], color="black", linestyle="--", linewidth=0.5)
                ax.plot([2, 2], [max + diff/1.5, max+diff], color="black", linestyle="--", linewidth=0.5)
                stars = "*" 
                ax.text(1.5, max + diff, stars, ha="center", va="center", color="black")


        else:
            ax.plot([], [], ' ', label="T-test:\n• t={}\n• p{}0.05".format(round(ttest_rel(df_pre[col], df_post[col]).statistic,3), '<=' if round(ttest_rel(df_pre[col], df_post[col]).pvalue,3) <=0.05 else '>'), color="#FFFFFF", marker="s")
            test_1 = {"var1":f"{col}_pre", "var2":f"{col}_post", "stat":round(ttest_rel(df_pre[col], df_post[col]).pvalue,3), "pvalue":round(ttest_rel(df_pre[col], df_post[col]).pvalue,3), "intervention":col, "test":"ttest_rel"}
            df_cpy = master_dict.copy()
            df_cpy["test_name"] = "ttest"
            df_cpy["variable_1"] = f'Pre intervention for {col}'
            df_cpy["variable_2"] = f'Post intervention for {col}'
            df_cpy["value_ID"] = "p-value"
            df_cpy["value"] = ttest_rel(df_pre[col], df_post[col]).pvalue
            values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)

            df_cpy = master_dict.copy()
            df_cpy["test_name"] = "ttest"
            df_cpy["variable_1"] = f'Pre intervention for {col}'
            df_cpy["variable_2"] = f'Post intervention for {col}'
            df_cpy["value_ID"] = "t-value"
            df_cpy["value"] = ttest_rel(df_pre[col], df_post[col]).statistic
            values_Df = pd.concat([values_Df, pd.DataFrame.from_dict(df_cpy, orient="index").T], axis=0)
            


            test_1 = pd.DataFrame.from_dict(test_1, orient="index", columns=["Pre"]).T 
            signif_test_Df = pd.concat([signif_test_Df, test_1], axis=0)

            if round(ttest_rel(df_pre[col], df_post[col]).pvalue,3) < 0.05:
                # add link between two box plots and asterisks
                max = df_pre[col].max() if df_pre[col].max() > df_post[col].max() else df_post[col].max()
                diff = max/10
                ax.plot([1, 2], [max + diff, max + diff], color="black", linestyle="--", linewidth=0.5)
                ax.plot([1, 1], [max + diff/1.5, max+diff], color="black", linestyle="--", linewidth=0.5)
                ax.plot([2, 2], [max + diff/1.5, max+diff], color="black", linestyle="--", linewidth=0.5)
                stars = "*"
                ax.text(1.5, max + diff, stars, ha="center", va="center", color="black")
                

        ax.plot([],[], ' ', label=" ", color="#FFFFFF", marker="s")
        ax.plot([],[], ' ', label='*p<=0.05', color="#FFFFFF", marker="s")

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)

        if col not in unlimited_cols:
            ax.set_ylim(bottom=0, top=350)    
        

        plt.tight_layout()
        # plt.show()
        plt.savefig(PLOTPATH + "/comparison_{}.png".format(col), dpi=300)
        plt.close()
    print(stats_df)
    print(signif_test_Df)
    print(values_Df)
    # stats_df.to_csv(PLOTPATH + "/stats.csv")
    # signif_test_Df.to_csv(PLOTPATH + "/signif_test.csv")

        # break
    values_Df.to_csv(PLOTPATH + "/values_all.csv")


def zonal_combine_mid_control(df_pre, df_post):
    df_pre = combine_df(df_pre)
    df_post = combine_df(df_post)
    df_pre = df_pre[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    df_post = df_post[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    print(df_pre)
    print(df_post)
    df_pre['combined']  = df_pre['Middle_time'] + df_pre['Control_time']
    df_post['combined']  = df_post['Middle_time'] + df_post['Control_time']

    fig, ax = plt.subplots(figsize=(5, 5))


    # plot with x axis as zones and y axis as time
    ax.boxplot([df_pre["Shoal_time"], df_pre["combined"]],
                positions=[1,3],
                patch_artist=True,
                boxprops=dict(facecolor="#a79baa"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                )
    ax.boxplot([df_post["Shoal_time"], df_post["combined"]],
                positions=[2,4],
                patch_artist=True,
                boxprops=dict(facecolor="#f8e6ca"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                )
    
    
    


    ax.set_ylabel("Time (s)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Zones", fontsize=16, fontweight='bold')
    # ax.set_title("Time Spent in Zones", fontsize=16, fontweight='bold')

    ax.set_xticks([1.5, 3.5])
    ax.set_xticklabels(["Shoal", "Middle and\nShoal-averse zones"], fontsize=16)


    # custom legend
    ax.plot([], [], ' ', label="Pre Int.", color="#a79baa", marker="s")
    ax.plot([], [], ' ', label="Post Int.", color="#f8e6ca", marker="s")
    ax.plot([], [], ' ', label="N=26", color="#FFFFFF", marker="s")
    ax.legend()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(PLOTPATH + "/zonal_boxplot_combined.png", dpi=300)
    pass

def zomal_ignore_middle(df_pre,df_post):
    df_pre = df_pre[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    df_post = df_post[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]

    df_pre = df_pre[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    df_post = df_post[["fish_name", "Shoal_time", "Middle_time", "Control_time"]]
    print(df_pre)
    print(df_post)
    df_pre['combined']  = df_pre['Middle_time'] + df_pre['Control_time']
    df_post['combined']  = df_post['Middle_time'] + df_post['Control_time']

    fig, ax = plt.subplots(figsize=(5, 5))      


    # plot with x axis as zones and y axis as time
    ax.boxplot([df_pre["Shoal_time"], df_pre["Control_time"]],
                positions=[1,3],
                patch_artist=True,
                boxprops=dict(facecolor="#a79baa"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                )
    ax.boxplot([df_post["Shoal_time"], df_post["Control_time"]],
                positions=[2,4],
                patch_artist=True,
                boxprops=dict(facecolor="#f8e6ca"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                )
    
    
    


    ax.set_ylabel("Time (s)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Zones", fontsize=16, fontweight='bold')
    ax.set_title("Time Spent in Zones", fontsize=16, fontweight='bold')

    ax.set_xticks([1.5, 3.5])
    ax.set_xticklabels(["Shoa Zone", "Shoal-averse Zone"], fontsize=16)


    # custom legend
    ax.plot([], [], ' ', label="Pre Int.", color="#a79baa", marker="s")
    ax.plot([], [], ' ', label="Post Int.", color="#f8e6ca", marker="s")
    ax.plot([], [], ' ', label="N=26", color="#FFFFFF", marker="s")
    ax.legend()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(PLOTPATH + "/zonal_boxplot_shoal_control.png", dpi=300)
    pass

def lets_anova(df_pre,df_post):

    print("----------- ----- ---------")
    print("----------- ANOVA ---------")
    print("----------- ----- ---------")
    df_pre = df_pre[[ "Shoal_time", "Middle_time", "Control_time"]]
    df_post = df_post[[ "Shoal_time", "Middle_time", "Control_time"]]

    df_pre['intervention'] = "pre"
    df_post['intervention'] = "post"
    df = pd.concat([df_pre, df_post], axis=0)
    # make shoal time, middle time, control time into one column
    df = pd.melt(df, id_vars=['intervention'], value_vars=['Shoal_time',  'Middle_time',  'Control_time'], var_name='zone', value_name='time')
    # anova with eta squared
    print(len(df))
    model = ols('time ~ C(zone) + C(intervention) + C(zone):C(intervention)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    eta = (anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table['sum_sq'].sum()))**2
    anova_table['eta'] = eta
    anova_table = anova_table.round(3)

    print(anova_table)
    pass


def other_chi(df_pre, df_post):

    print("----------- ---------- ---------")
    print("----------- CHI-SQUARE ---------")
    print("----------- ---------- ---------")
    df_pre = combine_df(df_pre)
    df_post = combine_df(df_post)



    df_pre = df_pre[["Shoal_time", "Middle_time", "Control_time"]]
    df_post = df_post[["Shoal_time", "Middle_time", "Control_time"]]
    df_pre['sum'] = df_pre.sum(axis=1)
    df_post['sum'] = df_post.sum(axis=1)
    print(df_pre)
    print(df_post)

    # norm so that df_pre[i].sum() == df_post[i].sum()
    for i in ["Shoal_time", "Middle_time", "Control_time"]:
        df_post[i] = df_post[i] * (df_pre[i].sum() / df_post[i].sum())



    from scipy.stats import chisquare
    chi = chisquare(f_exp=df_pre[['Shoal_time', 'Middle_time', 'Control_time']], f_obs=df_post[['Shoal_time', 'Middle_time', 'Control_time']])
    chi = pd.DataFrame(chi, index=["Chi-Square", "p-value"], columns=["Shoal_time", "Middle_time", "Control_time"])
    print(chi)





if __name__ == "__main__":
    post_path = f"{DATAPATH}/post_intervention/metrics/metrics.csv"
    pre_path = f"{DATAPATH}/pre_intervention/metrics/metrics.csv"

    post_df = pd.read_csv(post_path)
    post_df["sum"] = post_df["Shoal_time"] + post_df["Control_time"]
    post_df["ratio"] =  post_df["Shoal_time"] / post_df["sum"]


    pre_df = pd.read_csv(pre_path)
    pre_df["sum"] = pre_df["Shoal_time"] + pre_df["Control_time"]
    pre_df["ratio"] =  pre_df["Shoal_time"] / pre_df["sum"]
    

    print(pre_df)
    
    # sys.exit(0)

    # print(pre_df.columns)
    # print(post_df.columns)

    
    # sig(pre_df, post_df)
    # plot_zonal(pre_df, post_df)

    # post_df = pd.read_csv(post_path)
    # pre_df = pd.read_csv(pre_path)
    # plot_detailed_zonal(pre_df, post_df)

    # post_df = pd.read_csv(post_path)
    # pre_df = pd.read_csv(pre_path)
    plot_stats(pre_df, post_df)

    # post_df = pd.read_csv(post_path)
    # pre_df = pd.read_csv(pre_path)
    # zonal_combine_mid_control(pre_df, post_df)

    # post_df = pd.read_csv(post_path)
    # pre_df = pd.read_csv(pre_path)
    # zomal_ignore_middle(pre_df, post_df)

    # post_df = pd.read_csv(post_path)
    # pre_df = pd.read_csv(pre_path)
    # lets_anova(pre_df, post_df)

    # post_df = pd.read_csv(post_path)
    # pre_df = pd.read_csv(pre_path)
    # other_chi(pre_df, post_df)

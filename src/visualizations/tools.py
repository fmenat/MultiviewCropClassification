import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_conf_matrix(ax, cf_matrix, add_title=""):
	ax = sns.heatmap(cf_matrix/np.sum(cf_matrix,axis=1, keepdims=True), 
		annot=True, fmt='.2%', cmap='Blues',  vmin=0, vmax=1, cbar=False)
	ax.set_xlabel('\nPredicted Values')
	ax.set_ylabel('Actual Values ')
	ax.set_title(f"Confusion {add_title}")


def plot_prob_dist_bin(ax, y_pred_prob, y_true, add_title=""):
	mask_non_crop = y_true == 0
	mask_crop = y_true == 1
	ax.hist(y_pred_prob[mask_non_crop,1], label="Non crop", bins=15, alpha=0.5)
	ax.hist(y_pred_prob[mask_crop,1], label="crop", bins=15, alpha=0.5)
	ax.set_xlim(0,1)
	ax.set_title(f"Prob dist {add_title}")
	ax.legend(loc="upper center")


def plot_comparison_radar(ax, metrics, metrics_names, categories, add_title=""):
    angles = np.linspace(0, 2*np.pi, len(metrics[0]), endpoint=False).tolist() #in radans
    angles += angles[:1] # repeat first angle to close poly    # plot
    
    for metric, metric_names in zip(metrics, metrics_names):
        ax.plot(angles, [*metric, metric[0]], linestyle="--", marker="o", label=metric_names) 
        ax.fill(angles, [*metric, metric[0]], alpha=0.3) 

    ax.set_xticks(angles[:-1])   
    ax.set_xticklabels(categories)
    ax.set_rlim(0,1)
    ax.set_rticks(np.linspace(0,1, 6))
    ax.set_rlabel_position(90*angles[1]/np.pi) 
    ax.grid(True)
    ax.set_title(f"Methods comparison {add_title}")
    ax.legend(loc="lower right")


def plot_attention(ax, means_ ,stds_, view_names = [], add_title=""):
    if len(view_names) == 0:
        view_names = [f"S{str(v)}" for v in np.arange(len(means_))]
    ax.plot(means_, "o-")
    ax.fill_between(np.arange(len(view_names)), means_-stds_, means_+stds_, alpha=0.5)
    ax.set_xticks(np.arange(len(view_names)))
    ax.set_xticklabels(view_names)
    ax.set_ylim(0,1)
    ax.set_title("mean rounded by std "+add_title)

def plot_raws_att_boxplot(ax, data, view_names=[], add_title=""):
    if len(view_names) == 0:
        view_names = [f"S{str(v)}" for v in np.arange(data.shape[1])]
    sns.boxplot(data=data, ax=ax)
    ax.set_xticks(np.arange(len(view_names)))
    ax.set_xticklabels(view_names)
    ax.set_ylim(0,1)
    ax.set_title("raw values for diff examples "+ add_title)

def plot_raws_att(ax, data, view_names=[], add_title=""):
    if len(view_names) == 0:
        view_names = [f"S{str(v)}" for v in np.arange(data.shape[1])]
    ax.plot(data,  "o--")
    ax.set_xticks(np.arange(len(view_names)))
    ax.set_xticklabels(view_names)
    ax.set_ylim(0,1)
    ax.set_title("raw attention for diff examples "+ add_title)
    
def plot_att_sources_p_class(ax, matrix,view_names=[], class_names=[]):
    if len(view_names) == 0:
        view_names = [f"S{str(v)}" for v in np.arange(matrix.shape[1])]
    if len(class_names) == 0:
        class_names = [f"class-{str(v)}" for v in np.arange(matrix.shape[0])]
    sns.heatmap(matrix, ax=ax, annot=True, fmt='.2f', cmap='Reds',  vmin=0, vmax=1, mask=0,
                     linewidths=.5, annot_kws={"size": 14})
    ax.set_xticklabels(view_names)
    ax.set_yticklabels(class_names, rotation = 0)

def plot_col_feat_result(ax, df, col, marker, add_title="", **args_plot):
    ax.plot(df[col].values, label = col, marker = marker, **args_plot)
    ax.set_title(col+add_title)
    ax.set_ylim(-0.05,1.05)
    ax.set_xlabel("features-axis")
    ax.set_xticks(np.arange(len(df[col])))

def plot_col_feat_mean_std(axx, df_mean,df_std, makers_plot, **args_plot):
    for i, col in enumerate(df_mean.columns):
        axx[0,0].plot(df_mean[col], label = col, marker = makers_plot[i], **args_plot)
        axx[0,1].plot(df_std[col], label = col, marker = makers_plot[i], **args_plot)
    for i in range(axx.shape[1]):
        axx[0,i].set_ylim(-0.05,1.05)
        axx[0,i].set_xlabel("features-axis")
        axx[0,i].legend()
    axx[0,0].set_ylabel("MEAN over runs")
    axx[0,1].set_ylabel("STD over runs")

def plot_heatmat_views_relation(ax, matrix, view_names, add_title="", annot=True):
    if np.nanmax(matrix) <= 1:
        matrix = matrix*100
    mask = matrix == 0
    sns.heatmap(matrix, annot=annot, fmt='.2f', cmap='Reds',  vmin=0, vmax=100, mask=mask,
                     linewidths=.5, annot_kws={"size": 14}, ax=ax)
    ax.set_xticklabels(view_names)
    ax.set_yticklabels(view_names, rotation=0)
    ax.set_title(f"Views relation {add_title}")

def plot_negheatmat_views_relation(ax, matrix, view_names, add_title=""):
    if np.nanmax(matrix) <= 1:
        matrix = matrix*100
    mask = matrix == 0
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdBu',  vmin=-100, vmax=100, mask=mask,
                     linewidths=.5, annot_kws={"size": 14}, ax=ax)
    ax.set_xticklabels(view_names)
    ax.set_yticklabels(view_names, rotation=0)
    ax.set_title(f"Views {add_title}")
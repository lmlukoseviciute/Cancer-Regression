import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np


def plot_pairwise_in_chunks(df, hue=None, max_plots_per_fig=9):
    """
    Plots all pairwise feature combinations (including across groups) in chunks.
    
    Parameters:
        df: DataFrame with features and target.
        hue: Optional column name for color-coding by class.
        max_plots_per_fig: Number of plots per figure (e.g., 9 = 3x3 grid).
    """
    features = df.drop(columns=[hue] if hue else []).select_dtypes(include='number').columns.tolist()
    pairs = [(y, x) for y in features for x in features]
    chunks = [pairs[i:i + max_plots_per_fig] for i in range(0, len(pairs), max_plots_per_fig)]

    for chunk_i, chunk in enumerate(chunks):
        n_plots = len(chunk)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1) 

        for ax, (y_feat, x_feat) in zip(axes, chunk):
            if x_feat == y_feat:
                if hue:
                    for val in df[hue].unique():
                        sns.kdeplot(data=df[df[hue] == val], x=x_feat,
                                    fill=True, alpha=0.4, label=str(val), ax=ax)
                    ax.legend().set_title(hue)
                else:
                    sns.kdeplot(data=df, x=x_feat, fill=True, ax=ax)
                ax.set_ylabel("Density")
            else:
                sns.scatterplot(data=df, x=x_feat, y=y_feat, hue=hue, legend=False, ax=ax)

            ax.set_title(f"{y_feat} vs {x_feat}")
            ax.tick_params(labelsize=8)

        for ax in axes[n_plots:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.suptitle(f"Pairwise Feature Plots - Batch {chunk_i + 1}", y=1.02, fontsize=16)
        plt.show()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COLORS = [(0.922526, 0.385626, 0.209179), (0.280264, 0.715, 0.429209)]

sns.set_context("notebook")
sns.set_style("ticks")
params = {
    "axes.edgecolor": "black",
    "axes.labelsize": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": "large",
    "font.family": "DejaVu Sans",
    "legend.fontsize": "medium",
    "pdf.fonttype": 42,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
}
mpl.rcParams.update(params)


def stripplot(
    data, x, y, dodge=True, palette=COLORS, s=4, alpha=1, outfile=None, **kwargs
):
    """
    Seaborn stripplot wrapper with different default parameters.
    """
    ax = sns.stripplot(
        data=data, x=x, y=y, dodge=dodge, palette=palette, s=s, alpha=alpha, **kwargs
    )
    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")

    return ax


def _add_grid(along="x", linewidth=1, alpha=0.5, color="gray", ax=None, **kwargs):
    """
    Add dashed lines to separate ticks along an axis.
    """
    xlim_start = ax.get_xlim()
    ylim_start = ax.get_ylim()

    if ax is None:
        ax = plt.gca()

    if along == "x":
        grid = np.array(ax.get_xticks())
        line_min, line_max = ylim_start
    else:
        grid = np.array(ax.get_yticks())
        line_min, line_max = xlim_start

    delta = (grid[1] - grid[0]) / 2
    grid_positions = grid[:-1] + delta

    for grid_position in grid_positions:
        if along == "x":
            x = 2 * [grid_position]
            y = [line_min, line_max]
        else:
            x = [line_min, line_max]
            y = 2 * [grid_position]

        ax.plot(x, y, "--", linewidth=linewidth, alpha=alpha, color=color, **kwargs)

    ax.set_xlim(xlim_start)
    ax.set_ylim(ylim_start)

    return ax


def add_grid(along_x=True, along_y=False, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if along_x:
        ax = _add_grid(along="x", ax=ax, **kwargs)

    if along_y:
        ax = _add_grid(along="y", ax=ax, **kwargs)

    return ax


def _boxplot(
    data,
    x,
    y,
    palette=COLORS,
    saturation=1,
    fliersize=0,
    linewidth=1,
    alpha=0.5,
    rotation=0,
    outfile=None,
    **kwargs
):
    """
    Seaborn boxplot wrapper supporting transparency/alpha for the boxes.
    """
    ax = sns.boxplot(
        data=data,
        x=x,
        y=y,
        palette=palette,
        saturation=saturation,
        fliersize=fliersize,
        linewidth=linewidth,
        **kwargs
    )
    # https://github.com/mwaskom/seaborn/issues/979
    for patch in ax.patches:
        r, g, b, _ = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")

    return ax


def boxplot(
    data,
    x,
    y,
    hue=None,
    hue_order=None,
    palette=COLORS,
    dashed_lines=True,
    ax=None,
    outfile=None,
    kwargs_boxplot=None,
    kwargs_stripplot=None,
    kwargs_dashes=None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 4))

    if kwargs_boxplot is None:
        kwargs_boxplot = dict()

    if kwargs_stripplot is None:
        kwargs_stripplot = dict()

    if kwargs_dashes is None:
        kwargs_dashes = dict()

    ax = _boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        **kwargs_boxplot
    )

    # remove boxplot labels to avoid double labeling in the legend
    for patch in ax.patches:
        patch.set_label(s=None)

    ax = stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        **kwargs_stripplot
    )

    if dashed_lines:
        ax = add_grid(along_x=True, ax=ax)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")

    return ax

import numpy as np
from matplotlib import pyplot as plt

def latexify():
    import matplotlib
    params_MPL_Tex = {
                'text.usetex': True,
                'font.family': 'serif',
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 10,
                "font.size": 10,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8
              }
    matplotlib.rcParams.update(params_MPL_Tex)


def setFigureSize(width_pt, fraction=1, n_rows = 1):
    """Set figure dimensions to avoid scaling in LaTeX.
    Sets the height to the figure to the golden ratio times the width.
    Use \the\textwidth to get the width in LaTeX.

    from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width_pt: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    n_rows: int, optional, the number of rows in the figure

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, n_rows*fig_height_in)

    return fig_dim


def makeArrowAxes(ax: plt.Axes, arrows='both'):
    assert arrows in ['both', 'x', 'y', 'none']
    ax.spines[['left','right','top','bottom']].set_visible(False)

    if arrows == 'both' or arrows == 'x':
        ax.plot(1, 0, '>k', transform=ax.transAxes, clip_on=False, markersize=5)
        ax.spines['bottom'].set_visible(True)

    if arrows == 'both' or arrows == 'y':
        ax.spines['left'].set_visible(True)
        ax.plot(0, 1, '^k', transform=ax.transAxes, clip_on=False, markersize=5)




def smoothPlotLimits(value1, value2, offset_percent=0.05) -> (float, float):
    """
    Generates the bounds for a plot, such that both values are contained in the plot, with a certain offset.
    :param value1: lower bound
    :param value2: upper bound
    :param offset_percent: Offset in percent
    :return: (min, max) tuple
    """
    min_val = min(value1, value2)
    max_val = max(value1, value2)
    return min_val - offset_percent * (max_val - min_val), max_val + offset_percent * (max_val - min_val)


def arraytoAlphaVals(array: np.array, log=True) -> np.array:
    """ Converts a given array into alpha values [0.1,1] for each element in the array.
        The values are assumed to be on a log scale, and are normalized to the maximum value in the array.
        The minimum alpha returned is 0.1
    """
    if log:
        array = np.log(array)
    array = array - np.min(array)
    array = array / np.max(array)
    return 0.1 + 0.9*array

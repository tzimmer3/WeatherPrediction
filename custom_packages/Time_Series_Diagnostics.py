import numpy as np
import pandas as pd

import matplotlib.cm as cm
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

import math
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose


########################################
# ACF Plot
########################################


def acf_plot(data):
    """
    Define ACF plot with 95% statistical signficance level.
    Interpret: 

    Inputs:
    Data - Data to analyze with the ACF plot.

    Outputs:
    Auto Corrleation Function plot.
    """
    dataAcf = pd.DataFrame(acf(data))[1:]
    signLevel = 2 / math.sqrt(len(dataAcf))
    ax = sns.barplot(x=dataAcf.index[range(len(dataAcf))], y = dataAcf.iloc[:,0], color = "black",)
    plt.axhline(signLevel, color='r', linestyle = '--')
    plt.axhline(-signLevel, color='r', linestyle = '--')
    ax.set(xlabel='lag', ylabel='ACF')



########################################
# Seasonality Plot
########################################


#Define seasonplot function for Python using cufflinks and plotly
def season_plot(data, x_period):
    traces = []

    colors = cm.rainbow(np.linspace(0, 1, data.index.year.nunique()), bytes = True)
    colorsRgb = ["rgb(" + str(x[0]) + "," + str(x[1]) + "," + str(x[2]) + "," + str(x[3]) + ")" for x in colors]

    yearCount = 0
    for year in data.index.year.unique():
        if x_period == "month":
            x = data.index.month[data.index.year == year]
        elif x_period == "quarter":
            x = data.index.quarter[data.index.year == year]
        assert (x.nunique() == len(x)), "Non-unique y-values for each x-value.  Edit data or try increasing the granularity of x_period."

        y = data.iloc[:,0][data.index.year == year]
        traces.append(
            go.Scatter(
                x = x,
                y = y,
                mode = 'lines',
                connectgaps = True, 
                line = {
                        "color": colorsRgb[yearCount],
                        "width": 2
                       },
                name = year
            )
        )
        yearCount = yearCount + 1
    return go.Figure(data=traces)
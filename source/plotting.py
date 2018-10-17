"""
Utility functions for plotting. 
"""
import logging
import os

import numpy as np
import seaborn as sns


def prepare_sns(sns, params):
    """ Prepares seaborn for plotting according to the plot settings specified in
    params. 
    Args: 
        params (object) dictionary containing settings for each of the seaborn plots
    """
    sns.set_context('paper', font_scale=1) 
    sns.set(palette=tuple(getattr(params, "plot_palette", ["#E03C3F", "#FF9300", 
                                                           "#F8BA00", "#CB297B", 
                                                           "#6178A8", "#56C1FF"])),
            font=getattr(params, "plot_font", "Times New Roman"))
    sns.set_style(getattr(params, "plot_style", "ticks"),  
                  {'xtick.major.size': 5.0, 
                   'xtick.minor.size': 5.0, 
                   'ytick.major.size': 5.0, 
                   'ytick.minor.size': 5.0})

def plot_scatter(data, params):
    """
    """
    prepare_sns(sns, params)
    sns.scatterplot(x=data[0], y=data[1])



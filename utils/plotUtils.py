import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import math
import re
import calendar
import string
import pandas as pd
import bambi as bmb

from matplotlib import patches
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

from utils.plotParams import (
    FONT_SIZE_PLOT,
    getLabels,
    getPalettes,
    getLegends,
    WT_LABEL,
    ALPHA_LABEL,
    DELTA_LABEL,
    OMICRON_LABEL,
    DATE_RANGE_DOMINANT_WILDTYPE,
    DATE_RANGE_DOMINANT_ALPHA,
    DATE_RANGE_DOMINANT_DELTA,
    DATE_RANGE_DOMINANT_OMICRON,
)
from utils.regression import calcErrors, sampleProb, SEED
from utils.dataUtils import roundHalfUp, addJitterCol

PLOT_DIR = Path("..", "output", "figures")

LABEL = getLabels()
PALETTE = getPalettes()
LEGEND = getLegends()
PARAM_NAME_MAPPINGS = {
    "testDevice[1]": "Roche - Abbott",
    "immun2YN": LABEL.immun2YN,
    "variant[1]": f"{LABEL.variant}: {ALPHA_LABEL} - " f"{WT_LABEL}",
    "variant[2]": f"{DELTA_LABEL} - {WT_LABEL}",
    "variant[3]": f"{OMICRON_LABEL} - {WT_LABEL}",
    "gender[1]": "Male - Female",
    "binDaysPostOnset[1]": (f"{LABEL.binDaysPostOnset}: " f"2-7 days - 0-2 days"),
    "daysVariant[0]": (f"{LABEL.binDaysPostOnset}: " f"2-7 days - 0-2 days | Pre-VOC"),
    "daysVariant[1]": ("2-7 days - 0-2 days | Alpha"),
    "daysVariant[2]": ("2-7 days - 0-2 days | Delta"),
    "daysVariant[3]": ("2-7 days - 0-2 days | Omicron"),
    "daysImmun2YN[0]": ("Days post symptom onset: 2-7 days - 0-2 days | not immunised"),
    "daysImmun2YN[1]": ("2-7 days - 0-2 days | multiply immunised"),
    "zMaxSympDays[0]": LABEL.days,
    "zAge[0]": "Age",
    "zVl": LABEL.vl,
    "zVlVariant[0]": LABEL.vl + " | Pre-VOC",
    "zVlVariant[1]": LABEL.vl + " | Alpha",
    "zVlVariant[2]": LABEL.vl + " | Delta",
    "zVlVariant[3]": LABEL.vl + " | Omicron",
    "zVlImmun2YN[0]": (LABEL.vl + " | not immunised"),
    "zVlImmun2YN[1]": (LABEL.vl + " | multiply immunised"),
}


def returnPlotDir():
    return PLOT_DIR


def getXticks(nCats, nCatsHue, hue, dodge=True):
    """
    Return the positions for the x-tick labels if there is a factor (C{hue}) variable
    to stratify by.
    @param nCats: A C{int} specifying the number of categories for the main variable
    (i.e. the x-axis variable).
    @param nCatsHue: A C{int} specifying the number of categories for the factor
    variable (C{hue}).
    @param hue: A C{str} specifying the name of the factor variable.
    @param dodge: A C{bool} specifying whether the data corresponding to the
    categories in hue are dodged in the displayed plot.
    @return: A C{list} of C{floats} specifying the positions of the x-tick labels.
    """
    assert nCatsHue in (2, 3, 4), (
        f"Number of categories for variable {hue} " f"needs to be 2, 3 or 4."
    )
    if not dodge:
        return list(range(nCats))

    if nCatsHue == 2:
        xTicks = [[i - 0.2, i, i + 0.2] for i in range(nCats)]
    elif nCatsHue == 3:
        xTicks = [[i - 0.25, i, i + 0.0001, i + 0.25] for i in range(nCats)]
    else:
        xTicks = [[i - 0.32, i - 0.11, i, i + 0.11, i + 0.32] for i in range(nCats)]

    return [xTick for xTickSubset in xTicks for xTick in xTickSubset]


def getPointCoords(ax, scatter=True):
    """
    @param ax: A C{matplotlib} axes object containing a point plot.
    @return: The x and y coordinates of each point in the plot.
    """
    if scatter:
        x_coords = []
        y_coords = []
        for pointPair in ax.collections:
            for x, y in pointPair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)
    else:
        lines = ax.lines
        x_coords = list(lines[0].get_xdata())
        y_coords = list(lines[0].get_ydata())
        for line in lines[1:]:
            x_coords.append(list(line.get_xdata()[1]))
            y_coords.append(list(line.get_ydata()[1]))

    return x_coords, y_coords


def addNumbersWithoutHue(
    ax,
    colors,
    values=None,
    barplot=True,
    weight="bold",
    decPlaces=2,
    asPercent=False,
    centre=True,
    size=8,
):
    """
    Add values to each bar or point in a plot.
    @param ax: A C{matplotlib axes} object for plotting.
    @param colors: A C{list} or C{str} specifying the color in which count labels are
    shown.
    @param barplot: A C{bool} indicating whether the counts are to be added to a
    barplot (as opposed to a pointplot).
    @param values: An C{iterable} of values to plot instead of inferring them from
    the plot itself.
    @param weight: A C{str} specifying the annotation font type.
    @param centre: A C{bool} specifying whether to center the annotations.
    @param size: An C{int} specifying the font size of the annotations.
    """

    # This is to get rid of the warning message when converting masked values to nan.
    def maskedValueIdcs(coordList):
        idcs = []
        for i, coord in enumerate(coordList):
            if np.isnan(coord) or np.ma.is_masked(coord):
                idcs.append(i)
        return idcs

    if barplot:
        if isinstance(colors, str):
            colors = [colors] * len(ax.patches)
        for i, p in enumerate(ax.patches):
            if values:
                count = values[i]
            else:
                count = p.get_height()
            x = p.get_x() + p.get_width() / 2 - 0.1
            y = p.get_y() + p.get_height()

            if not centre:
                y += 0.3
            ax.annotate(
                f"{count:.0f}",
                (x + 0.1, y),
                size=size,
                weight=weight,
                color=colors[i],
                alpha=0.8,
            )

    else:
        xCoords, yCoords = getPointCoords(ax)
        if values:
            # When ax.errorbar() has been called previously, a [0, 0] coordinate gets
            # added to the plot (which we don't want to take into account).
            if len(values) != len(xCoords):
                xCoords = xCoords[1:]
            yCoords = values
        # Remove masked or NaN values.
        idxXMasked, idxYMasked = map(maskedValueIdcs, (xCoords, yCoords))
        assert idxXMasked == idxYMasked
        xCoords = [coord for i, coord in enumerate(xCoords) if i not in idxXMasked]
        yCoords = [coord for i, coord in enumerate(yCoords) if i not in idxYMasked]
        if isinstance(colors, str):
            colors = [colors] * len(xCoords)
        else:
            colors = [color for i, color in enumerate(colors) if i not in idxXMasked]

        for i, (x, y) in enumerate(zip(xCoords, yCoords)):
            if not centre:
                x += 0.01
                y -= 0.025
            if asPercent:
                annotateString = f"{int(roundHalfUp(y * 100, decimals=0))}%"
            else:
                annotateString = f"{y:.{decPlaces}f}".replace(".", "\u00B7")
            ax.annotate(
                annotateString,
                (x + 0.05, y),
                size=size,
                weight=weight,
                color=colors[i],
                alpha=0.8,
            )


def annotateWithLetter(ax, letter, size=20, coords=(-0.05, 1)):
    """
    Annotate a C{matplotlib} axes object with a letter in the upper left corner of
    the plot.
    @param ax: A C{matplotlib} axes object.
    @param letter: A C{str} specifying the letter to annotate C{ax} with.
    @param coords: The x, y coordinate to position the letter at.
    """
    x, y = coords
    ax.text(x, y, letter, transform=ax.transAxes, size=size, weight="bold")


def annotateWithLetters(axes, size=20, coords=(-0.05, 1)):
    """
    Annotate each C{matplotlib} axes object with a letter in alphabetical order.
    @param size: The font size of the letters.
    @param axes: An C{iterable} of C{matplotlib axes} objects.
    @param coords: For each figure, the x, y coordinate to position the respective
    letter at.
    """
    # Give each subplot a letter (A, B, C...).
    for letter, ax in zip(string.ascii_uppercase, axes):
        annotateWithLetter(ax, letter, size, coords)


def returnCountLabels(
    df,
    var,
    cats,
    subVar="agrdt",
    subVarCats=(0, 1),
    sep1="\n\n",
    sep2="\n",
    dodge=True,
    countTuple=False,
    showNCounts=True,
):
    """
    Return labels specifying counts for a plot with data stratified by 2 variables.
    @param df: A C{pd.DataFrame} containing the columns C{var} and C{subVar}.
    @param var: A C{str} specifying the main variable for which to return count labels.
    @param cats: A C{list} of C{str}s of the categories of the given variable C{var}
    by which to stratify, given in the display order, or a C{dict} mapping the
    categories to the corresponding C{str} to be displayed in the plot. The order
    of the keys specify the display order.
    @param subVar: A C{str} specifying a variable to stratify by.
    @param subVarCats: An C{iterable} of C{str}s specifying the categories of C{subVar}.
    @param sep1: A C{str} specifying how to separate the category C{str}s displayed
    in the x-ticklabels for C{var} from those of C{subVar}.
    @param sep2: A C{str} specifying the separator between categories in an x-ticklabel
    if C{dodge}==False.
    @param countTuple: A C{bool} specifying whether to display counts as tuples
    (if dodge==False).
    @param showNCounts: A C{bool} specifying whether to use the "n=" prefix for showing
    counts on the x-axis.
    @return: A C{list} of C{str} labels indicating data counts when stratifying by
    C{var} and C{subVar}.
    """

    # Return labels indicating the counts for each category present in a plot.
    if not isinstance(cats, dict):
        cats = dict(zip(cats, cats))

    def checkNoData(data):
        if data:
            return False
        return " "

    def createIntervalLabels(subCatDict, subCat, i):
        if not dodge and not countTuple:
            labels = f"{subVarCats[int(subCat)]}: {subCatDict[subCat][i]}"
        elif showNCounts:
            labels = f"n={subCatDict[subCat][i]:,}"
        else:
            labels = f"{subCatDict[subCat][i]:,}"

        return labels

    subCatDict = {}
    for subCat in subVarCats:
        subCatDict[subCat] = [
            ((df[var] == cat) & (df[subVar] == subCat)).sum() for cat in cats
        ]

    labelDict = {}
    for subCat in subVarCats:
        labelDict[subCat] = [
            checkNoData(subCatDict[subCat][i])
            or createIntervalLabels(subCatDict, subCat, i)
            for i, _ in enumerate(cats)
        ]
    labels = [
        [f"{label}" for label in labels]
        for labels in zip(*(labelDict[k] for k in labelDict))
    ]

    # Add category (cats) labels.
    for labelList, cat in zip(labels, cats):
        if dodge:
            labelList.insert(len(labelList) // 2, sep1 + cats[cat])

    # If dodge is False, we need to concatenate the labels in each list of labels.
    if dodge:
        newLabels = labels
    else:
        newLabels = []
        for labelList, cat in zip(labels, cats):
            if countTuple:
                newLabel = f'({", ".join(labelList)})' + sep1 + cats[cat]
                newLabels.append([newLabel])
            else:
                newLabels.append([sep2.join(labelList) + sep1 + cats[cat]])

    # Flatten label lists.
    finalLabelsList = [label for labelList in newLabels for label in labelList]

    return finalLabelsList


def boxNSwarmplot(
    df,
    x,
    y,
    xLabel,
    yLabel,
    xCats,
    dfSwarmplot=None,
    palBoxplot=None,
    palSwarmplot=None,
    legend=None,
    hueSwarmplot=None,
    hueBoxplot=None,
    hueOrder=None,
    dodge=True,
    countTuple=False,
    markersize=3,
    markeredgecolor=None,
    alphaBoxplot=1,
    alphaSwarmplot=1,
    ySwarmplot=None,
    axesFontSize=None,
    ticksFontSize=None,
    addCounts=False,
    rotateCountLabels=False,
    showNCounts=True,
    ax=None,
):
    """
    Create a boxplot with a swarmplot on top of it. Both the boxplot and the
    swarmplot are stratified by the variable given in C{hue}.

    @param df: A C{pd.DataFrame} containing columns "x" and "y".
    @param x: A C{str} specifying the x-variable.
    @param y: A C{str} specifying the y-variable.
    @param xLabel: A C{str} specifying the x-axis label.
    @param yLabel: A C{str} specifying the y-axis label.
    @param xCats: A C{list} of C{str}s of the categories of C{x} by which to stratify,
    given in the display order or a C{dict} mapping the categories to the
    corresponding C{str} to be displayed in the plot. The order of the keys specify
    the display order.
    @param dfSwarmplot: A second C{pd.DataFrame} used for making the swarmplot (e.g. if
    there are too many data points for plotting in one category of the original data
    frame C{df}).
    @param palBoxplot: A C{dict} specifying the color mapping for each category in xCats
    or hue in the boxplot. Alternatively a C{list} or C{tuple} specifying the color
    of each box individually. If set to 'none', no color will be used.
    @param palSwarmplot: A C{dict} specifying the color mapping for each
    category in xCats or hue in the swarmplot.
    @param legend: A C{matplotlib.artist} object of the legend to be shown in the plot.
    @param hueSwarmplot: A C{str} specifying which variable to use for stratifying
    the swarmplot.
    @param hueBoxplot: A C{str} specifying which variable to use for stratifying the
    boxplot.
    @param hueOrder: A C{iterable} specifying the display order of the categories
    specified by hue.
    @param dodge: A C{bool} indicating whether to dodge data points in the swarmplot
    that are stratified wrt C{hueSwarmplot}.
    @param countTuple: A C{bool} specifying whether to display counts as tuples
    (if dodge==False).
    @param markeredgecolor: A C{str} specifying the color for the markeredge of a dot in
    the swarmplot.
    @param alphaBoxplot: A C{float} specifying the opacity of the boxplot.
    @param alphaSwarmplot: A C{float} specifying the opacity of the markers in the
    swarmplot.
    @param ySwarmplot: A C{str} specifying the y-variable used in the swarmplot. This
    can be useful if we want to plot jittered data but still want to show the
    original stats in the boxplot.
    @param axesFontSize: A C{int} specifying the font size of the axis labels.
    @param ticksFontSize: A C{int} specifying the font size of the tick labels.
    @param addCounts: A C{bool} specifying whether to add counts to x-tick labels.
    @param rotateCountLabels: A C{bool} specifying whether to rotate the count labels
    on the x-axis.
    @param showNCounts: A C{bool} specifying whether to use the "n=" prefix for showing
    counts on the x-axis.
    @param ax: A C{matplotlib axes} object for plotting.
    @return: The C{matlotlib axes} object used for plotting.
    """

    return boxNStripOrSwarmplot(
        df,
        x,
        y,
        xLabel,
        yLabel,
        xCats,
        dfStripOrSwarmplot=dfSwarmplot,
        stripplot=False,
        palBoxplot=palBoxplot,
        palStripOrSwarmplot=palSwarmplot,
        legend=legend,
        hueStripOrSwarmplot=hueSwarmplot,
        hueBoxplot=hueBoxplot,
        hueOrder=hueOrder,
        dodge=dodge,
        countTuple=countTuple,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        alphaBoxplot=alphaBoxplot,
        alphaStripOrSwarmplot=alphaSwarmplot,
        yStripOrSwarmplot=ySwarmplot,
        axesFontSize=axesFontSize,
        ticksFontSize=ticksFontSize,
        addCounts=addCounts,
        rotateCountLabels=rotateCountLabels,
        showNCounts=showNCounts,
        ax=ax,
    )


def boxNStripOrSwarmplot(
    df,
    x,
    y,
    xLabel,
    yLabel,
    xCats,
    dfStripOrSwarmplot=None,
    stripplot=True,
    palBoxplot=None,
    palStripOrSwarmplot=None,
    legend=None,
    hueStripOrSwarmplot=None,
    hueBoxplot=None,
    hueOrder=None,
    dodge=True,
    countTuple=False,
    markersize=3,
    markeredgecolor=None,
    alphaBoxplot=1,
    alphaStripOrSwarmplot=1,
    yStripOrSwarmplot=None,
    axesFontSize=None,
    ticksFontSize=None,
    addCounts=False,
    rotateCountLabels=False,
    showNCounts=True,
    ax=None,
):
    """
    Create a boxplot with a stripplot or swarmplot on top of it. Both the boxplot and
    the strip/swarmplot are stratified by the variable given in C{hue}.
    @param df: A C{pd.DataFrame} containing columns "x" and "y".
    @param x: A C{str} specifying the x-variable.
    @param y: A C{str} specifying the y-variable.
    @param xLabel: A C{str} specifying the x-axis label.
    @param yLabel: A C{str} specifying the y-axis label.
    @param xCats: A C{list} of C{str}s of the categories of C{x} by which to stratify,
    given in the display order or a C{dict} mapping the categories to the
    corresponding C{str} to be displayed in the plot. The order of the keys specify
    the display order.
    @param dfStripOrSwarmplot: A second C{pd.DataFrame} used for making the strip or
    swarmplot (e.g. if there are too many data points for plotting in one category
    of the original data frame C{df}).
    @param stripplot: A C{bool} specifying whether to make a stripplot (as opposed to
    a swarmplot).
    @param palBoxplot: A C{dict} specifying the color mapping for each category in xCats
    or hue in the boxplot. Alternatively a C{list} or C{tuple} specifying the color
    of each box individually. If set to 'none', no color will be used.
    @param palStripOrSwarmplot: A C{dict} specifying the color mapping for each
    category in xCats or hue in the strip or swarmplot.
    @param legend: A C{matplotlib.artist} object of the legend to be shown in the plot.
    @param hueStripOrSwarmplot: A C{str} specifying which variable to use for
    stratifying the strip or swarmplot.
    @param hueBoxplot: A C{str} specifying which variable to use for stratifying the
    boxplot.
    @param hueOrder: A C{iterable} specifying the display order of the categories
    specified by hue.
    @param dodge: A C{bool} indicating whether to dodge data points in the strip or
    swarmplot that are stratified wrt C{hueStripOrSwarmplot}.
    @param countTuple: A C{bool} specifying whether to display counts as tuples
    (if dodge==False).
    @param markeredgecolor: A C{str} specifying the color for the markeredge of a dot in
    the swarmplot.
    @param alphaBoxplot: A C{float} specifying the opacity of the boxplot.
    @param alphaSwarmplot: A C{float} specifying the opacity of the markers in the
    swarmplot.
    @param yStripOrSwarmplot: A C{str} specifying the y-variable used in the strip or
    swarmplot. This can be useful if we want to plot jittered data but still want
    to show the original stats in the boxplot.
    @param axesFontSize: A C{int} specifying the font size of the axis labels.
    @param ticksFontSize: A C{int} specifying the font size of the tick labels.
    @param addCounts: A C{bool} specifying whether to add counts to x-tick labels.
    @param rotateCountLabels: A C{bool} specifying whether to rotate the count labels
    on the x-axis.
    @param showNCounts: A C{bool} specifying whether to use the "n=" prefix for showing
    counts on the x-axis.
    @param ax: A C{matplotlib axes} object for plotting.
    @return: The C{matlotlib axes} object used for plotting.
    """
    assert hueStripOrSwarmplot == hueBoxplot or hueStripOrSwarmplot and not hueBoxplot
    dodge = False if not hueStripOrSwarmplot else dodge

    def returnOrder(data, x, xCats):
        # It seems that swarmplot is rather strict about the data type (float or int)
        # given in the order argument. So we are checking which datatype is in the
        # corresponding x column of the given data frame so that it corresponds to what
        # is in "order".
        if x:
            if data[x].dtype == "float64":
                order = [float(cat) for cat in xCats]
            elif data[x].dtype == "int64":
                order = [int(cat) for cat in xCats]
            else:
                order = list(xCats)
            return order
        else:
            return xCats

    # We need to drop all data points where y is NaN to get the correct counts.
    df = df.dropna(subset=[y])

    if isinstance(xCats, list):
        xCats = dict(zip(xCats, xCats))

    plotFunc = sns.stripplot if stripplot else sns.swarmplot
    linewidth = 1 if markeredgecolor else 0

    if palBoxplot == "none":
        boxprops = dict(alpha=alphaBoxplot, facecolor="none")
        palBoxplot = None
    else:
        boxprops = dict(alpha=alphaBoxplot)

    # Create boxplot (don't show outliers as they are shown by the swarmplot
    # already).
    bp = sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hueBoxplot,
        palette=palBoxplot,
        order=xCats,
        hue_order=hueOrder,
        boxprops=boxprops,
        showfliers=False,
        ax=ax,
    )
    # If palBoxplot is a C{list} of colors, set the boxes in the plot individually.
    if isinstance(palBoxplot, (list, tuple)):
        axPatches = [
            patch for patch in ax.patches if isinstance(patch, patches.PathPatch)
        ]
        assert len(axPatches) == len(palBoxplot)
        for patch, color in zip(axPatches, palBoxplot):
            patch.set_facecolor(color)

    # Create strip or swarmplot.
    yStripOrSwarmplot = yStripOrSwarmplot or y
    data = df if dfStripOrSwarmplot is None else dfStripOrSwarmplot

    order = returnOrder(data, x, xCats)
    hueOrder = returnOrder(data, hueStripOrSwarmplot, hueOrder)

    kwargsStripOrSwarmplot = {
        "data": data,
        "x": x,
        "y": yStripOrSwarmplot,
        "hue": hueStripOrSwarmplot,
        "dodge": dodge,
        "size": markersize,
        "order": order,
        "hue_order": hueOrder,
        "edgecolor": markeredgecolor,
        "linewidth": linewidth,
        "alpha": alphaStripOrSwarmplot,
        "ax": ax,
    }
    if palStripOrSwarmplot:
        kwargsStripOrSwarmplot["palette"] = palStripOrSwarmplot
    else:
        kwargsStripOrSwarmplot["color"] = "k"

    sp = plotFunc(**kwargsStripOrSwarmplot)

    if hueStripOrSwarmplot is None or hueBoxplot is None:
        if addCounts:
            labels = [
                f"{xCatLabel}\n(n={(df[x] == xCat).sum()})"
                for xCat, xCatLabel in xCats.items()
            ]
        else:
            labels = [xCats[cat] for cat in order]
    else:
        assert hueOrder, 'You must provide an order of the hue categories ("hueOrder")!'
        if addCounts:
            sep1 = "\n\n\n\n" if rotateCountLabels else "\n\n"
            labels = returnCountLabels(
                df,
                x,
                xCats,
                subVar=hueStripOrSwarmplot,
                subVarCats=hueOrder,
                dodge=dodge,
                countTuple=countTuple,
                sep1=sep1,
                showNCounts=showNCounts,
            )
        else:
            # We only want to display the labels for the main x categories, otherwise
            # the labelling becomes to cluttered.
            nCatsHue = len(hueOrder)
            if nCatsHue == 2:
                labels = [["", xCats[cat], ""] for cat in order]
            elif nCatsHue == 3:
                labels = [["", "", xCats[cat], ""] for cat in order]
            elif nCatsHue == 4:
                labels = [["", "", xCats[cat], "", ""] for cat in order]
            # Flatten the list.
            labels = [label for labelGroup in labels for label in labelGroup]
        xTicks = getXticks(len(xCats), len(hueOrder), hueStripOrSwarmplot, dodge=dodge)
        ax.set_xticks(xTicks)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    xticks = ax.xaxis.get_major_ticks()
    for xtick in xticks:
        # Hide the tick marks for non-count labels.
        if not "=" in xtick.label.get_text() or xtick == "none":
            xtick.tick1line.set_visible(False)

    if rotateCountLabels:
        for label in ax.xaxis.get_ticklabels():
            if "n=" in label.get_text():
                label.set_rotation(45)
                label.set_ha("right")

    # Adjust font size.
    if axesFontSize:
        ax.xaxis.label.set_size(axesFontSize)
        ax.yaxis.label.set_size(axesFontSize)
    # Adjust font size of tick labels.
    if ticksFontSize:
        ax.tick_params(axis="both", which="major", labelsize=ticksFontSize)
        ax.tick_params(axis="both", which="minor", labelsize=ticksFontSize)
    # Set legend.
    if legend:
        if legend == "none":
            ax.get_legend().remove()
        else:
            replaceLegend(ax, legend, loc="upper right")

    return bp, sp


def computeTicks(x, step=5):
    """
    Compute positions of x-ticks based on step size (C{step}).
    @param x: A list-like object of integers or floats.
    @param step: Tick frequency (with respect to C{x}).
    """
    xMax, xMin = math.ceil(max(x)), math.floor(min(x))
    dMax, dMin = (
        xMax + abs((xMax % step) - step) + (step if (xMax % step != 0) else 0),
        xMin - abs((xMin % step)),
    )
    return range(dMin, dMax, step)


def hideGridX(ax):
    ax.grid(False, axis="x")


def hideGridY(ax):
    ax.grid(False, axis="y")


def plotMeansNErrors(
    xs, ys, yerrs, colors, ax, markers=None, markersize=20, fontsize=FONT_SIZE_PLOT
):
    """
    Create a plot showing means (with annotated values) and corresponding error bars.
    @param xs: An C{iterable} of x-coordinates.
    @param ys: An C{iterable} of y-coordinates.
    @param yerrs: An C{iterable} of errors (as pairs of lower and upper limit) to plot
    along the y-axis.
    @param colors: An C{iterable} of colors for each value (of length len(xs)).
    @param ax: A C{matplotlib axes} object used for plotting.
    @param markers: An C{iterable} of markers (e.g. 'o') to use for plotting the mean
    values.
    @param markersize: A C{int} specifying the size of the markers.
    @param fontsize: A C{int} specifying the size of the font.
    """
    assert len(xs) == len(ys) == len(colors) == len(yerrs)
    if markers and len(markers) > 1:
        assert len(xs) == len(markers)
        for x, y, color, marker in zip(xs, ys, colors, markers):
            ax.scatter(x, y, color=color, marker=marker, s=markersize)
    elif markers:
        ax.scatter(xs, ys, color=colors, marker=markers, s=markersize)
    else:
        ax.scatter(xs, ys, color=colors, s=markersize)
    # Add mean values to the plot.
    addNumbersWithoutHue(
        ax,
        colors=colors,
        values=None,
        barplot=False,
        weight="normal",
        decPlaces=2,
        asPercent=True,
        size=fontsize,
    )
    # Add errors.
    for i, (x, yerr) in enumerate(zip(xs, yerrs)):
        lower, upper = yerr
        ax.vlines(x=x, ymin=lower, ymax=upper, color=colors[i])


def _addErrors(errors, cats1, cats2, palette, y_coords, x_coords, ax):
    """
    Add the errorbars (94% credible intervals) calculated by MCMC sampling to the plot.
    @param errors: A C{list} of C{np.array}s, each containing the lower and upper limit
    of a 94% credible interval for a corresponding point in the plot.
    @param cats1: The categories of variable C{var} which the errors were computed for.
    @param cats2: The categories of the second variable ("var2" used for further
    stratification) which the errors were computed for.
    @param palette: A C{list} or C{dict} specifying the color used for each datapoint.
    @param y_coords: Am {np.array} containing the y-coordinates of the points in the
    plot.
    @param x_coords: An {np.array} containing the x-coordinates of the points in the
    plot.
    @param ax: A C{matplotlib axes} object with errorbars added.
    """
    # Flatten list and turn into (2, n) dimensional array. Leave out errors with value 0
    # (no point in pointplot).
    errors = np.stack(
        np.array([error for errorList in errors for error in errorList]), axis=1
    )
    colors = []
    if cats2 is None:
        if isinstance(palette, list):
            colors.extend(palette[: len(cats1)])
        elif isinstance(palette, dict):
            colors.extend(palette[cat] for cat in cats1)
        else:  # C{palette} is string specifying the color.
            colors.extend(palette for _ in cats1)
    else:
        for i, cat2 in enumerate(cats2):
            if isinstance(palette, list):
                colors.extend([palette[i]] * len(cats1))
            elif isinstance(palette, dict):
                colors.extend([palette[cat2]] * len(cats1))
            else:  # C{palette} is string specifying the color.
                colors.extend([palette] * len(cats1))

    yerrLower, yerrUpper = abs(y_coords - errors)

    # Sometimes ci intervals don't cover 1.0 or 0.0 which leads to error bars exceeding
    # 1 or 0. Cut these off.
    yerrLower = [
        yerr if y_coord != 0 or np.isnan(yerr) else 0
        for yerr, y_coord in zip(yerrLower, y_coords)
    ]
    yerrUpper = [
        yerr if y_coord != 1 else 0 for yerr, y_coord in zip(yerrUpper, y_coords)
    ]
    yerr = [yerrLower, yerrUpper]

    ax.errorbar(
        x_coords, y_coords, yerr=yerr, ecolor=colors, fmt=" ", elinewidth=1.3, alpha=0.8
    )


def pointPlotNErrors(
    x,
    y,
    df,
    statsDict,
    palette,
    xorder,
    xTicklabels,
    onlyHDI=False,
    join=False,
    seed=10,
    asPercent=False,
    ax=None,
):
    """
    Create a point plot with error bars estimated using Bayesian MCMC. Returns a
    C{tuple} of calculated means.
    @param x: A C{str} specifying the variable appearing on the x-axis.
    @param y: A C{str} specifying the variable appearing on the y-axis.
    @param df: A C{pd.DataFrame} containing columns "x" and "y".
    @param statsDict: A nested C{defaultdict} with three levels with the following
    keys on each level:
    level 1: The variable (C{var}) of interest, e.g. "symptoms",
    level 2: The corresponding categories (C{cats}) of interest, e.g. 0 or 1.
    level 3: The posterior samples of probabilities ("prob"), e.g. for a positive
    AgRDT result or PCR positivity, the corresponding 95% credible interval ("hdi")
    and the corresponding mean ("mean"). Thus the probabilities correspond to
    estimates for AgRDT sensitivity or PCR positive rate.
    @param palette: A C{dict} specifying the color mapping for each category in x.
        Alternatively a list of colors or a single RGBA tuple specifying a color to
        use for each category can be passed.
    @param xorder: A C{iterable} specifying the display order of categories in x.
    @param xTicklabels: An C{iterable} of x-tick labels to display.
    @param onlyHDI: A C{bool} specifying wheter to only add the HDI (not the means).
    @param join: A C{bool} indicating whether to join the dots in the scatter plot by
        lines.
    @param ax: A C{matplotlib} axes for plotting.
    @return: A C{tuple} specifying the means calculated by Bayesian MCMC (in the order
        of C{xorder}) if C{onlyHDI}==False, else the y-coordinates as inferred from
        the given plot.
    """

    # Calculate the type of error to plot as the error bars.
    # Make sure the order is the same as the one the points were looped over.
    errors = defaultdict(list)
    means = defaultdict(list)
    calcErrors(df, statsDict, y, errors, means, x, xorder, seed)
    errors = [errors[cat] for cat in xorder]
    if isinstance(palette, dict):
        colors = np.array([palette[cat] for cat in xorder], object)
    elif isinstance(palette, list):
        colors = palette
    else:  # palette is a string specifying a color.
        colors = [palette] * len(xorder)
        palette = colors

    if onlyHDI:
        x_coords, y_coords = getPointCoords(ax)
    else:
        x_coords = list(range(len(xorder)))
        y_coords = [mean for cat in xorder for mean in means[cat]]
    _addErrors(errors, xorder, None, palette, y_coords, x_coords, ax)
    if not onlyHDI:
        ax.scatter(x_coords, y_coords, c=colors, s=10, alpha=0.8)
        if join:
            # Plot lines between points (skip NaNs).
            ax.plot(
                [x for x, y in zip(x_coords, y_coords) if not np.isnan(y)],
                [y for y in y_coords if not np.isnan(y)],
                c=colors[0],
                alpha=0.8,
            )
    if xTicklabels:
        if xTicklabels == "none":
            xTicklabels = []
        ax.set_xticks(x_coords)
        ax.set_xticklabels(labels=xTicklabels)

    if asPercent:
        labelsPercent = [
            100 * float(label.get_text()) for label in ax.get_yticklabels()
        ]
        ax.set_yticklabels(list(map(int, labelsPercent)))

    return y_coords


def pointPlotNErrorsNMeans(
    x, y, df, statsDict, pal, order, xTicklabels, join, size, asPercent, ax
):
    """
    Create point plot with error bars estimated using Bayesian MCMC and
    annotations in the plot specifying the means.
    @param x: A C{str} specifying the variable appearing on the x-axis.
    @param y: A C{str} specifying the variable appearing on the y-axis.
    @param df: A C{pd.DataFrame} containing columns "x" and "y".
    @param statsDict: A nested C{defaultdict} with three levels with the following
        keys on each level:
        level 1: The variable (C{var}) of interest, e.g. "symptoms",
        level 2: The corresponding categories (C{cats}) of interest, e.g. 0 or 1.
        level 3: The posterior samples of probabilities ("prob"), e.g. for a positive
        Ag-RDT result or PCR positivity, the corresponding 95% credible interval
        ("hdi") and the corresponding mean ("mean"). Thus the probabilities correspond
        to estimates for AgRDT sensitivity or PCR positive rate.
    @param pal: A C{dict} specifying the color mapping for each category in x.
        Alternatively a list of colors or a single RGBA tuple specifying a color to
        use for each category can be passed.
    @param order: A C{iterable} specifying the display order of categories in x.
    @param xTicklabels: An C{iterable} of x-tick labels to display.
    @param join: A C{bool} indicating whether to join the dots in the scatter plot by
        lines.
    @param size: An C{int} specifying the font size of the value labels.
    @param ax: A C{matplotlib} axes for plotting.
    """
    means = pointPlotNErrors(
        x,
        y,
        df,
        statsDict,
        palette=pal,
        xorder=order,
        xTicklabels=xTicklabels,
        asPercent=asPercent,
        join=join,
        ax=ax,
    )

    if isinstance(pal, dict):
        colors = list(pal.values())
    elif isinstance(pal, list):
        colors = pal
    else:  # palette is a string specifying a color.
        colors = [pal] * len(order)
    addNumbersWithoutHue(
        ax,
        colors=colors,
        values=means,
        barplot=False,
        weight="normal",
        decPlaces=2,
        asPercent=asPercent,
        size=size,
    )
    return means


def reformatLegend(ax, legend, label, loc, ncol=2, border=True, fontSizePlot=12):
    framealpha = 0.7 if border else 0
    ax.legend(
        handles=legend[1:],
        ncol=ncol,
        title=label,
        columnspacing=1,
        borderaxespad=0.2,
        loc=loc,
        fontsize=fontSizePlot,
        title_fontsize=fontSizePlot,
        handleheight=0.5,
        handlelength=1.3,
        framealpha=framealpha,
    )


def removeLegend(ax):
    ax.legend_.remove()


def replaceLegend(ax, handles, loc="upper left"):
    """
    Replace the current legend in ax with a custom one.
    @param ax: A C{matplotlib} axes for plotting.
    @param handles: The handles you want to appear in the legend.
    @param loc: The location you want the legend to appear at.
    """
    try:  # Sometimes there is no legend.
        ax.legend_.remove()
    except AttributeError:
        pass
    ax.legend(handles=handles, loc=loc)


def addHDIlinesToRidgePlot(ax, iData, varNames, hdi=(0.03, 0.97)):
    """
    Add highest posterior density intervals as horizontal lines to a ridge plot.
    @param ax: A C{matplotlib} axes containing a ridge plot representing the posterior
        distributions of paramater values estimated by MCMC.
    @param iData: An C{arviz.InferenceData} object.
    @param varNames: An C{iterable} specifying the variables (in the desired order)
        for which to add HDI lines to the plot.
    @param hdi: A C{tuple} specifying the HDI to plot.
    @param omit: A C{iterable} of variables to not show in the plot.
    """
    color = "black"
    i = 0
    yTicks = ax.get_yticks()[::-1]
    post = iData.posterior
    iDataList = list(az.sel_utils.xarray_sel_iter(post, combined=True))
    iDataListOrdered = []
    for varName in varNames:
        for coord in iDataList:
            if varName in coord:
                iDataListOrdered.append(coord)

    for var, coords, _ in iDataListOrdered:
        yCoord = yTicks[i] - 0.2
        varPost = post[var].sel(coords).values.flatten()
        mean = varPost.mean()
        hdi50 = np.quantile(varPost, (0.25, 0.75))
        hdiPassed = np.quantile(varPost, hdi)

        ax.fill_between(
            hdiPassed, yCoord - 0.03, yCoord + 0.03, color=color, alpha=0.8, linewidth=0
        )
        ax.fill_between(
            hdi50, yCoord - 0.05, yCoord + 0.05, color=color, alpha=0.8, linewidth=0
        )
        ax.plot(
            mean,
            yCoord,
            marker="o",
            markersize=3.5,
            markeredgewidth=0.5,
            markeredgecolor="white",
            markerfacecolor=color,
        )
        i += 1


def ridgeForestPlot(
    iData,
    varNames=None,
    yTickLabels=None,
    figsize=(10, 10),
    fontsize=10,
    xlim=None,
    ax=None,
):
    """
    Make a mixture of a ridge and forest plot for displaying posterior distributions
    of parameters as determined by an MCMC run.
    @param iData: An C{arviz.InferenceData} object.
    @param varNames: An C{iterable} specifying the variables (in the desired order)
        whose posterior distributions should be plotted.
    @param yTickLabels: An C{iterable} of y-axis ticklabels to use.
    @param figsize: A C{tuple} specifying the size of the output figure.
    @param fontsize: The size of the font to appear in the plot.
    @param xlim: A C{tuple} or C{None} specifying which (if any) limits to use for
        the x-axis.
    @param ax: The C{matplotlib} axes to use for plotting.
    @return: A C{matplotlib.pyplot.axis} object containing a mix of a ridge and
        a forest plot for displaying posterior distributions.
    """
    if not varNames:
        varNames = (
            "testDevice",
            "vaccYN7",
            "variant",
            "gender",
            "binDaysPostOnset",
            "zAge",
            "zVl",
        )

    # Make a ridgeplot.
    if not yTickLabels:
        yTickLabels = []
        for varName in varNames:
            varName = varName.replace("|", r"\|")  # This gets interpreted
            # wrongly in the regex otherwise.
            if varName == "zVl":
                varName = "zVl$"
            for paramKey, displayLabel in PARAM_NAME_MAPPINGS.items():
                if re.match(rf"{varName}(?!\|)", paramKey):
                    yTickLabels.append(displayLabel)
        assert yTickLabels, (
            f"None of the provided variable names "
            f'({", ".join(varNames)!r} are recognized.'
        )
    yTickLabels = yTickLabels[::-1]  # Have to be reversed because of how plotting is
    # done.

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    az.plot_forest(
        iData,
        hdi_prob=1,
        combined=True,
        var_names=list(varNames),
        kind="ridgeplot",
        ridgeplot_truncate=False,
        ridgeplot_overlap=0.55,
        ridgeplot_alpha=0.3,
        markersize=3,
        colors="blue",
        ax=ax,
    )
    leftLim, rightLim = xlim or ax.get_xlim()
    xTicks = []
    step = 4 if abs(leftLim) >= 10 or abs(rightLim) >= 10 else 2
    xTickRange = range(-12, 13, step)
    for x in xTickRange:
        if leftLim < x < rightLim:
            linewidth = 0.15 if x else 0.5
            ax.vlines(x, ymin=-1, ymax=18, color="gray", linewidth=linewidth)
            xTicks.append(x)
    # Add 94% HDI lines as we know them from forest plots.
    addHDIlinesToRidgePlot(ax, iData, varNames=varNames)
    ax.set_yticklabels(yTickLabels)
    ax.set_xticks(xTicks)
    ax.set_xlabel("Estimated parameter size")
    ax.set_xlim(leftLim, rightLim)
    ax.set_xticklabels(xTicks)
    ax.tick_params(axis="y", which="both", labelleft=False, labelright=True, size=10)
    setFontSize(ax, size=fontsize)

    plt.tight_layout()

    return ax


def setFontSize(ax, size=15):
    """
    Change the font size of all texts in a C{matplolib} axis.
    @param ax: A C{matplotlib} axes for plotting.
    @param size: The font size you want the text in ax to have.
    """
    texts = (
        [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    )

    legend = ax.get_legend()
    if legend:
        texts.append(legend.get_texts())
        texts.append(legend.get_title())

    for text in texts:
        if isinstance(text, list):
            for ele in text:
                ele.set_fontsize(size)
        else:
            text.set_fontsize(size)

    ax.title.set_fontsize(size + 2)


def _levellingAgrdt(row, var="variant"):
    # Utility function for plotting datapoints on separate levels in the spaghetti plot.
    order = row[var]
    gapSize = 0.02
    times = -1 if not row["agrdt"] else 1
    return int(row["agrdt"]) + times * gapSize + order * times * gapSize


def spaghettiPlotCategorical(
    ax,
    feature,
    featureLevels,
    predDict,
    df,
    legendLoc=(0.5, 0.15),
    palette=None,
    legend=None,
    fontsize=FONT_SIZE_PLOT,
):
    colors = (
        list(palette.values()) if palette else list(getattr(PALETTE, feature).values())
    )
    handles = legend or getattr(LEGEND, f"{feature}Lines")
    newData = predDict[feature]["data"]
    posteriorChainMean = predDict[feature]["ppSamples"]
    vl = predDict[feature]["vl"]
    randIdcs = np.random.choice(posteriorChainMean.shape[1], size=500)

    for i, featureLevel in enumerate(featureLevels):
        # Which rows in new_data correspond to "feature"?
        idx = newData.index[newData[feature] == featureLevel].tolist()
        ax.plot(vl, posteriorChainMean[idx][:, randIdcs], alpha=0.03, color=colors[i])
        ax.plot(
            vl,
            posteriorChainMean[idx].mean(axis=1),
            alpha=1,
            linewidth=2,
            color=colors[i],
        )

    ax.set_ylabel(f"P(Ag-RDT=positive | {LABEL.vl})")
    ax.set_xlabel(LABEL.vl)
    yMargin = 0.02 * len(featureLevels) + 0.02
    ax.set_ylim(-yMargin, 1 + yMargin)
    ax.set_xlim(3, 11)
    ax.legend(handles=handles)

    # Add actual data points.
    dfCopy = df.copy()
    dfCopy["agrdtLevels"] = dfCopy.apply(
        lambda row: _levellingAgrdt(row, feature), axis=1
    )

    sns.scatterplot(
        data=dfCopy,
        x="vl",
        y="agrdtLevels",
        hue=feature,
        palette=dict(zip(range(len(featureLevels)), colors)),
        s=20,
        alpha=0.5,
        zorder=1,
        ax=ax,
    )
    ax.legend(handles=handles, loc=legendLoc)
    setFontSize(ax, fontsize)


def variantDominantTimesNumeric(df, timeVar="samplingMonth2"):
    """
    Return date ranges during which SARS-CoV-2 variants were dominant as numeric values,
    indicating start points, end points and distances along the x-axis (representing
    time) --> see Fig 1D.
    """
    preliminaryMonthDict = (
        df.groupby(timeVar)["samplingMonth"].unique().agg(list).to_dict()
    )

    monthDict = defaultdict(list)
    for timeBin, samplingMonths in preliminaryMonthDict.items():
        samplingMonths = sorted(
            list(map(lambda dt: datetime.strptime(dt, "%Y-%m").date(), samplingMonths))
        )
        endMonth, endYear = samplingMonths[-1].month, samplingMonths[-1].year
        # Add the start date of the sampling period.
        # Add the end date of the sampling period.
        monthDict[timeBin].append(samplingMonths[0])
        monthDict[timeBin].append(
            date(endYear, endMonth, calendar.monthrange(endYear, endMonth)[-1])
        )
    # Add last date of data collection.
    studyStart, studyEnd = date(2020, 12, 1), date(2022, 2, 11)
    monthDict[max(monthDict)][-1] = studyEnd

    dateRanges = [
        DATE_RANGE_DOMINANT_WILDTYPE,
        DATE_RANGE_DOMINANT_ALPHA,
        DATE_RANGE_DOMINANT_DELTA,
        DATE_RANGE_DOMINANT_OMICRON,
    ]

    # DATE_RANGE_DOMINANT_WILDTYPE starts way before our project, therefore, we set the
    # start date to the the start date of the actual time period we are looking at (
    # i.e., to 1.12.2020).
    dateRanges[0] = studyStart, DATE_RANGE_DOMINANT_WILDTYPE[1]
    # Do the same for the Omicron date range but with the end point.
    dateRanges[-1] = DATE_RANGE_DOMINANT_OMICRON[0], studyEnd

    startPoints = []
    endPoints = []

    for variantStart, variantEnd in dateRanges:
        i = 0
        for start, end in monthDict.values():
            if start <= variantStart <= end:
                startRatio = (variantStart - start).days / (end - start).days
                startPoints.append(i - 0.5 + startRatio)

            if start <= variantEnd <= end:
                endRatio = (variantEnd - start).days / (end - start).days
                endPoints.append(i - 0.5 + endRatio)

            i += 1

    assert len(startPoints) == len(endPoints)
    durations = [abs(end - start) for start, end in zip(startPoints, endPoints)]

    return startPoints, endPoints, durations


def boldStr(text):
    """
    @param text: A C{str}.
    @return: text in a bold font.
    """
    return r"$\bf{" + str(text) + "}$"


def plotVariantDominantTimes(
    dfPos, palette, abbrvDictPaper, ax, timeVar="samplingMonth2", fontsize=12
):
    xlim = (-0.5, 7.5) if timeVar == "samplingMonth2" else (-0.5, 14.5)
    # Plot variant dominant times.
    startPoints, endPoints, durations = variantDominantTimesNumeric(
        dfPos, timeVar=timeVar
    )
    ax.barh(
        (0, 0, 0, 0),
        durations,
        left=startPoints,
        color=palette.variant.values(),
        zorder=-1,
    )

    # Add variant labels on top.
    xTicks = [
        startPoint + duration / 2
        for startPoint, duration in zip(startPoints, durations)
    ]
    ax_2 = ax.twiny()
    ax_2.set_xlim(startPoints[0], endPoints[-1])
    ax_2.set_xticks(xTicks)
    ax_2.set_xticklabels(abbrvDictPaper["variant"].values())
    # Set colors of x-ticklabels.
    for i, color in enumerate(palette.variant.values()):
        ax_2.get_xticklabels()[i].set_color(color)
    # Remove xticks.
    ax.xaxis.set_ticks_position("none")
    ax_2.xaxis.set_ticks_position("none")
    # Remove gridlines.
    hideGridY(ax)
    hideGridX(ax_2)
    hideGridY(ax_2)

    ax.set_yticklabels("")
    ax.set_xticklabels("")
    ax.set_xlim(*xlim)
    ax_2.set_xlim(*xlim)

    setFontSize(ax_2, fontsize)


def plotFig1_A(
    dfFigure1_A,
    dfPos,
    statsDictTime,
    order,
    label,
    palette,
    abbrvDictPaper,
    ax,
    fontSizePlot=12,
):
    iData, model = None, None
    ax1, ax2 = ax
    # Antigen test sensitivities.
    if "samplingMonth2" not in statsDictTime:
        n = 1250
        # There is very little data for category 3, so we skip it
        newData = pd.DataFrame(
            {
                "samplingMonth2": np.concatenate(
                    list(
                        np.repeat(months, n)
                        for months in order.samplingMonth2
                        if months != 3
                    )
                )
            }
        )
        formula = "agrdt ~ (1|samplingMonth2)"
        priors = {
            "1|samplingMonth2": bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=2)
            )
        }
        model, iData = sampleProb(
            df=dfFigure1_A,
            outcome="agrdt",
            catVars=["samplingMonth2"],
            seed=SEED,
            target_accept=0.99,
            bambi=True,
            formula=formula,
            priors=priors,
            newData=newData,
        )

        for samplingMonth2 in order.samplingMonth2:
            if samplingMonth2 == 3:
                continue
            idx = newData.index[newData.samplingMonth2 == samplingMonth2].tolist()
            samples = iData.posterior.agrdt_mean.stack(sample=("chain", "draw"))[
                idx
            ].values.flatten()
            mean = np.mean(samples)
            hdi = az.hdi(samples, 0.94)
            statsDictTime["samplingMonth2"][samplingMonth2]["mean"] = mean
            statsDictTime["samplingMonth2"][samplingMonth2]["hdi"] = hdi
            statsDictTime["samplingMonth2"][samplingMonth2]["samples"] = samples

    pointPlotNErrorsNMeans(
        x="samplingMonth2",
        y="agrdt",
        df=dfFigure1_A,
        statsDict=statsDictTime,
        pal="k",
        order=order.samplingMonth2,
        xTicklabels="none",
        join=False,
        size=10,
        asPercent=True,
        ax=ax1,
    )

    hideGridX(ax1)
    ax1.set_ylabel(label.sensPercent)
    ax1.set_xticklabels([])

    plotVariantDominantTimes(
        dfPos,
        palette,
        abbrvDictPaper,
        ax=ax2,
        timeVar="samplingMonth2",
        fontsize=fontSizePlot,
    )

    # Set axes limits.
    xlim = (-0.5, 7.5)
    ax1.set_ylim(0, 1)
    ax1.set_yticklabels(list(range(0, 101, 20)))
    ax1.set_xlim(*xlim)

    setFontSize(ax1, fontSizePlot)

    plt.tight_layout(pad=0.5)

    return model, iData


def plotFig1_B(
    dfFigure1_B,
    dfPos,
    timeVar,
    xOrder,
    palette,
    label,
    abbrvDictPaper,
    ax,
    fontSizePlot=12,
):
    ax1, ax2 = ax
    # Viral loads for symptomatic AgRDT tests (first positive PCRs).
    boxNSwarmplot(
        dfFigure1_B,
        x=timeVar,
        y="vl",
        xLabel="",
        yLabel=label.vl,
        xCats=xOrder,
        ax=ax1,
        palBoxplot="none",
        palSwarmplot=None,
        legend=None,
        hueSwarmplot=None,
        hueOrder=None,
        markersize=2,
        markeredgecolor="None",
        dodge=False,
        alphaSwarmplot=1,
    )

    ax1.set_xticklabels([])

    plotVariantDominantTimes(
        dfPos, palette, abbrvDictPaper, ax2, timeVar=timeVar, fontsize=fontSizePlot
    )

    # Set axes limits.
    xlim = (-0.5, 7.5) if timeVar == "samplingMonth2" else (-0.5, 14.5)
    ax1.set_ylim(3, 11)
    ax1.set_xlim(*xlim)

    setFontSize(ax1, fontSizePlot)

    plt.tight_layout(pad=0.85)


def plotFig1_C(
    dfFigure1_C,
    timeVar,
    xOrder,
    palette,
    label,
    legend,
    ax,
    legendLoc=(0.07, 0.63),
    fontSizePlot=12,
):
    # Vaccination status.
    ax_2 = ax.twinx()
    # Plot counts.
    sns.countplot(
        data=dfFigure1_C,
        x=timeVar,
        hue="immunN",
        order=xOrder,
        palette=palette.immunN,
        ax=ax,
        zorder=1,
    )
    # Determine rates and plot them.
    y = []
    for monthBin, group in dfFigure1_C.groupby(timeVar):
        mean = (group.immun2YN == 1).sum() / len(group)
        y.append(mean)
    y.insert(3, np.nan)  # Account for June/July (with missing/removed values)
    ax_2.scatter(x=list(xOrder), y=y, s=10, alpha=0.8, color="k", zorder=2)
    ax_2.plot(list(xOrder), y, alpha=0.5, color="gray", zorder=2)
    ax_2.plot((2, 4), (y[2], y[4]), "--", alpha=0.5, color="gray", zorder=2)

    addNumbersWithoutHue(ax_2, colors="k", barplot=False, size=10, weight="normal")

    # Formatting.
    xlim = (-0.5, 7.5) if timeVar == "samplingMonth2" else (-0.5, 14.5)
    ax.set_ylim(-1.5, 250)
    ax.set_xlim(*xlim)

    ax.set_ylabel("Count")
    ax_2.set_ylabel("Multiply immunised fraction", rotation=270, labelpad=15)

    ax_2.grid(False)
    ax.axes.get_xaxis().get_label().set_visible(False)
    ax_2.axes.get_xaxis().get_label().set_visible(False)
    ax_2.set_xticklabels([])
    ax_2.set_ylim(-0.01, 1)

    # Process legend.
    ax.get_legend().remove()
    ax_2.legend(
        handles=legend.immunNPatch[1:],
        fontsize=fontSizePlot,
        title_fontsize=fontSizePlot,
        ncol=2,
        title=label.immunNLegend,
        columnspacing=3,
        borderaxespad=0.2,
        loc=legendLoc,
    )

    setFontSize(ax, fontSizePlot)
    setFontSize(ax_2, fontSizePlot)


def addFig1_D_x_labels(ax, labels, fontSizePlot=12):
    ax.set_xlabel("\nSampling period")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"\n{label}" for label in labels], rotation=45, ha="right")

    setFontSize(ax, fontSizePlot)


def plotFig1_D(dfPos, timeVar, palette, labels, abbrvDictPaper, fontSizePlot, ax):
    plotVariantDominantTimes(
        dfPos, palette, abbrvDictPaper, ax, timeVar=timeVar, fontsize=fontSizePlot
    )

    addFig1_D_x_labels(ax, labels=labels, fontSizePlot=fontSizePlot)


def plotFig2_A(dfFigure2_A, label, palette, abbrvDictPaper, ax, fontSizePlot=12):
    addJitterCol(dfFigure2_A, "daysPostOnset", "daysPostOnsetJitter", 0.01)
    bp, sp = boxNSwarmplot(
        dfFigure2_A,
        x="variant",
        y="daysPostOnset",
        xLabel=label.variant,
        yLabel=label.days,
        xCats=abbrvDictPaper["variant"],
        ax=ax,
        palBoxplot=palette.variant,
        markersize=1.7,
        alphaBoxplot=0.7,
        ySwarmplot="daysPostOnsetJitter",
    )

    bp.set_yticks(computeTicks(dfFigure2_A.daysPostOnset.dropna(), step=2))
    bp.set_ylim(-0.5, 14.5)
    setFontSize(bp, size=fontSizePlot)


def plotFig2_B(dfFigure2_B, palette, label, order, abbrvDictPaper, ax, fontSizePlot=12):
    # Use seaborns color parameter
    colors = [color for color in palette.variant.values() for i in range(2)]

    bp, sp = boxNSwarmplot(
        dfFigure2_B,
        x="variant",
        y="vl",
        xLabel=label.variant,
        yLabel=label.vl,
        xCats=abbrvDictPaper["variant"],
        palBoxplot=colors,
        alphaBoxplot=0.7,
        palSwarmplot=None,
        legend="none",
        hueSwarmplot="agrdtYN",
        hueBoxplot="agrdtYN",
        hueOrder=order.agrdtYN,
        markersize=1.3,
        markeredgecolor="k",
        ax=ax,
    )
    bp.set_yticks(computeTicks(dfFigure2_B.vl, step=1))
    setFontSize(bp, size=fontSizePlot)


def plotFig3_A(dfFigure3, immunSympDict, palette, order, label, ax, fontSizePlot=12):
    markersize = 2.9

    # All people, split by symptom status and vaccination/recovery status.
    boxNSwarmplot(
        dfFigure3,
        x="immun2YN:symptoms",
        y="vl",
        xLabel="",
        yLabel="",
        xCats=immunSympDict,
        ax=ax,
        palBoxplot="none",
        palSwarmplot=palette.res,
        legend="none",
        hueSwarmplot="agrdt",
        hueOrder=order.agrdt,
        markersize=markersize,
        markeredgecolor=None,
        dodge=False,
        alphaSwarmplot=1,
    )

    ax.set_ylabel(label.vl)
    ax.set_xticklabels([])
    ax.set_ylim(3, 11)
    ax.set_xlim(-0.85, 4.85)

    ax.set_yticks(computeTicks(dfFigure3.vl, step=1))
    ax.grid(color="gray", axis="y", alpha=0.23)
    setFontSize(ax, fontSizePlot)

    ax.vlines(x=2, ymin=0, ymax=13, color="lightgray")


def plotFig3_B(
    dfFigure3,
    statsDictFigure3,
    immunSympDict,
    immunSympPalette,
    label,
    ax,
    fontSizePlot=12,
):
    model, iData = None, None
    varName = "immun2YN:symptoms"
    if not varName in statsDictFigure3:
        n = 2500
        newData = pd.DataFrame(
            {
                "symptoms": np.concatenate(
                    (np.zeros(2 * n, dtype=int), np.ones(2 * n, dtype=int))
                ),
                "immun2YN": np.concatenate(
                    (
                        np.zeros(n, dtype=int),
                        np.ones(n, dtype=int),
                        np.zeros(n, dtype=int),
                        np.ones(n, dtype=int),
                    )
                ),
            }
        )
        model, iData = sampleProb(
            df=dfFigure3,
            outcome="agrdt",
            catVars=["immun2YN", "symptoms"],
            seed=SEED,
            target_accept=0.9,
            bambi=True,
            interaction=True,
            newData=newData,
        )

        for catCombination in immunSympDict:
            if catCombination == "dummy":
                continue
            immun, symp = map(int, catCombination.split(", "))
            idx = newData.index[
                (newData.immun2YN == immun) & (newData.symptoms == symp)
            ].tolist()
            samples = iData.posterior.agrdt_mean.stack(sample=("chain", "draw"))[
                idx
            ].values.flatten()
            mean = np.mean(samples)
            hdi = az.hdi(samples, 0.94)
            statsDictFigure3[varName][catCombination]["hdi"] = hdi
            statsDictFigure3[varName][catCombination]["mean"] = mean
            statsDictFigure3[varName][catCombination]["samples"] = samples

    pointPlotNErrorsNMeans(
        x="immun2YN:symptoms",
        y="agrdt",
        df=dfFigure3,
        statsDict=statsDictFigure3,
        pal=immunSympPalette,
        order=immunSympDict,
        xTicklabels="none",
        join=False,
        size=10,
        asPercent=True,
        ax=ax,
    )

    ax.set_ylabel(label.sensPercent)
    ax.set_xlabel(label.immun2YN, labelpad=15)
    ax.set_xticklabels(immunSympDict.values())
    ax.set_xlim(-0.85, 4.85)
    ax.set_ylim(0, 1)
    ax.set_yticklabels(list(range(0, 101, 20)))
    hideGridX(ax)
    ax.grid(color="gray", axis="y", alpha=0.23)
    setFontSize(ax, fontSizePlot)
    ax.vlines(x=2, ymin=0, ymax=1, color="lightgray")

    return model, iData


def plotFigA2_A(
    dfFigureA2, variantSympDict, label, palette, order, ax, fontSizePlot=12
):
    markersize = 2

    # All people, split by symptom status and vaccination/recovery status.
    boxNSwarmplot(
        dfFigureA2,
        x="variant:symptoms",
        y="vl",
        xLabel="",
        yLabel=label.vl,
        xCats=variantSympDict,
        ax=ax,
        palBoxplot="none",
        palSwarmplot=palette.res,
        legend="none",
        hueSwarmplot="agrdt",
        hueOrder=order.agrdt,
        markersize=markersize,
        markeredgecolor=None,
        dodge=False,
        alphaSwarmplot=1,
    )

    ax.set_xticklabels([])
    ax.set_ylim(3, 11)
    ax.set_xlim(-0.85, 8.85)
    ax.set_yticks(computeTicks(dfFigureA2.vl, step=1))
    ax.grid(color="gray", axis="y", alpha=0.23)
    ax.vlines(x=4, ymin=0, ymax=13, color="lightgray")
    setFontSize(ax, fontSizePlot)


def plotFigA2_B(
    dfFigureA2,
    statsDictFigure3,
    variantSympDict,
    variantSympPalette,
    label,
    ax,
    fontSizePlot=12,
):
    varName = "variant:symptoms"
    model, iData = None, None
    if not varName in statsDictFigure3:
        n = 1250
        variantsNew = pd.Series(
            (("wildtype",) * n + ("alpha",) * n + ("delta",) * n + ("omicron",) * n) * 2
        )
        newData = pd.DataFrame(
            {
                "symptoms": np.concatenate(
                    (np.zeros(4 * n, dtype=int), np.ones(4 * n, dtype=int))
                ),
                "variant": variantsNew,
            }
        )

        model, iData = sampleProb(
            df=dfFigureA2,
            outcome="agrdt",
            catVars=["variant", "symptoms"],
            seed=SEED,
            target_accept=0.9,
            bambi=True,
            interaction=True,
            newData=newData,
        )

        for catCombination in variantSympDict:
            if catCombination == "dummy":
                continue
            variant, symp = catCombination.split(", ")
            idx = newData.index[
                (newData.variant == variant) & (newData.symptoms == int(symp))
            ].tolist()
            samples = iData.posterior.agrdt_mean.stack(sample=("chain", "draw"))[
                idx
            ].values.flatten()
            mean = np.mean(samples)
            hdi = az.hdi(samples, 0.94)
            statsDictFigure3[varName][catCombination]["mean"] = mean
            statsDictFigure3[varName][catCombination]["hdi"] = hdi
            statsDictFigure3[varName][catCombination]["samples"] = samples

    pointPlotNErrorsNMeans(
        x="variant:symptoms",
        y="agrdt",
        df=dfFigureA2,
        statsDict=statsDictFigure3,
        pal=variantSympPalette,
        order=variantSympDict,
        xTicklabels="none",
        join=False,
        size=10,
        asPercent=True,
        ax=ax,
    )

    ax.set_xlabel(label.variant, rotation=0, labelpad=15)
    ax.set_ylabel(label.sensPercent)
    ax.set_xlim(-0.85, 8.85)
    ax.set_ylim(0, 1)
    ax.set_yticklabels(list(range(0, 101, 20)))
    hideGridX(ax)
    ax.grid(color="gray", axis="y", alpha=0.23)
    setFontSize(ax, fontSizePlot)

    xticklabels = variantSympDict.values()
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.vlines(x=4, ymin=0, ymax=1, color="lightgray")

    return model, iData


def plotFigA4_left(df, palette, label, ax, fontSizePlot=12):
    # Pre-VOC and Alpha.
    dfPrevocAlpha = df[df.variant.isin((0, 1))]
    sns.swarmplot(
        data=dfPrevocAlpha,
        x="immun2YN",
        y="vl",
        hue="agrdt",
        palette=palette.res,
        ax=ax,
    )
    ax.legend_.remove()
    ax.set_xlabel("")
    ax.set_ylabel(
        label.vl,
    )
    ax.set_xticklabels(("no",))
    ax.set_ylim(3, 11)
    ax.set_title("Pre-VOC and Alpha", fontweight="bold")

    xticklabels = ["no"]
    ax.set_xticklabels(xticklabels)
    setFontSize(ax, fontSizePlot)


def plotFigA4_right(df, negPointsYCoords, palette, legend, ax, fontSizePlot=12):
    # Delta and Omicron.
    dfDeltaOmicron = df[df.variant.isin((2, 3))]
    sns.swarmplot(
        data=dfDeltaOmicron,
        x="immun2YN",
        y="vl",
        hue="agrdt",
        palette=palette.res,
        ax=ax,
        zorder=1,
    )

    # Highlight the two data points that are removed to check their influence on the
    # variant/vaccination parameter.
    ax.scatter(
        [0, 0], negPointsYCoords, s=170, facecolors="none", edgecolors="r", zorder=2
    )

    ax.set_xticks((0, 1))
    xticklabels = ["no", "yes"]
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(-0.5, 1.5)
    replaceLegend(ax, legend.resPoints, loc="lower left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Delta and Omicron", fontweight="bold", fontsize=fontSizePlot)

    setFontSize(ax, fontSizePlot)


def returnIDataParamMapping():
    return PARAM_NAME_MAPPINGS


def writeIDataSummaryTable(summaryDf, varnames, outfile):
    varnameRegex = "|".join(varnames)
    paramNameMappings = PARAM_NAME_MAPPINGS
    paramNameMappings["immun2YN[1]"] = LABEL.immun2YN
    paramNameMappings["zAge"] = "Age"

    params = [param for param in summaryDf.index if re.match(varnameRegex, param)]
    params = [param for param in params if param in paramNameMappings]
    # Extract relevant rows.
    summaryDf = summaryDf.loc[params]
    # Extract relevant columns.
    summaryDf = summaryDf[["mean", "sd", "hdi_3%", "hdi_97%"]]

    summaryDf.rename(
        index={
            param: PARAM_NAME_MAPPINGS.get(param) or param for param in summaryDf.index
        },
        columns={"hdi_3%": "3% HPDI", "hdi_97%": "97% HPDI"},
        inplace=True,
    )
    summaryDf = summaryDf.round(decimals=2)
    summaryDf.to_csv(outfile, sep="\t")

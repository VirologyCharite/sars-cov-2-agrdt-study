import pandas as pd
import seaborn as sns
from datetime import date

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

DATE_RANGE_DOMINANT_WILDTYPE = date(2019, 11, 1), date(2021, 2, 7)
DATE_RANGE_DOMINANT_ALPHA = date(2021, 3, 29), date(2021, 5, 23)
DATE_RANGE_DOMINANT_DELTA = date(2021, 7, 12), date(2021, 12, 19)
DATE_RANGE_DOMINANT_OMICRON = date(2022, 1, 17), date.today()

FONT_SIZE_PLOT = 12
_GRAY = "0.6"

# Define brackets for days post symptom onset and corresponding labels.
_SYMP_DAYS_BRACKETS = ((-0.001, 2), (2, 7))
_SYMP_DAYS_BINS = [
    pd.Interval(low, up, closed="right") for low, up in _SYMP_DAYS_BRACKETS
]


_SYMP_DAYS_BINS_STR = (
    f"0 - {_SYMP_DAYS_BRACKETS[0][1]} days",
    f"{_SYMP_DAYS_BRACKETS[0][1]} - " f"{_SYMP_DAYS_BRACKETS[1][1]} days",
)


_COLORS_VARIANTS = (
    ["tab:blue"]
    + [sns.color_palette("colorblind")[4]]
    + sns.color_palette("colorblind")[2:4]
)
_COLORS_GENDER = sns.color_palette("tab10")[8:10][::-1]
_COLORS_IMMUN = sns.color_palette("rocket_r") + ["black"]
_COLORS_IMMUN_2YN = sns.color_palette("colorblind")[:2]
_COLORS_RES = sns.color_palette("hls", 2) + [_GRAY]

FEMALE = 0
MALE = 1

WT = 0
ALPHA = 1
DELTA = 2
OMICRON = 3

WT_LABEL = "Pre-VOC"
ALPHA_LABEL = "Alpha"
DELTA_LABEL = "Delta"
OMICRON_LABEL = "Omicron"

NEG = 0
POS = 1
MISSING = 2

CCM = 0
CVK = 1
CBF = 2

ROCHE = 0
ABBOTT = 1

_COLOR_DAYS_POST_FIRST_POS_PCR_BIN = sns.color_palette()


def _combine(list1, list2):
    return list1 + list2


# Palettes for plotting.
_PALETTES = {
    "symp": {0: "firebrick", 1: "indigo"},
    "res": {NEG: _COLORS_RES[NEG], POS: _COLORS_RES[POS]},
    "resPCR": {NEG: _COLORS_RES[NEG], POS: _COLORS_RES[POS]},
    "agrdt": {NEG: "black", POS: "white"},
    "agrdtYN": {False: "gray", True: "darkred"},
    "gender": {"F": _COLORS_GENDER[FEMALE], "M": _COLORS_GENDER[MALE]},
    "variant": {
        "wildtype": _COLORS_VARIANTS[WT],
        "alpha": _COLORS_VARIANTS[ALPHA],
        "delta": _COLORS_VARIANTS[DELTA],
        "omicron": _COLORS_VARIANTS[OMICRON],
    },
    "variantCodes": {i: _COLORS_VARIANTS[i] for i in range(4)},
    "immunN": dict(zip(range(5), _COLORS_IMMUN)),
    "immun2YN": {0.0: _COLORS_IMMUN_2YN[0], 1.0: _COLORS_IMMUN_2YN[1]},
    "binDaysPostOnset": {
        sympDaysBin: col
        for col, sympDaysBin in zip(
            sns.color_palette("colorblind")[4:6], _SYMP_DAYS_BINS
        )
    },
    "binDaysPostFirstPosPcr": {
        0: _COLOR_DAYS_POST_FIRST_POS_PCR_BIN[0],
        1: _COLOR_DAYS_POST_FIRST_POS_PCR_BIN[1],
    },
    "cellculture": {
        "BA.1_CaCo2": "firebrick",
        "BA.1_VeroE6": "maroon",
        "BA.2_CaCo2": "turquoise",
        "BA.2_VeroE6": "lightseagreen",
        "WT_CaCo2": "hotpink",
        "WT_VeroE6": "palevioletred",
        "Mock_CaCo2": "skyblue",
        "Mock_VeroE6": "deepskyblue",
    },
}

_ABBRVS = {
    "agrdt": {0: "negative", 1: "positive"},
    "agrdtYN": {False: "no", True: "yes"},
    "pcrPositive": {False: "negative", True: "positive"},
    "symptoms": {0: "asymptomatic", 1: "symptomatic"},
    "gender": {"F": "women", "M": "men"},
    "variant": {
        "wildtype": WT_LABEL,
        "alpha": ALPHA_LABEL,
        "delta": DELTA_LABEL,
        "omicron": OMICRON_LABEL,
    },
    "binDaysPostOnset": dict(zip(_SYMP_DAYS_BINS, _SYMP_DAYS_BINS_STR)),
    "binDaysPostFirstPosPcr": {0: "0", 1: ">= 7"},
    "vaccN": dict(zip(range(4), range(4))),
    "immunN": dict(zip(range(4), range(4))),
    "immun2YN": {0.0: "no", 1.0: "yes"},
    "symptom": {0: "no", 1: "yes"},
    "recoveredRecently": {0: "no", 1: "yes"},
    # First month is November 2020 but we actually only have antigen test
    # data for this month and no PCR data (missing PCR ids).
    "samplingMonth": dict(
        enumerate(
            (
                "Dec '20",
                "Jan '21",
                "Feb '21",
                "March '21",
                "April '21",
                "May '21",
                "June '21",
                "July '21",
                "Aug '21",
                "Sep '21",
                "Oct '21",
                "Nov '21",
                "Dec '21",
                "Jan '22",
                "Feb '22",
            )
        )
    ),
    "samplingMonth2": dict(
        enumerate(
            (
                "Dec '20 - Jan '21",
                "Feb '21 - March '21",
                "April '21 - May '21",
                "June '21 - July '21",
                "Aug '21 - Sep '21",
                "Oct '21 - Nov '21",
                "Dec '21 - Jan '22",
                "Feb '22",
            )
        )
    ),
    "cellculture": {
        "BA.1_CaCo2": "BA.1, Caco-2",
        "BA.1_VeroE6": "BA.1, Vero E6",
        "BA.2_CaCo2": "BA.2, Caco-2",
        "BA.2_VeroE6": "BA.2, Vero E6",
        "WT_CaCo2": "Wildtype, Caco-2",
        "WT_VeroE6": "Wildtype, Vero E6",
        "Mock_CaCo2": "Mock, Caco-2",
        "Mock_VeroE6": "Mock, Vero E6",
    },
}


_LABELS = {
    "hdi": "94% HPDI",
    "sens": "Ag-RDT sensitivity",
    "sensPercent": "Ag-RDT sensitivity (%)",
    "res": "Ag-RDT result",
    "resPCR": "PCR result",
    "samplingMonth2": "Sampling period",
    "testline": "Testline strength",
    "vl": r"$\mathregular{Log_{10}}$ viral load",
    "zVl": r"$\mathregular{Log_{10}}$ viral load",
    # This is the format that gets properly displayed when saving a C{
    # pd.DataFrame} (it doesn't work in C{matplotlib} figures though.
    "vlRNA": r"$\mathregular{Log_{10}}$ SARS-CoV-2 mRNA/mL",
    "vlDf": "log\u2081\u2080 viral load",
    "zVlDf": "log\u2081\u2080 viral load",
    "zAge": "Age",
    "days": "Days post symptom onset",
    "binDaysPostOnset": "Days post symptom onset",
    "daysSinceFirstPosPcr": "Days since first positive PCR",
    "pcr": "PCR positive rate",
    "symptoms": "Symptomatic status",
    "gender": "Sex",
    "variant": "SARS-CoV-2 variant",
    "immunN": "Sum of vaccinations and\nprior infections",
    "immunNLegend": "Sum of vaccinations\nand prior infections",
    "agrdtYN": "Ag-RDT performed",
    "immun2YN": "Multiply immunised",
    "immun2YNLegend": "Multiply immunised",
}

_ABBRV_TO_COLOR = {
    "binDaysPostOnset": tuple(
        zip(
            _ABBRVS["binDaysPostOnset"].values(), _PALETTES["binDaysPostOnset"].values()
        )
    ),
    "binDaysPostFirstPosPcr": tuple(
        zip(
            _ABBRVS["binDaysPostFirstPosPcr"].values(),
            _PALETTES["binDaysPostFirstPosPcr"].values(),
        )
    ),
    "immunN": tuple(zip(_ABBRVS["immunN"].values(), _PALETTES["immunN"].values())),
    "immun2YN": tuple(
        zip(_ABBRVS["immun2YN"].values(), _PALETTES["immun2YN"].values())
    ),
    "agrdtYN": tuple(zip(_ABBRVS["agrdtYN"].values(), _PALETTES["agrdtYN"].values())),
    "cellculture": tuple(
        zip(_ABBRVS["cellculture"].values(), _PALETTES["cellculture"].values())
    ),
}


# Legends for plotting.
_LEGENDS = {
    "sympAsymp": [
        Line2D(
            [0],
            [0],
            marker="o",
            color="tab:orange",
            label="symptomatic",
            markerfacecolor="tab:orange",
            markersize=8,
            linestyle=None,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="tab:blue",
            label="asymptomatic",
            markerfacecolor="tab:blue",
            markersize=8,
            linestyle=None,
        ),
    ],
    "sympAsympPatch": [
        Patch(color="tab:blue", label="asymptomatic"),
        Patch(color="tab:orange", label="symptomatic"),
    ],
    "resPatch": [
        Rectangle(
            (0, 0), 0, 0, label=_LABELS["res"], facecolor="none", edgecolor="none"
        ),
        Patch(color=_COLORS_RES[POS], label="positive", alpha=1),
        Patch(color=_COLORS_RES[NEG], label="negative", alpha=1),
    ],
    "resPCRPatch": [
        Rectangle(
            (0, 0), 0, 0, label=_LABELS["resPCR"], facecolor="none", edgecolor="none"
        ),
        Patch(color=_COLORS_RES[POS], label="positive", alpha=1),
        Patch(color=_COLORS_RES[NEG], label="negative", alpha=1),
    ],
    "resPoints": [
        Rectangle(
            (0, 0), 0, 0, label=_LABELS["res"], facecolor="none", edgecolor="none"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_PALETTES["res"][POS],
            label="positive",
            markerfacecolor=_PALETTES["res"][POS],
            markeredgecolor="none",
            markersize=8,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_PALETTES["res"][NEG],
            label="negative",
            markerfacecolor=_PALETTES["res"][NEG],
            markeredgecolor="none",
            markersize=8,
            linestyle="None",
        ),
    ],
    "agrdtYNPoints": _combine(
        [
            Rectangle(
                (0, 0),
                0,
                0,
                label=_LABELS["agrdtYN"],
                facecolor="none",
                edgecolor="none",
            )
        ],
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color=col,
                label=cat,
                markerfacecolor=col,
                markersize=8,
                linestyle="None",
            )
            for cat, col in _ABBRV_TO_COLOR["agrdtYN"]
        ],
    ),
    "variantLines": [
        Rectangle(
            (0, 0), 0, 0, label="SARS-CoV-2 variant", facecolor="none", edgecolor="none"
        ),
        Line2D(
            [0],
            [0],
            marker=None,
            color=_COLORS_VARIANTS[WT],
            label=WT_LABEL,
            markerfacecolor=_COLORS_VARIANTS[WT],
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker=None,
            color=_COLORS_VARIANTS[ALPHA],
            label=ALPHA_LABEL,
            markerfacecolor=_COLORS_VARIANTS[ALPHA],
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker=None,
            color=_COLORS_VARIANTS[DELTA],
            label=DELTA_LABEL,
            markerfacecolor=_COLORS_VARIANTS[DELTA],
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker=None,
            color=_COLORS_VARIANTS[OMICRON],
            label=OMICRON_LABEL,
            markerfacecolor=_COLORS_VARIANTS[OMICRON],
            markersize=8,
            linestyle="-",
        ),
    ],
    "variantPoints": [
        Rectangle(
            (0, 0), 0, 0, label="SARS-CoV-2 variant", facecolor="none", edgecolor="none"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_COLORS_VARIANTS[WT],
            label=WT_LABEL,
            markerfacecolor=_COLORS_VARIANTS[WT],
            markersize=8,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_COLORS_VARIANTS[ALPHA],
            label=ALPHA_LABEL,
            markerfacecolor=_COLORS_VARIANTS[ALPHA],
            markersize=8,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_COLORS_VARIANTS[DELTA],
            label=DELTA_LABEL,
            markerfacecolor=_COLORS_VARIANTS[DELTA],
            markersize=8,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_COLORS_VARIANTS[OMICRON],
            label=OMICRON_LABEL,
            markerfacecolor=_COLORS_VARIANTS[OMICRON],
            markersize=8,
            linestyle="None",
        ),
    ],
    "variantPointsLines": [
        Rectangle(
            (0, 0), 0, 0, label="SARS-CoV-2 variant", facecolor="none", edgecolor="none"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_COLORS_VARIANTS[WT],
            label=WT_LABEL,
            markerfacecolor=_COLORS_VARIANTS[WT],
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_COLORS_VARIANTS[ALPHA],
            label=ALPHA_LABEL,
            markerfacecolor=_COLORS_VARIANTS[ALPHA],
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_COLORS_VARIANTS[DELTA],
            label=DELTA_LABEL,
            markerfacecolor=_COLORS_VARIANTS[DELTA],
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_COLORS_VARIANTS[OMICRON],
            label=OMICRON_LABEL,
            markerfacecolor=_COLORS_VARIANTS[OMICRON],
            markersize=8,
            linestyle="-",
        ),
    ],
    "variantPatch": [
        Rectangle(
            (0, 0), 0, 0, label="SARS-CoV-2 variant", facecolor="none", edgecolor="none"
        ),
        Patch(color=_COLORS_VARIANTS[WT], label=WT_LABEL),
        Patch(color=_COLORS_VARIANTS[ALPHA], label=ALPHA_LABEL),
        Patch(color=_COLORS_VARIANTS[DELTA], label=DELTA_LABEL),
        Patch(color=_COLORS_VARIANTS[OMICRON], label=OMICRON_LABEL),
    ],
    "binDaysPostOnset": _combine(
        [
            Rectangle(
                (0, 0),
                0,
                0,
                label=_LABELS["binDaysPostOnset"],
                facecolor="none",
                edgecolor="none",
            )
        ],
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color=col,
                label=cat,
                markerfacecolor=col,
                markersize=7,
            )
            for cat, col in _ABBRV_TO_COLOR["binDaysPostOnset"]
        ],
    ),
    "binDaysPostOnsetLines": _combine(
        [
            Rectangle(
                (0, 0),
                0,
                0,
                label=_LABELS["binDaysPostOnset"],
                facecolor="none",
                edgecolor="none",
            )
        ],
        [
            Line2D(
                [0],
                [0],
                marker=None,
                color=col,
                label=cat,
                markerfacecolor=col,
                markersize=7,
            )
            for cat, col in _ABBRV_TO_COLOR["binDaysPostOnset"]
        ],
    ),
    "immunNPatch": _combine(
        [
            Rectangle(
                (0, 0),
                0,
                0,
                label=_LABELS["immunNLegend"],
                facecolor="none",
                edgecolor="none",
            )
        ],
        [Patch(color=col, label=cat) for cat, col in _ABBRV_TO_COLOR["immunN"]],
    ),
    "immun2YNPatch": _combine(
        [
            Rectangle(
                (0, 0),
                0,
                0,
                label=_LABELS["immun2YN"],
                facecolor="none",
                edgecolor="none",
            )
        ],
        [
            Patch(color=_COLORS_IMMUN_2YN[int(i)], label=label)
            for i, label in _ABBRVS["immun2YN"].items()
        ],
    ),
    "immun2YNLines": _combine(
        [
            Rectangle(
                (0, 0),
                0,
                0,
                label=_LABELS["immun2YNLegend"],
                facecolor="none",
                edgecolor="none",
            )
        ],
        [
            Line2D(
                [0],
                [0],
                marker=None,
                color=col,
                label=cat,
                markerfacecolor=col,
                markersize=7,
            )
            for cat, col in _ABBRV_TO_COLOR["immun2YN"]
        ],
    ),
    "immun2YNPoints": _combine(
        [
            Rectangle(
                (0, 0),
                0,
                0,
                label=_LABELS["immun2YNLegend"],
                facecolor="none",
                edgecolor="none",
            )
        ],
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color=col,
                label=cat,
                markerfacecolor=col,
                markersize=7,
                linestyle="none",
            )
            for cat, col in _ABBRV_TO_COLOR["immun2YN"]
        ],
    ),
    "binDaysPostOnsetPointsLines": [
        Rectangle(
            (0, 0),
            0,
            0,
            label=_LABELS["binDaysPostOnset"],
            facecolor="none",
            edgecolor="none",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            label=_SYMP_DAYS_BINS_STR[0],
            markerfacecolor="gray",
            markersize=6,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="gray",
            label=_SYMP_DAYS_BINS_STR[1],
            markerfacecolor="gray",
            markersize=6,
            linestyle="-",
        ),
    ],
    "cellculture": [
        Line2D(
            [0],
            [0],
            marker=None,
            color=col,
            label=cat,
            markerfacecolor=col,
            markersize=8,
            linestyle="-",
        )
        for cat, col in _ABBRV_TO_COLOR["cellculture"]
    ],
    "variantLinesWtOmicron": [
        Rectangle(
            (0, 0), 0, 0, label="SARS-CoV-2 variant", facecolor="none", edgecolor="none"
        ),
        Line2D(
            [0],
            [0],
            marker=None,
            color=_COLORS_VARIANTS[WT],
            label=WT_LABEL,
            markerfacecolor=_COLORS_VARIANTS[WT],
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker=None,
            color=_COLORS_VARIANTS[OMICRON],
            label=OMICRON_LABEL,
            markerfacecolor=_COLORS_VARIANTS[OMICRON],
            markersize=8,
            linestyle="-",
        ),
    ],
}

_ORDERS = {
    "agrdt": tuple(_PALETTES["agrdt"]),
    "agrdtYN": tuple(_PALETTES["agrdtYN"]),
    "pcrPositive": tuple(_PALETTES["resPCR"]),
    "variant": tuple(_PALETTES["variant"]),
    "gender": tuple(_PALETTES["gender"]),
    "symp": tuple(_PALETTES["symp"]),
    "binDaysPostOnset": tuple(_PALETTES["binDaysPostOnset"]),
    "immunN": tuple(_PALETTES["immunN"]),
    "immun2YN": tuple(_PALETTES["immun2YN"]),
    "samplingMonth": (
        "2020-11",
        "2020-12",
        "2021-01",
        "2021-02",
        "2021-03",
        "2021-04",
        "2021-05",
        "2021-06",
        "2021-07",
        "2021-08",
        "2021-09",
        "2021-10",
        "2021-11",
        "2021-12",
        "2022-01",
        "2022-02",
    ),
    "samplingMonth2": tuple(_ABBRVS["samplingMonth2"]),
    "cellculture": tuple(_ABBRVS["cellculture"]),
}


# Palettes for plotting.
_PALETTES = {
    "res": {NEG: _COLORS_RES[NEG], POS: _COLORS_RES[POS]},
    "agrdtYN": {False: "gray", True: "darkred"},
    "gender": {"F": _COLORS_GENDER[FEMALE], "M": _COLORS_GENDER[MALE]},
    "variant": {
        "wildtype": _COLORS_VARIANTS[WT],
        "alpha": _COLORS_VARIANTS[ALPHA],
        "delta": _COLORS_VARIANTS[DELTA],
        "omicron": _COLORS_VARIANTS[OMICRON],
    },
    "immunN": dict(zip(range(5), _COLORS_IMMUN)),
    "immun2YN": {0.0: _COLORS_IMMUN_2YN[0], 1.0: _COLORS_IMMUN_2YN[1]},
    "binDaysPostOnset": {
        sympDaysBin: col
        for col, sympDaysBin in zip(
            sns.color_palette("colorblind")[4:6], _SYMP_DAYS_BINS
        )
    },
    "binDaysPostFirstPosPcr": {
        0: _COLOR_DAYS_POST_FIRST_POS_PCR_BIN[0],
        1: _COLOR_DAYS_POST_FIRST_POS_PCR_BIN[1],
    },
    "cellculture": {
        "BA.1_CaCo2": "firebrick",
        "BA.1_VeroE6": "maroon",
        "BA.2_CaCo2": "turquoise",
        "BA.2_VeroE6": "lightseagreen",
        "WT_CaCo2": "hotpink",
        "WT_VeroE6": "palevioletred",
        "Mock_CaCo2": "skyblue",
        "Mock_VeroE6": "deepskyblue",
    },
}


class PlotParams:
    def __init__(self, paramDict):
        self.__dict__.update(paramDict)


def getAbbrvsDict():
    return _ABBRVS


def getLabels():
    return PlotParams(_LABELS)


def getLegends():
    return PlotParams(_LEGENDS)


def getOrders():
    return PlotParams(_ORDERS)


def getPalettes():
    return PlotParams(_PALETTES)

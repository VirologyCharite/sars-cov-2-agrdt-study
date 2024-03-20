import decimal
import numpy as np
from datetime import date
from pathlib import Path

from utils.plotParams import (
    DATE_RANGE_DOMINANT_WILDTYPE,
    DATE_RANGE_DOMINANT_ALPHA,
    DATE_RANGE_DOMINANT_DELTA,
    DATE_RANGE_DOMINANT_OMICRON,
)

OUTPUT_TABLE_EMPLOYEES = Path("..", "output", "employeeTable.tsv")
OUTPUT_TABLE_SYMPTOMS = Path("..", "output", "symptomsTable.tsv")


def addJitterCol(df, origCol, jitterCol, sd=0.01):
    """
    Add a column "jitterCol" with random jittering applied to the values in "origCol".
    @param df: A C{pandas DataFrame}.
    @param origCol: The name of the column whose values you want to apply jittering to.
    @param jitterCol: The name of the column to be created, containing the values that
    have been jittered.
    """
    stdev = sd * (df[origCol].max() - df[origCol].min())
    df.loc[:, jitterCol] = df[origCol] + np.random.randn(len(df[origCol])) * stdev


def dataFrameAgrdt(df):
    """
    @param df: The pandas dataframe returned by calling dataFrameCusma.
    @return: A C{pd.DataFrame} containing only tests with an Ag-RDT result available.
    """
    return df.dropna(subset=("agrdt"))


def dataFrameFirstPosPcr(df):
    """
    Return a data frame containing only data points that correspond to the first
    positive PCR of a person within an infection.
    """
    return df[df.isFirstPosPcr == 1].copy()


def dataFrameIndependent(df):
    """
    @param df: The pandas dataframe returned by calling dataFrameCusma.
    @return: A dataframe that only has one test per person (the first) so that test data
    points are independent of each other.
    """

    # Make sure that values are sorted by pcr date so that only the first test will be
    # kept for each person to keep things consistent.
    df = df.sort_values("pcrDate", ignore_index=True)
    return df.drop_duplicates(subset=["personHash"], keep="first")


def dataFramePCRpos(df):
    """
    @param df: The pandas dataframe returned by calling dataFrameCusma.
    @return: A C{pd.DataFrame} containing only data with positive PCR result.
    """
    return df[df.pcrPositive].copy()


def dataFrameSymptoms(df):
    """
    @param df: The pandas dataframe returned by calling dataFrameCusma.
    @return: A dataframe that only has tests in them where the respective person was
    symptomatic.
    """
    return df[df.symptoms == 1].copy()


def getPCRsOutsideVariantPrevalentRanges(df):
    return df[
        (
            (df.pcrDate >= DATE_RANGE_DOMINANT_WILDTYPE[1])
            & (df.pcrDate <= DATE_RANGE_DOMINANT_ALPHA[0])
        )
        | (
            (df.pcrDate >= DATE_RANGE_DOMINANT_ALPHA[1])
            & (df.pcrDate <= DATE_RANGE_DOMINANT_DELTA[0])
        )
        | (
            (df.pcrDate >= DATE_RANGE_DOMINANT_DELTA[1])
            & (df.pcrDate <= DATE_RANGE_DOMINANT_OMICRON[0])
        )
    ]


def returnDeltaOmicronTransitionPCRs(df):
    """
    Return data points from the time of delta/omicron transition.
    """
    return df[
        (df.pcrDate > DATE_RANGE_DOMINANT_DELTA[1])
        & (df.pcrDate < DATE_RANGE_DOMINANT_OMICRON[0])
    ].copy()


def removeDeltaOmicronPCRs(df):
    """
    Remove data points from the time when delta/omicron typing PCRs were done
    (Dec 2021 - Jan 2022) so that they don't give a false impression of viral
    load levels for those variants (typing PCRs in this period were done for
    high viral load level samples only).
    """
    return df[
        (df.pcrDate <= DATE_RANGE_DOMINANT_DELTA[1])
        | (df.pcrDate >= DATE_RANGE_DOMINANT_OMICRON[0])
    ].copy()


def returnWtAlphaTransitionPCRs(df):
    """
    Retrun data points from the time of wildtype/alpha transition.
    """
    return df[
        (df.pcrDate > DATE_RANGE_DOMINANT_WILDTYPE[1])
        & (df.pcrDate < DATE_RANGE_DOMINANT_ALPHA[0])
    ].copy()


def removeWtAlphaPCRs(df):
    """
    Remove data points from the time when wildtype/alpha typing PCRs were done
    (Feb 2021 - March 2021) so that they don't give a false impression of viral
    load levels for those variants (typing PCRs in this period were done for
    high viral load level samples only).
    """
    return df[
        (df.pcrDate <= DATE_RANGE_DOMINANT_WILDTYPE[1])
        | (df.pcrDate >= DATE_RANGE_DOMINANT_ALPHA[0])
    ].copy()


def removeAlphaDeltaPCRs(df):
    """
    Remove data points from the time when alpha/delta typing PCRs were done
    (May 2021 - July 2021) so that they don't give a false impression of viral
    load levels for those variants (typing PCRs in this period were done for
    high viral load level samples only).
    """
    return df[
        (df.pcrDate <= DATE_RANGE_DOMINANT_ALPHA[1])
        | (df.pcrDate >= DATE_RANGE_DOMINANT_DELTA[0])
    ].copy()


def removeAllUnclearVariantOrTypingPCRs(df):
    df = removeWtAlphaPCRs(df)
    df = removeAlphaDeltaPCRs(df)
    df = removeDeltaOmicronPCRs(df)
    return df


def removeReleaseTesting(df):
    """
    Remove data points from release testing.
    """
    return df[df.reasonPres != "releaseTesting"].copy()


def standardize(df, origCol, standCol):
    """
    @param df: A C{pandas DataFrame}.
    @param origCol: The name of the column whose values you want to standardize
        (turn into z-scores).
    @param standCol: The name of the column to be created, containing the standardized
    values.
    """
    df[standCol] = (df[origCol] - df[origCol].mean()) / df[origCol].std()


def dataFrameNoRecovered(df):
    """
    @param df: The pandas dataframe returned by calling dataFrameCusma.
    @return: A C{pd.DataFrame} containing no PCRs of people who we know have had a
    previous SARS-CoV-2 infection at the time of testing.
    """
    return df[df.recovered != 1].copy()


def dataFrameFemaleMale(df):
    return df[df.gender.isin(("F", "M"))].copy()


def IQRQuartiles(column):
    return tuple(column.quantile([0.25, 0.75]))


def IQR(column):
    q25, q75 = IQRQuartiles(column)
    return q75 - q25


def roundHalfUp(value, decimals=2):
    with decimal.localcontext() as ctx:
        d = decimal.Decimal(value)
        ctx.rounding = decimal.ROUND_HALF_UP
        return float(round(d, decimals))


def mapRoundHalfUp(values, decimals=2):
    return [roundHalfUp(value, decimals=decimals) for value in values]


def createNewString(stat1, stat2, stat1Value, stat2Value):
    stat1Value = float(stat1Value)
    try:
        stat2Value = float(stat2Value)
    except TypeError:
        stat2Value = tuple(map(float, stat2Value))

    if stat2 == "mean":
        stat2Value *= 100
        stat2Value = f"({roundHalfUp(stat2Value)}%)"
    elif stat2 == "IQRQuartiles":
        stat2Value = tuple(map(roundHalfUp, stat2Value))
    else:
        stat2Value = f"({roundHalfUp(stat2Value)})"
    if stat1 == "sum":
        stat1Value = int(stat1Value)
    else:
        stat1Value = roundHalfUp(stat1Value)

    return str(stat1Value) + " " + f"{stat2Value}"


def writeTableEmployees(outfile, df, colFuncsDict):
    summaryDict = dict(df.groupby("symptoms2").agg(colFuncsDict))
    summaryDictTotal = dict(df.agg(colFuncsDict))

    categoryDict = {
        "Female": "Gender",
        "Age": "Age (years)",
        "Cough": "Symptoms",
        "RT-PCR": "Positive SARS-CoV-2 test result",
    }

    with open(outfile, "w") as oE:
        nSymp = int((df.symptoms2 == "Symptomatic").sum())
        nAsymp = int((df.symptoms2 == "Asymptomatic").sum())
        nUnknown = int((df.symptoms2 == "Unknown").sum())
        nTotal = len(df)

        oE.write(
            "\t".join(
                (
                    "Tests from",
                    "",
                    f"Symptomatic employees (n={nSymp})",
                    f"Asymptomatic employees (n={nAsymp})",
                    f"Employees with unknown symptom status (n={nUnknown})",
                    f"Total (n={nTotal})",
                )
            )
            + "\n"
        )
        for feature, stats in colFuncsDict.items():
            try:
                categoryString = categoryDict[feature]
            except KeyError:
                categoryString = ""

            printStrings = [categoryString, feature]
            stat1, stat2 = stats
            if stat2 == IQRQuartiles:
                stat2 = "IQRQuartiles"

            for symptomStatus in ("Symptomatic", "Asymptomatic", "Unknown"):
                newString = createNewString(
                    stat1,
                    stat2,
                    summaryDict[(feature, stat1)][symptomStatus],
                    summaryDict[(feature, stat2)][symptomStatus],
                )
                printStrings.append(newString)
            # Total counts.
            newString = createNewString(
                stat1,
                stat2,
                summaryDictTotal[feature][stat1],
                summaryDictTotal[feature][stat2],
            )
            printStrings.append(newString)

            oE.write("\t".join(printStrings) + "\n")


def writeTableSymptoms(outfile, df, colFuncsDict):
    summaryDict = dict(df[df.symptoms2 == "Symptomatic"].agg(colFuncsDict))

    with open(outfile, "w") as oS:
        oS.write("\t".join(["Symptom", "Count (mean)\n"]))
        for feature, stats in colFuncsDict.items():
            stat1, stat2 = stats
            if stat2 == IQRQuartiles:
                stat2 = "IQRQuartiles"
            newString = createNewString(
                stat1, stat2, summaryDict[feature][stat1], summaryDict[feature][stat2]
            )
            oS.write("\t".join([feature, newString]) + "\n")


def writeSummaryTables(df):
    symptoms = sorted(
        [
            "fatigue",
            "headache",
            "vertigo",
            "melalgia",
            "fever",
            "cough",
            "runnyNose",
            "soreThroat",
            "dyspnea",
            "noSmell",
            "noTaste",
            "nausea",
            "noAppetite",
            "vomiting",
            "diarrhea",
        ]
    )

    # Prepare a dataframe
    df["symptoms2"] = df.symptoms.replace({0: "Asymptomatic", 1: "Symptomatic"})
    df["symptoms2"] = df["symptoms2"].fillna("Unknown")

    df["Male"] = df.gender == "M"
    df["Female"] = df.gender == "F"
    df["Unknown"] = df.gender == "U"

    colsOfInterest = [
        "Female",
        "Male",
        "Unknown",
        "age",
        "pcrPositive",
        "agrdt",
        "symptoms2",
    ] + symptoms
    dfTable = df[colsOfInterest].copy()
    dfTable = dfTable.rename(
        columns={"age": "Age", "pcrPositive": "RT-PCR", "agrdt": "Ag-RDT"}
    )

    # Rename symptom columns.
    symptomDisplayDict1 = {
        "noAppetite": "No appetite",
        "noSmell": "No smell",
        "noTaste": "No taste",
        "runnyNose": "Runny nose",
        "soreThroat": "Sore throat",
    }
    symptomDisplayDict2 = {
        symptom: symptom.capitalize()
        for symptom in symptoms
        if symptom not in symptomDisplayDict1
    }
    symptomDisplayDict = {**symptomDisplayDict1, **symptomDisplayDict2}
    dfTable = dfTable.rename(columns=symptomDisplayDict)

    # Specify which summary values to compute.
    colFuncsFeatures = {
        "Female": ["sum", "mean"],
        "Male": ["sum", "mean"],
        "Unknown": ["sum", "mean"],
        "Age": ["median", IQRQuartiles],
    }
    colFuncsTests = {"RT-PCR": ["sum", "mean"], "Ag-RDT": ["sum", "mean"]}
    colFuncsSymptoms = {
        symptom: ["sum", "mean"] for symptom in sorted(symptomDisplayDict.values())
    }
    colFuncsDict = {**colFuncsFeatures, **colFuncsTests}

    writeTableEmployees(OUTPUT_TABLE_EMPLOYEES, dfTable, colFuncsDict)
    writeTableSymptoms(OUTPUT_TABLE_SYMPTOMS, dfTable, colFuncsSymptoms)


def createDataFramesFigures(df):

    # Date frame containing only PCRs with corresponding Ag-RDT results.
    dfAgrdt = df.dropna(subset=["agrdt"]).copy()

    dfAllInd = dataFrameIndependent(df.dropna(subset=("pcrPositive", "agrdt")))

    # Data frame containing only positive PCRs.
    dfAllPos = dataFramePCRpos(df)

    # Data frames containing only rapid test and PCR data with positive PCRs.
    dfPos = dataFrameAgrdt(dfAllPos)
    dfSymp = dataFrameSymptoms(dfPos)

    # Data frames containing all data with positive PCRs.
    dfAllPosNoRelease = removeReleaseTesting(dfAllPos)

    # Use only the data points corresponding to the first positive PCR of a person
    # (within an infection). Exclude people who presented for release testing.
    dfAllFirstPosPcrsNoRelease = dataFrameFirstPosPcr(dfAllPosNoRelease)
    dfAllFirstPosPcrsSympNoRelease = dataFrameSymptoms(dfAllFirstPosPcrsNoRelease)

    # Figure 1
    # Show only symptomatic people, remove tests from release testing, take the first
    # positive Pcr of an infection.
    dfFigure1 = dataFrameFirstPosPcr(removeReleaseTesting(dfSymp))
    # Only consider PCR tests after Nov 2020 as there are very few data points before
    # and we don't want to display
    # them when stratifying by sampling month (as we start in December).
    dfFigure1 = dfFigure1[dfFigure1.pcrDate >= date(2020, 12, 1)].copy()
    dfFigure1 = dfFigure1.dropna(subset=["samplingMonth2"]).copy()
    # Remove single datapoint in June-July bin.
    dfFigure1_A = dfFigure1[dfFigure1.samplingMonth2 != 3].copy()
    dfFigure1_B = dfFigure1_A.copy()
    # Remove the five datapoints with 4 immunisations (would be hardly visible in the
    # plot).
    dfFigure1_C = dfFigure1_A[dfFigure1_A.immunN < 4].copy()

    # Figure 2
    # PCRs typed during the delta-omicron transition period have higher viral loads
    # (because that's how they are selected for typing), so viral loads from these
    # samples are not representative for VOCs delta/omicron.
    # Also, release testing (done at the end of an infection) is not representative for
    # viral loads --> don't use.
    # dfFigure2 = dfAllFirstPosPcrsSympNoRelease.copy()
    # dfFigure2 = removeDeltaOmicronTyping(dfAllFirstPosPcrsSympNoRelease).copy()
    dfFigure2 = removeAllUnclearVariantOrTypingPCRs(
        dfAllFirstPosPcrsSympNoRelease
    ).copy()
    dfFigure2 = dfFigure2[dfFigure2.pcrDate >= date(2020, 12, 1)].copy()

    dfFigure2_A = dfFigure2.dropna(subset=["daysPostOnset", "variant"])
    dfFigure2_B = dfFigure2.dropna(subset=["variant", "agrdtYN"])

    # Figure 3
    # Data frame for Figure 3
    dfFigure3 = dataFrameFirstPosPcr(removeReleaseTesting(dfPos))
    # dfFigureA2 = dataFrameFirstPosPcr(removeReleaseTesting(dfPos))
    # dfFigureA2 = dataFrameFirstPosPcr(
    #     removeReleaseTesting(removeDeltaOmicronTyping(dfPos))
    # )
    dfFigureA2 = dataFrameFirstPosPcr(
        removeReleaseTesting(removeAllUnclearVariantOrTypingPCRs(dfPos))
    )

    return (
        dfAgrdt,
        dfPos,
        dfAllInd,
        dfFigure1,
        dfFigure1_A,
        dfFigure1_B,
        dfFigure1_C,
        dfFigure2,
        dfFigure2_A,
        dfFigure2_B,
        dfFigure3,
        dfFigureA2,
    )

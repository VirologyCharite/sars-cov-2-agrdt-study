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
    Add a column C{jitterCol} with random jittering applied to the values in C{origCol}.
    @param df: A C{pd.DataFrame}.
    @param origCol: The name of the column which jittering should be applied to.
    @param jitterCol: The name of the column to be created, containing the values that
    have been jittered.
    """
    stdev = sd * (df[origCol].max() - df[origCol].min())
    df.loc[:, jitterCol] = df[origCol] + np.random.randn(len(df[origCol])) * stdev


def dataFrameAgrdt(df):
    """
    @param df: A C{pd.DataFrame} with a column "agrdt".
    @return: A C{pd.DataFrame} containing only tests with an Ag-RDT result.
    """
    return df.dropna(subset=("agrdt"))


def dataFrameFirstPosPcr(df):
    """
    Return a data frame containing only data points that correspond to the first
    positive PCR of a person within an infection.

    @param df: A C{pd.DataFrame} with a column "isFirstPosPcr".
    @return: A C{pd.DataFrame} containing only tests that mark the first positive PCR
    in an infection.
    """
    return df[df.isFirstPosPcr == 1].copy()


def dataFrameIndependent(df):
    """
    @param df: A C{pd.DataFrame} with columns "pcrDate" and "personHash".
    @return: A dataframe that contains only one test per person (the first) so that
    test data points are independent of each other.
    """
    # Make sure that values are sorted by pcr date so that only the first test will be
    # kept for each person to keep things consistent.
    df = df.sort_values("pcrDate", ignore_index=True)
    return df.drop_duplicates(subset=["personHash"], keep="first")


def dataFramePCRpos(df):
    """
    @param df: A C{pd.DataFrame} with a column "pcrPositive".
    @return: A C{pd.DataFrame} containing only data with positive PCR result.
    """
    return df[df.pcrPositive].copy()


def dataFrameSymptoms(df):
    """
    @param df: A C{pd.DataFrame} with a column "symptoms".
    @return: A dataframe that only has tests in them where the respective person was
    symptomatic.
    """
    return df[df.symptoms == 1].copy()


def getPCRsOutsideVariantPrevalentRanges(df):
    """
    @param df: A C{pd.DataFrame} with a column "pcrDate".
    @return: A C{pd.DataFrame} containing only PCR tests from SARS-CoV-2 variant
    transition time, i.e. when no variant made up >= 90% of samples in Berlin.
    """
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
    @param df: A C{pd.DataFrame} with a column "pcrDate".
    @return: A C{pd.DataFrame} containing only PCR tests from the time of variant
    transition from VOC Delta to VOC Omicron.

    Return data points from the time of Delta/Omicron transition.
    """
    return df[
        (df.pcrDate > DATE_RANGE_DOMINANT_DELTA[1])
        & (df.pcrDate < DATE_RANGE_DOMINANT_OMICRON[0])
    ].copy()


def removeDeltaOmicronPCRs(df):
    """
    @param df: A C{pd.DataFrame} with a column "pcrDate".
    @return: A C{pd.DataFrame} containing no PCR tests from the time of variant
    transition from VOC Delta to VOC Omicron.

    Remove data points from the time when Delta/Omicron typing PCRs were done
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
    @param df: A C{pd.DataFrame} with a column "pcrDate".
    @return: A C{pd.DataFrame} containing only PCR tests from the time of variant
    transition from Wildtype to VOC Alpha.

    Return data points from the time of Wildtype/Alpha transition.
    """
    return df[
        (df.pcrDate > DATE_RANGE_DOMINANT_WILDTYPE[1])
        & (df.pcrDate < DATE_RANGE_DOMINANT_ALPHA[0])
    ].copy()


def removeWtAlphaPCRs(df):
    """
    @param df: A C{pd.DataFrame} with a column "pcrDate".
    @return: A C{pd.DataFrame} containing no PCR tests from the time of variant
    transition from Wildtype to VOC Alpha.

    Remove data points from the time when Wildtype/Alpha typing PCRs were done
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
    @param df: A C{pd.DataFrame} with a column "pcrDate".
    @return: A C{pd.DataFrame} containing no PCR tests from the time of variant
    transition from VOC Alpha to VOC Delta.

    Remove data points from the time when Alpha/Delta typing PCRs were done
    (May 2021 - July 2021) so that they don't give a false impression of viral
    load levels for those variants (typing PCRs in this period were done for
    high viral load level samples only).
    """
    return df[
        (df.pcrDate <= DATE_RANGE_DOMINANT_ALPHA[1])
        | (df.pcrDate >= DATE_RANGE_DOMINANT_DELTA[0])
    ].copy()


def removeAllUnclearVariantOrTypingPCRs(df):
    """
    @param df: A C{pd.DataFrame} with a column "pcrDate".
    @return: A C{pd.DataFrame} containing no PCR tests from the time of variant
    transition from Wildtype to VOC Alpha, VOC Alpha to VOC Delta or VOC Delta to
    VOC Omicron.
    """
    df = removeWtAlphaPCRs(df)
    df = removeAlphaDeltaPCRs(df)
    df = removeDeltaOmicronPCRs(df)
    return df


def removeReleaseTesting(df):
    """
    @param df: A C{pd.DataFrame} with a column "releaseTesting".
    @return: A C{pd.DataFrame} containing no PCR tests from release testing,
    i.e. PCR testing 7 days after a first positive PCR, in order to go back to work.
    """
    return df[df.reasonPres != "releaseTesting"].copy()


def standardize(df, origCol, standCol):
    """
    @param df: A C{pd.DataFrame}.
    @param origCol: The name of the column whose values you want to standardize
    (turn into z-scores).
    @param standCol: The name of the column to be created, containing the standardized
    values.
    """
    df[standCol] = (df[origCol] - df[origCol].mean()) / df[origCol].std()


def dataFrameNoRecovered(df):
    """
    @param df: A {pd.DataFrame} with a column "recovered".
    @return: A C{pd.DataFrame} containing no PCRs of people who we know have had a
    previous SARS-CoV-2 infection at the time of testing.
    """
    return df[df.recovered != 1].copy()


def dataFrameFemaleMale(df):
    """
    @param df: A {pd.DataFrame} with a column "gender".
    @return: A C{pd.DataFrame} containing only PCRs of people whose gender is either "M"
    or "F".
    """
    return df[df.gender.isin(("F", "M"))].copy()


def IQRQuartiles(series):
    """
    @param df: A {pd.Series} with numerical values.
    @return: A C{tuple} containing the first and the third quartile of the values in
    C{series}.
    """
    return tuple(series.quantile([0.25, 0.75]))


def IQR(series):
    """
    @param df: A {pd.Series} with numerical values.
    @return: The interquartile range (IQR) of the values in C{series}.
    """
    q25, q75 = IQRQuartiles(series)
    return q75 - q25


def roundHalfUp(value, decimals=2):
    """
    @param value: A C{float}.
    @param decimals: The C{int} number of decimals to round to.
    @return: A rounded C{float}, when rounding up from half.
    """
    with decimal.localcontext() as ctx:
        d = decimal.Decimal(value)
        ctx.rounding = decimal.ROUND_HALF_UP
        return float(round(d, decimals))


def mapRoundHalfUp(values, decimals=2):
    """
    @param values: A C{iterable} of C{float} values.
    @param decimals: The C{int} number of decimals to round to.
    @return: A C{list} of rounded C{float} values, when rounding up from half.
    """
    return [roundHalfUp(value, decimals=decimals) for value in values]


def createNewString(stat1, stat2, stat1Value, stat2Value):
    """
    @param stat1: A C{str} describing the first summary statistic.
    @param stat2: A C{str} describing the second summary statistic
    @param stat1Value: A C{int} or C{float} summary statistic.
    @param stat2Value: A single or an C{iterable} of C{int} or C{float} summary
    statistics.
    @return: A C{str} containing the value of both summary statistics, rounded to two
    decimals places.
    """
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
    """
    @param outfile: A path to the file the output should be written to.
    @param df: A C{pd.DataFrame} containing columns that characterize the study
    population.
    @param colFuncsDict: A C{dict} mapping a column name to (multiple) functions
    that compute a summary statistic on the column.
    A summary table of population characteristics will be created.
    """

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
    """
    @param outfile: A path to the file the output should be written to.
    @param df: A C{pd.DataFrame} containing binary columns indicating whether the
    person experienced the symptom.
    @param colFuncsDict: A C{dict} mapping a column name to (multiple) functions
    that compute a summary statistic on the column.
    A summary table of symptom counts will be created.
    """
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
    """
    @param df: A C{pd.DataFrame} containing the study's data.
    Two summary tables (tsv files) will be created. One specifying symptoms listed in
    the questionnaire and corresponding counts, the other one providing counts on
    gender, age, symptomatic status and PCR tests and Ag-RDTs.
    """
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
    """
    @param df: A C{pd.DataFrame} containing the study's data.
    @return: A C{tuple} of C{pd.DataFrames} used for the different figures.
    """
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

import sys
import pandas as pd
import numpy as np
import pymc as pm
import bambi as bmb
import arviz as az

from pandas.api.types import is_numeric_dtype
from itertools import combinations
from pathlib import Path

from utils.dataUtils import (
    dataFrameIndependent,
    dataFrameNoRecovered,
    dataFrameSymptoms,
    dataFrameFemaleMale,
    dataFramePCRpos,
    standardize,
)

SEED = 20

IND_CAT_VARS_AGRDT_SENS = (
    "testDevice",
    "gender",
    "binDaysPostOnset",
    "variant",
)
IMMUN_VARS = (
    "vaccYN5",
    "vaccYN5_2",
    "vaccYN6",
    "vaccYN6_2",
    "vaccYN7",
    "vaccYN7_2",
    "immun2YN",
)

IMMUN_VAR = "immun2YN"
IND_CONT_VARS_AGRDT_SENS = ("age", "vl")
IND_CONT_VARS_TESTLINE = ("age", "vl")

CATEGORY_ORDERS = {
    "variant": ("wildtype", "alpha", "delta", "omicron"),
    "gender": ("F", "M"),
    "testCentre": ("CCM", "CVK", "CBF"),
    "testDevice": ("abbott", "roche"),
}

IND_CONT_VARS_AGRDT_SENS_Z = tuple(
    f"z{contVar[0].upper() + contVar[1:]}" for contVar in IND_CONT_VARS_AGRDT_SENS
)

IDATA_DIR = Path("..", "output", "iData")
IDATA_DIR_RESUBMISSION2_2 = Path("..", "paper", "20240316-resubmission-2", "iData")
IDATA_DIR_NOV_23 = Path("..", "output", "iDataNov23")


def returnIDataDir():
    return IDATA_DIR

def returnIDataDirResubmission2_2():
    return IDATA_DIR_RESUBMISSION2_2


def returnIDataDirNov23():
    return IDATA_DIR_NOV_23


def _regressionVars(immunVar=None):
    immunVar = (immunVar,) if immunVar else ()

    indVars = IND_CONT_VARS_AGRDT_SENS
    indCatVars = IND_CAT_VARS_AGRDT_SENS + immunVar

    zIndVars = tuple(f"z{indVar[0].upper() + indVar[1:]}" for indVar in indVars)

    return indVars, zIndVars, indCatVars


def logisticRegressionVars(immunVar=None):
    """
    Return C{tuples} of the input variables (not including specific symptoms) for the
    regression on Ag-RDT sensitivity as the outcome (and therefore considering only PCR
    positive samples.)

    @param immunVar: A C{str} indicating which variable specifying the
        vaccination status to include as an input variable in the regression.
    @return: Three C{tuples}, the first one containing C{str}s corresponding to the
    continuous variables to be used in the regression, the second one containing
    C{str}s corresponding to the standardized continuous variables used in the
    regression and the last one containing the C{str}s corresponding to the categorical
    variables used in the regression.
    """

    return _regressionVars(immunVar)


def makeRegressionDf(df, indVars, indCatVars, categories):
    """
    Process C{df} so it can be used for regression.

    @param df: A C{pd.DataFrame} containing the columns specified in C{indVars} and
        C{indCatVars}.
    @param indVars: An C{iterable} of continuous independent variables used in the
        regression.
    @param indCatVars: An C{iterable} of categorical independent variables used in the
        regression.
    @param categories: A nested C{dict} mapping the categories of one or more
        categorical variables to their designated categorical values.
    @return: The original C{df} with the values in categorical columns turned into codes
        and additional columns for the standardized values of the continuous variables.
        Also, all rows with NaN values in any of of C{indVars} and C{indCatVars} are
        removed from the returned DataFrame.
    """
    allIndVars = list(indVars) + list(indCatVars)
    df = df.dropna(subset=allIndVars).copy()

    for indCatVar in indCatVars:
        if indCatVar == "pcrId":
            continue
        if indCatVar in categories:
            df[indCatVar] = pd.Categorical(
                df[indCatVar], categories=categories[indCatVar]
            ).codes
        else:
            df[indCatVar] = pd.Categorical(df[indCatVar]).codes

    zIndVars = []
    for indVar in indVars:
        zIndVar = f"z{indVar[0].upper() + indVar[1:]}"
        zIndVars.append(zIndVar)
        standardize(df, indVar, zIndVar)

    return df[allIndVars + zIndVars].copy()


def logisticRegressionDf(
    df, immunVar=None, outcome="agrdt", indVars=None, indCatVars=None, replaceVals=None
):
    """
    Create a C{pd.DataFrame} only containing the data relevant for the regression
    (with AgRDT sensitivity as the outcome and not using specific symptoms as input
    variables).

    @param df: A C{pd.DataFrame} containing all the variables relevant for the
        regression on AgRDT sensitivity as the outcome as columns.
    @param immunVar: A C{str} indicating which variable specifying the vaccination
        status to include as an input variable in the regression.
    @param outcome: A C{str} specifying the outcome variable of the regression.
    @param indVars: An C{iterable} of C{str}s specifying the continuous independent
        variables to use in the regression.
    @param indCatVars: An C{iterable} of C{str}s specifying the categorical independent
        variables to use in the regression.
    @param replaceVals: A C{dict} specifying for each column whose values should be
        replaced the mapping from old to new value.
    @return: A C{pd.DataFrame} containing only the relevant columns (+ pcrId) for the
        regression.
    """
    if not indVars:
        indVars, _, _ = logisticRegressionVars(immunVar=immunVar)
    if not indCatVars:
        _, _, indCatVars = logisticRegressionVars(immunVar=immunVar)

    if immunVar in indCatVars and immunVar not in IMMUN_VARS:
        print("Excluding recovered people", file=sys.stderr)
        df = dataFrameNoRecovered(df)
    dfRegr = dataFrameIndependent(
        dataFrameSymptoms(dataFrameFemaleMale(dataFramePCRpos(df)))
    )
    dfRegr = makeRegressionDf(
        dfRegr, indVars, indCatVars + (outcome, "pcrId"), categories=CATEGORY_ORDERS
    )
    if replaceVals is not None:
        dfRegr.replace(replaceVals, inplace=True)

    return dfRegr


def _calcPosteriorStatsProb(
    iData, catVar, origVarname, cats, catDict, statsDict, hdi_prob
):
    """
    Calculate posterior statistics from inference data coming from MCMC sampling.

    @param iData: An C{arviz Inference Data} object containing the posterior
        samples from running an MCMC simulation with only C{var} as independent
        variable.
    @param catVar: An C{str} specifying the input variable.
    @param origVarname: A C{str} specifying the original name of the input variable
        used in the logistic regression. For example, "Cat" could be added to the
        original name to make explicit that it is a categorical variable and the
        categories' codes are being used to specify each category.
    @param cats: An C{iterable} of the categories of C{catVar}.
    @param catDict: A C{dict} mapping the original categories' names of C{catVar} to
        those currently used (e.g. might be codes instead of C{str}s).
    @param statsDict: A nested C{defaultdict} with three levels with the following
        keys on each level:
        level 1: The variable (C{var}) of interest, e.g. "symptoms",
        level 2: The corresponding categories (C{cats}) of interest, e.g. 0 or 1.
        level 3: The posterior samples of probabilities ("prob"), e.g. for a
        positive AgRDT result or PCR positivity, the corresponding 94% credible
        interval ("hdi") and the corresponding mean ("mean"). Thus the probabilities
        correspond to estimates for AgRDT sensitivity or PCR positive rate.
    @param hdi_prob: The size of the highest posterior density interval.
    """
    dimVar = f"{catVar}_coord"
    for origCat, cat in catDict.items():
        postSamples = iData.posterior[catVar].sel({dimVar: cat}).values.flatten()
        # probPostSamples = logistic(postSamples)

        statsDict[origVarname][origCat]["prob"] = postSamples
        statsDict[origVarname][origCat]["hdi"] = az.hdi(postSamples, hdi_prob=hdi_prob)
        statsDict[origVarname][origCat]["mean"] = np.mean(postSamples)

    for origCat1, origCat2 in combinations(cats, 2):
        # For some categories there might not be any data and thus we cannot compute
        # the differences.
        try:
            diffProb = (
                statsDict[origVarname][origCat1]["prob"]
                - statsDict[origVarname][origCat2]["prob"]
            )
        except KeyError:
            continue
        hdiDiff = az.hdi(diffProb, hdi_prob=hdi_prob)

        statsDict[origVarname][(origCat1, origCat2)]["prob"] = diffProb
        statsDict[origVarname][(origCat1, origCat2)]["hdi"] = hdiDiff
        statsDict[origVarname][(origCat1, origCat2)]["mean"] = np.mean(diffProb)


def sampleProb(
    df,
    outcome="agrdt",
    catVars=None,
    cats=None,
    seed=SEED,
    target_accept=0.9,
    bambi=True,
    interaction=True,
    formula=None,
    priors=None,
    newData=None,
):
    """
    Sample the posterior of a probability (e.g. probability of a positive Ag-RDT
    result (sensitivity) or probability of being PCR positive (PCR positive rate).

    @param df: A C{pd.DataFrame} containing the relevant data for the
        sampling (i.e. input variable (if any) and outcome as columns).
    @param outcome: A C{str} specifying the outcome variable.
    @param catVars: A C{iterable} specifying the categorical variables (if any) to
        stratify by.
    @param cats: A C{list} or C{dict} with values being lists (if more than one input
        variable is passed), specifying the categories/levels of C{var}.
    @param seed: The random seed used for sampling of the posterior.
    @param target_accept: A C{float} specifying the target acceptance rate
        for MCMC sampling.
    @param bambi: A C{bool} specifying whether to use bambi for inference.
    @param interaction: A C{bool} specifying whether to model an interaction effect
        (as opposed to an additive effect) for the passed variables in {catVars}.
    @param formula: A C{str} specifying the formula to be used (when bambi is set to
        True).
    @param priors: A C{dict} specifying the priors for one or more input variables of
        the model. The C{dict} has to be usable by bambi.
    @param newData: A C{pd.DataFrame} with data to make posterior predictions with.
    @return: The bambi model used for inference and posterior samples either as C{
        np.ndarray} or as C{az.InferenceData} object.
    """
    if formula is not None:
        assert bambi
    dropnaCols = [outcome]
    coords = {}
    catVars = list(catVars)

    returnInferenceData = False
    if catVars is not None:
        dropnaCols.extend(catVars)
        returnInferenceData = True
    dfCurr = df.dropna(subset=dropnaCols).copy()
    categoricals = catVars + [outcome]
    for currVar in categoricals:
        if is_numeric_dtype(dfCurr[currVar]):
            if dfCurr[currVar].dtype == np.float64:
                dfCurr = dfCurr.astype({currVar: np.int64})
    # Make sure that there is only one data point per person.
    dfCurr = dataFrameIndependent(dfCurr)

    if bambi:
        assert catVars
        assert newData is not None

        if formula is None:
            joinBy = ":" if interaction else " + "
            formula = f"{outcome} ~ 0 + {joinBy.join(catVars)}"
        model = bmb.Model(
            data=dfCurr,
            formula=formula,
            family="bernoulli",
            categorical=catVars,
            dropna=True,
            noncentered=True,
            priors={} if priors is None else priors,
        )
        model.build()
        iData = model.fit(
            target_accept=target_accept, tune=4000, draws=10000, random_seed=seed
        )
        # We have to make posterior predictions to get probabilities.
        model.predict(iData, data=newData)
        # Rename to orginal variable name.
        return model, iData

    else:
        assert priors is None
        assert len(catVars) == 1, (
            "Currently, the PyMC version of estimating "
            "probabilities has been only implemented for the "
            "case of one input variable."
        )
        if catVars:
            var = catVars[0]
            coordName = f"{var}_coord"
            coords[coordName] = cats[var]
        with pm.Model(coords=coords) as model:
            if catVars:
                # Non-centered implementation, see
                # https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-
                # centered/
                # mu = pm.Normal('mu', mu=0, sd=2.5)
                # sd = pm.HalfNormal('sd', sigma=2.5)
                # muOffset = pm.Normal('muOffset', mu=0, sd=1, dims=coordName)
                # catVar = pm.Normal(paramName, mu + sd * muOffset, dims=coordName)
                # prob = pm.Deterministic('prob', pm.invlogit(catVar[dfCurr[var]]))
                probParam = pm.Beta(var, alpha=1, beta=1, dims=coordName)
                prob = probParam[dfCurr[var]]
            else:
                prob = pm.Beta("prob", alpha=1, beta=1)

            pm.Bernoulli("obs", prob, observed=dfCurr[outcome])
            postSamples = pm.sample(
                draws=10000,
                tune=4000,
                chains=4,
                cores=4,
                random_seed=seed,
                target_accept=target_accept,
                return_inferencedata=returnInferenceData,
            )

    return model, postSamples


def sampleProb2(
    df,
    catVar,
    cats,
    statsDict,
    target_accept=0.9,
    hdi_prob=0.94,
    outcome="agrdt",
    seed=SEED,
):
    """
    Run a logistic regression with MCMC, specifying an outcome and ONE independent
    variable.

    @param df: A C{pd.DataFrame} containing a column C{catVar}.
    @param catVar: A C{str} specifying the independent variable used in the logistic
        regression.
    @param cats: An C{iterable} containing the categories (in the desired order) of
        C{catVar}.
    @param statsDict: A nested C{defaultdict} with three levels with the following
        keys on each level:
        level 1: The variable (C{catVar}) of interest, e.g. "symptoms",
        level 2: The corresponding categories (C{cats}) of interest, e.g. 0
        or 1.
        level 3: The posterior samples of probabilities ("prob"), e.g. for a
        positive AgRDT result or PCR positivity, the corresponding 94%
        credible interval ("hdi") and the corresponding mean ("mean"). Thus
        the probabilities correspond to estimates for AgRDT sensitivity or
        PCR positive rate.
    @param target_accept: A C{float} specifying the target acceptance rate for MCMC
        sampling.
    @param hdi_prob: The size of the highest posterior density interval.
    @param outcome: A C{str} specifying the variable used as the outcome in the
        logistic regression.
    @param seed: The random seed used for sampling of the posterior.
    """
    assert catVar is None and cats is None or catVar is not None and cats is not None

    # If there is only one category or no category at all (i.e. we just want to get the
    # 94% CIs for an outcome like AgRDT sensitivity).
    if catVar is None:
        model, trace = sampleProb(
            df, catVars=[outcome], seed=seed, target_accept=target_accept, bambi=False
        )
        statsDict["all"]["prob"] = trace["prob"]
        statsDict["all"]["mean"] = np.mean(trace["prob"])
        statsDict["all"]["hdi"] = np.quantile(trace["prob"], (0.03, 0.97))
    elif len(cats) == 1:
        origVarname = f"{catVar.replace('Code', '')}"
        # If we have only one category for which we want to determine
        # posterior statistics (e.g. the 94% credible interval):
        cat = cats[0]
        dfCat = df[df[catVar] == cat].copy()
        model, trace = sampleProb(
            dfCat,
            catVars=None,
            outcome=outcome,
            seed=seed,
            target_accept=target_accept,
            bambi=False,
        )
        statsDict[origVarname][cat]["prob"] = trace["prob"]
        statsDict[origVarname][cat]["mean"] = np.mean(trace["prob"])
        statsDict[origVarname][cat]["hdi"] = np.quantile(trace["prob"], (0.03, 0.97))
    else:
        # If there are at least two categories (e.g. asymptomatic vs symptomatic) for
        # which we want to get the 94% CIs of the outcome.
        dfCurr = df.dropna(subset=[catVar]).copy()
        catVarOrig = catVar
        # Create codes for non-numeric or interval variables.
        # Check if categories in C{var} are floats instead of integers. If so, convert
        # to integer.
        catDict = {}
        if is_numeric_dtype(dfCurr[catVar]):
            if dfCurr[catVar].dtype == np.float64:
                dfCurr = dfCurr.astype({catVar: np.int64})
            catDict[catVar] = dict(zip(cats, cats))
        else:
            var = f"{catVar}Code"
            dfCurr[var] = pd.Categorical(dfCurr[catVarOrig]).codes
            # Get new categorical value for each original value.
            catDict[catVarOrig] = {
                dfCurr[dfCurr[var] == cat][catVarOrig].iloc[0]: cat
                for cat in set(dfCurr[var])
            }

        model, iData = sampleProb(
            dfCurr,
            outcome,
            catVars=[catVar],
            cats=catDict,
            seed=seed,
            target_accept=target_accept,
        )
        _calcPosteriorStatsProb(
            iData=iData,
            catVar=catVar,
            origVarname=catVarOrig,
            cats=cats,
            catDict=catDict,
            statsDict=statsDict,
            hdi_prob=hdi_prob,
        )


def sampleAgrdtSpec(
    df,
    var,
    cats,
    newData,
    seed=SEED,
    target_accept=0.9,
    randomEffects=True,
):
    """
    @param df: A C{pd.DataFrame} containing the relevant data for the sampling (i.e.
        input variable (if any) and outcome as columns).
    @param var: A C{str} specifying the categorical variable (if any) to stratify by.
    @param cats: A C{list} specifying the categories/levels of C{var}.
    @param newData: A C{pd.DataFrame} with data to make posterior predictions with.
    @param seed: The random seed used for sampling of the posterior.
    @param target_accept: A C{float} specifying the target acceptance rate for MCMC
        sampling.
    @param randomEffects: Model C{var} as random effects variable.
    @return: The posterior mean(s) and HPDI(s) of the test's specificity.
    """

    # For calculating specificity we need to consider only PCR negative samples. If
    # we switch the labels (i.e. 0 to 1 and 1 to 0) and compute the agrdt positive
    # rate, we get the specificity.
    dfNeg = df[df.pcrPositive == 0].copy()
    dfNeg.agrdt.replace({0: 1, 1: 0}, inplace=True)
    formula = None
    if randomEffects:
        formula = f"agrdt ~ (1|{var})"
    return sampleProb(
        dfNeg,
        "agrdt",
        catVars=[var],
        cats=cats,
        seed=seed,
        target_accept=target_accept,
        bambi=True,
        newData=newData,
        formula=formula,
    )


def sampleVl(
    df,
    catVars,
    likelihood="skewnormal",
    interaction=True,
    formula=None,
    priors=None,
    tune=4000,
    draws=10000,
    seed=SEED,
    target_accept=0.9,
):
    """
    Samples the posterior of mean viral load using a Normal or Skew Normal distribution.

    @param df: A C{pd.DataFrame} containing the relevant data for the
        sampling (i.e. input variable (if any) and outcome as columns).
    @param catVars: A C{iterable} specifying the categorical variables (if any) to
        stratify by.
    @param likelihood: A C{str} specifying the likelihood function to use (must be
        either "normal" or "skewnormal").
    @param interaction: A C{bool} specifying whether to model an interaction effect
        (as opposed to an additive effect) for the passed variables in {catVars}.
    @param formula: A C{str} specifying the formula to be used (when bambi is set to
        True).
    @param priors: A C{dict} specifying the priors for one or more input variables of
        the model. The C{dict} has to be usable by bambi.
    @param seed: The random seed used for sampling of the posterior.
    @param target_accept: A C{float} specifying the target acceptance rate for MCMC
        sampling.
    @return: The posterior samples as an C{az.InferenceData} object.
    """
    assert "vl" in df.columns
    assert likelihood in (
        "normal",
        "skewnormal",
    ), f'Likelihood must be either "normal" or "skewnormal" but is "{likelihood}".'
    dropnaCols = ["vl"]
    vars = list(catVars)
    dropnaCols.extend(vars)
    dfCurr = df.dropna(subset=dropnaCols).copy()
    # Make sure that there is only one data point per person
    dfCurr = dataFrameIndependent(dfCurr)
    statsVl = {"mean": dfCurr.vl.mean(), "sd": dfCurr.vl.std()}
    standardize(dfCurr, "vl", "zVl")

    for catVar in catVars:
        if is_numeric_dtype(dfCurr[catVar]):
            if dfCurr[catVar].dtype == np.float64:
                dfCurr = dfCurr.astype({catVar: np.int64})

    if likelihood == "skewnormal":
        likelihood = bmb.Likelihood(
            "SkewNormal",
            parent="mu",
            sigma=bmb.Prior("HalfStudentT", nu=4, sigma=1),
            alpha=bmb.Prior("Normal", mu=0, sigma=5),
        )
        link = bmb.Link("identity")
        family = bmb.Family("skewnormal", likelihood, link)
    else:
        family = "gaussian"
    if formula is None:
        joinBy = ":" if interaction else " + "
        formula = f"zVl ~ {vars[0]}" if len(vars) == 1 else f"zVl ~ {joinBy.join(vars)}"
    model = bmb.Model(
        data=dfCurr,
        formula=formula,
        categorical=vars,
        dropna=True,
        family=family,
        priors={} if priors is None else priors,
    )
    model.build()
    iData = model.fit(
        target_accept=target_accept,
        tune=tune,
        draws=draws,
        random_seed=seed,
    )
    return model, iData, statsVl


def generateNewDataAgrdtSens(df, predDict, feature, featureCats):
    """
    Create new data for making posterior predictions for each category in C{feature}.

    @param df: The C{pd.DataFrame} containing the data the model was fitted with.
    @param predDict: A C{dict} containing the new data to make predictions with for
        each C{feature} of interest.
    @param feature: A C{str} specifying the feature whose categories we want to make
        predictions for (e.g. predictions of antigen test sensitivity|viral load for
        each SARS-CoV-2 variant).
    @param featureCats: An C{iterable} of the categories in C{feature}.
    """
    vlMean = df.vl.mean()
    vlSd = df.vl.std()
    nCats = len(featureCats)

    vl = np.arange(3, 11, step=0.1)
    zVl = (vl - vlMean) / vlSd
    nVl = len(zVl)
    n = nCats * nVl

    dtype = np.int32

    newData = pd.DataFrame(
        {
            "testDevice": np.zeros(n, dtype=dtype),
            "gender": np.zeros(n, dtype=dtype),
            "binDaysPostOnset": np.zeros(n, dtype=dtype),
            "variant": np.full(n, 3, dtype=dtype),
            IMMUN_VAR: np.ones(n, dtype=dtype),
            "zAge": np.zeros(n),
            "zVl": np.tile(zVl, nCats),
        }
    )

    assert list(newData.columns) == (
        list(IND_CAT_VARS_AGRDT_SENS) + [IMMUN_VAR] + list(IND_CONT_VARS_AGRDT_SENS_Z)
    )
    # Update.
    newData[feature] = np.repeat(list(featureCats), nVl)
    predDict[feature]["data"] = newData
    predDict[feature]["vl"] = vl


def predictionsNewData(df, model, iData, predDict, feature, featureCats):
    """
    Make posterior predictions of Ag-RDT sensitivity for each
    level/category in C{feature} and store them in C{predDict}.

    @param df: The C{pd.DataFrame} containing the data the model was fitted with.
    @param model: A C{bambi} or C{pymc} logistic regression model with antigen test
        result as the outcome.
    @param iData: An C{arviz Inference Data} object returned by fitting C{model}.
    @param predDict: A C{dict} containing the new data to make predictions with for
        each C{feature} of interest.
    @param feature: A C{str} specifying the feature whose categories we want to make
        predictions for (e.g. predictions of antigen test sensitivity|viral load for
        each SARS-CoV-2 variant).
    @param featureCats: An C{iterable} of the categories in C{feature}.
    """

    generateNewDataAgrdtSens(df, predDict, feature, featureCats)

    postOutcomeVarBambi = "agrdt_mean"

    model.predict(iData, data=predDict[feature]["data"], kind="mean")
    # See https://bambinos.github.io/bambi/main/notebooks/
    # logistic_regression.html
    # Note we stack chains and draws, then posteriorPreds has a shape of
    # (n_obs, n_chains * n_draws).
    posteriorPreds = (
        iData.posterior[postOutcomeVarBambi].stack(samples=("chain", "draw")).values
    )

    predDict[feature]["ppSamples"] = posteriorPreds


def calcErrors(df, statsDict, outcome, errors, means, catVar, cats, seed=SEED):
    """
    Calculate and add the errors of a binary outcome variable C{outcome} for the
    different categories (C{cats}) in the variable C{catVar}.

    @param df: A C{pandas.DataFrame} with a column C{catVar} containing the
    categories in C{cats} (e.g. 0 vs 1 for asymptomatic vs symptomatic).
    @param statsDict: A nested C{defaultdict} with three levels and the following
        keys on each level:
        level 1: The variable name for which posterior statistics were computed
        (e.g. C{catVar}).
        level 2: The category of variable C{catVar}.
        level 3: The posterior statistics/values we are interested in ("prob" for
        an array of all sampled probabilities, "mean" for the mean of the sampled
        probabilities and "hdi" for the 95% credible interval of the sampled
        probabilities).
    @param outcome: A C{str} specifying the outcome of interest which you want to
        compute the errors for, e.g. AgRDT sensitivity.
    @param errors: A C{defaultdict(list)} storing the computed errors (credible
        intervals) for each category.
    @param means: A C{defaultdict(list} storing the computed means for each category.
    @param catVar: A C{str} specifying the variable of interest, e.g. "symptoms" or
        "variant". The errors of the outcome (e.g. AgRDT sensitivity) will be computed
        after stratifying the data by each category of C{catVar}.
    @param cats: An C{iterable} containing the possible categories for C{catVar} (given
        in the order they appear in the plot).
    @param seed: The random seed used for sampling of the posterior.
    """
    # If there is only one category for a variable (e.g. for binVl = (10, 11], there
    # are only symptomatic people with a vl that high).
    catsOrig = cats
    if len(df[catVar].dropna().unique()) < 2:
        cats = list(df[catVar].dropna().unique())
    if catVar not in statsDict:
        # If there are any observations.
        if (df[catVar].notna() & df[outcome].notna()).sum():
            sampleProb2(
                df,
                catVar=catVar,
                cats=cats,
                statsDict=statsDict,
                target_accept=0.94,
                hdi_prob=0.94,
                outcome=outcome,
                seed=seed,
            )
    errorDict = statsDict[catVar]
    for cat in catsOrig:
        if (df[catVar] == cat).sum():
            means[cat].append(errorDict[cat]["mean"])
            errors[cat].append(errorDict[cat]["hdi"])
        else:
            # No data point --> no errors.
            means[cat].append(np.nan)
            errors[cat].append((np.nan, np.nan))

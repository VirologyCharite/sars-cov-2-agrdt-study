# SARS-CoV-2 rapid antigen test sensitivity and viral load in freshly symptomatic hospital employees in Berlin, Germany, December 2020 to February 2022 - an observational study

This repository contains data and code for "SARS-CoV-2 rapid antigen test sensitivity and viral load in freshly symptomatic hospital employees, December 2020 to February 2022 - an observational study".

## Data

There are three data files in the `data` directory.

### agrdtData.tsv

This file can be obtained upon request to the authors. It contains data from employee testing with the following columns:

1. `pcrId`: A unique Id, given to every RT-PCR test.
2. `pcrDate`: The date the RT-PCR was performed.
3. `personHash`: A unique Id for each person.
4. `pcrPositive`: Specifies whether the RT-PCR is positive.
5. `vl`: The logarithm base 10 of the RT-PCR estimated number of viral RNA
 molecules / mL.
6. `variant`: The SARS-CoV-2 variant of the sample, based on typing RT-PCR
 or epidemiological assignment.
7. `hasTyping`: Specifies whether a typing RT-PCR was performed.
8. `isFirstPosPcr`: Specifies whether the RT-PCR was the first positive one
 in a person's infection course.
9. `gender`: The person's gender (male or female).
10. `age`: The person's age.
11. `symptoms`: Specifies whether a person experiences Covid-19 like symptoms.
12. `daysPostOnset`: The number of days between onset of symptoms and
 presentation at the test centre (with a granularity of 12h during the
  first 24h of symptoms onset and a granularity of 24h thereafter).
13. `binDaysPostOnset`: The time period in which the test was performed
 relative to symptom onset (0-2 or 2-7 days post symptom onset).
14. `recovered`: Specifies whether a person had a prior infection at the
 time of testing.
15. `immun2YN`: Specifies whether the person was immunized at least twice
 (through vaccination and/or prior infection) at least 2 weeks prior to the 
 time of testing. Only people with no prior immunization will get a value of 0.
16. `immunYN`: Specifies whether the person was immunized at least once
 (though vaccination and/or prior infection) at the time of testing.
17. `immunN`: A person's number of prior immunizations (vaccination or prior
 infection).
18. `immunNatLeast`: A person's known number of prior immunizations (for
 some people, current immunization status wasn't available at the end of the
  study).
19. `vaccN`: A person's number of prior vaccinations.
20. `vaccNatLeast`: A person's known number of prior vaccination (for
 some people, current vaccination status wasn't available at the end of the
  study).
21. `surveyData`: Specifies whether data from the survey (e.g. about onset
 of symptoms) is available.
22. `agrdtYN`: Specifies whether an Ag-RDT was performed.
23. `agrdt`: Specifies the result of the Ag-RDT (0: negative, 1: positive).
24. `testDevice`: Specifies the test device that was used (Abbott Panbio
 COVID-19 Rapid Test Device or Roche SARS-CoV-2 Rapid Antigen Test).
25. `samplingMonth`: The time of sampling given in the following format
: YYYY-MM.
26. `samplingMonth2`: The time of sampling given in two-months intervals
 (starting in Dec. 2020).
27. `reasonPres`: The reason for presentation at the test centre.
28. `infectionKey`: A key specifying an infection.
29. 16 binary columns specifying presence of the following symptoms: feeling ill (`ill`),
    `fatigue`, `headache`, `vertigo`, `melalgia`, `fever`, `cough`, runny nose (`runnyNose`),
    sore throat (`soreThroat`), `dyspnea`, loss of sense of smell (`noSmell`), loss of sense
    of taste (`noTaste`), `nausea`, no appetite (`noAppetite`), `vomiting` and `diarrhea`.

### abbottVsRocheWildtype.tsv

Data from testing SARS-CoV-2 Wildtype samples using the Abbott Panbio COVID
-19 Rapid Test Device or the Roche SARS-CoV-2 Rapid Antigen Test. Contains
 the following columns:
 
 1. `vl`: The logarithm base 10 of the RT-PCR estimated number of viral RNA
 molecules / mL.
 2. `variant`: The SARS-CoV-2 variant of the sample, based on typing RT-PCR.
 3. `test`: The test device used, either the Abbott Panbio COVID-19 Rapid
  Test Device or the Roche SARS-CoV-2 Rapid Antigen Test (0: Abbott, 1: Roche).
 4. `agrdt`: Specifies the result of the Ag-RDT (0: negative, 1: positive).
 5. `zVl`: Viral load z-scores.
 
 
### abbottVsRocheOmicron.tsv

Data from testing SARS-CoV-2 Wildtype samples using the Abbott Panbio COVID
-19 Rapid Test Device or the Roche SARS-CoV-2 Rapid Antigen Test. Contains
 the following columns:
 
 1. `vl`: The logarithm base 10 of the RT-PCR estimated number of viral RNA
 molecules / mL.
 2. `variant`: The SARS-CoV-2 variant of the sample, based on typing RT-PCR.
 3. `batch`: Specifies the batch (tests were performed in two batches).
 4. `test`: The test device used, either the Abbott Panbio COVID-19 Rapid
  Test Device or the Roche SARS-CoV-2 Rapid Antigen Test (0: Abbott, 1: Roche).
 5. `agrdt`: Specifies the result of the Ag-RDT (0: negative, 1: positive).
 6. `zVl`: Viral load z-scores.
 

## Code
 
### Notebooks

Two jupyter notebooks, one containing the analysis comparing the performance
 of the Abbott Panbio COVID-19 Rapid Test Device and the Roche SARS-CoV-2
  Rapid Antigen Test (`abbottVsRoche.ipynb`), the other containing the main
   analyses on employee rapid test data (`agrdt.ipyb`).
   
### Utils

Utility functions for data manipulation (`dataUtils.py`), plotting
 (`plotUtils.py`) and statistical analyses (`regression.py`) and a file
  containing plotting parameter specifications (`plotParams.py`).
 



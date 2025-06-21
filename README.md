# Study of Decomposition Methods for Ordinal Quantification

This repository holds the code used for my Master's Dissertation (written in Spanish): [Study of Decomposition Methods for Ordinal Quantification](https://perezdavid-98.github.io/files/Master_Dissertation_Study_of_Decomposition_Methods_For_Ordinal_Quantification.pdf),
which can be read form the PDF in this repository or from my github.io page. 

I am currently working on translating the dissertation to English. 


## Abstract
The experiment carried out for this dissertation was to check the hypothesis that Decomposition Methods commonly used in Ordinal Classification, are not suitable for Ordinal Quantification problems, specifically the Frank & Hall decomposition method.

The basis of the hypothesis is that this method assumes a homogeneous distribution of classes in cases where they are grouped. This assumption is rarely accurate, causing these methods to perform worse than conventional methods.

To corroborate this hypothesis, two experiments were proposed in which five models widely used in the literature were trained:

* Adjusted Count (AC)
* Probabilistic Adjusted Count (PAC)
* Expectation Maximization (EM)
* Method based on the Hellinger Distance (HDy)
* Method based on the Energy Distance (EDy)

and compared against their counterparts using the F&H decomposition method. 
These experiments not only demonstrated that the decomposition methods are not adequate for ordinal quantification problems, but also refute the assumption of homogeneous distribution in the complementary class during the decomposition process.

## Experiment Overview

The two experiments proposed are described as follows:

* Experiment 1: In the first experiment, artificially generated datasets were used. In this case, a variable training dataset ranging from 50 to 2,000 examples in size is chosen, which will be evaluated on 300 test datasets of 2,000 examples each. That is, 300 test sets of size 2,000 will be available, which will be tested on training sets of variable size. This process was repeated k times (The dissertation was written using the results of k=10). Thus, the value reflected was be the average of 300 samples multiplied by 10 repetitions, that is, the average of 3,000 results. 
Two cases were presented: 
    * First, the classes were well-differentiated, without much overlap in borderline values.
    * Second, there was more overlap in borderline values, making the test more difficult.
* Experiment 2: In the second experiment, datasets obtained from real data were used. Again, two cases were presented:
    * In the first case, the dataset obtained from the [LeQua2024](https://lequa2024.github.io/) competition was used. Specifically, the dataset corresponding to task T3 of the competition was used, which corresponded to the ordinal case. 
    * For the second case, five datasets commonly used in the literature (Employee Selection (ESL), Lecture Evaluation (LEV), Social Workers Decisions (SWD), Boston Housing and Abalone) were used to formalize a test bench to obtain results on several datasets with real data.

The Error Metric used to check the results of the models was the Earth Mover's Distance (EMD). 
This metric measures the minimum amount of work required to transform one distribution into another, considering the distance between distributions.
Specifically, EMD calculates the probability mass that must be shifted to convert one distribution to another and ranges between 0 and k − 1 in this configuration. The lower the EMD value, the more similar the two distributions being compared will be, indicating that the predicted prevalences p' are close to the actual prevalences p.


## Code overview

The implementation is a bit messy. The files used to generate all the results are the following Python files:

* `tests_ordinal.py` For Experiment 1, cases 1 and 2. This script generates the artificial data and runs the tests, generating the results files located in the folders `artificial` and `artificial_preds` inside the `data` directory. It also generates some of the plots used in the dissertation.
* `tests_lequa.py` For Experiment 2, case 1. This file expects the data from the task T3 from the competition (it was not included in this repository as there were no changes made to it). The only requirement is to combine both zips found in the competition (train and test) inside the `data`folder, both have the same directory so just unzipping them in the same directory should suffice. 
* `tests_ordinal_dataset.py` For Experiment 2, case 2. Same as previous case, the original datasets are not included as no change was made to them. The only requirement is to use the ordinal versions of these datasets.

The rest of the scripts used were obtained from [LeQua2024_Scripts](https://github.com/HLT-ISTI/LeQua2024_scripts) and were used specifically to run the evaluations and to format the output of the experiment.

The `evaluate.py` script was modified, including the EMD metric to be used in the experiment.


Finally, the `plotter.py` Script generates the plots used in the dissertation. The only requirement is to create the `figures` directory inside the `data` folder as it was not included in the repo.
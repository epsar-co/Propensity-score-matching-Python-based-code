# Propensity_Score_Matching_Python-based_code
This repository offers 3 versions of a free, Python-based code for performing propensity score (PS) matching. As an initiative of the Camargo Cohort Study, it has been developed for clinicians and researchers with the aim of sharing the tool and disseminating the use of PS matching.

The code overcomes compatibility issues with R versions and R packages, and implements (i) logistic regression to compute PS, (ii) 1:N matching using the K-nearest neighbour (KNN) algorithm with a customisable caliper, (iii) sampling with/without replacement, and (iv) visualisations to assess matching quality. Outputs: Matched pairs stored as .csv file. Matched pairs can be easily identified, allowing a Coxreg to be performed. In addition, 5 diagnostic plots are saved in the specified output folder.

The code was developed using information from the Matplotlib, Numpy and Seaborn libraries and OpenAI's ChatGPT support and refinements. No funding was received for conducting this work and there are no financial or non-financial interests to disclose

Usage
Refine the code with your current research:
- Rename C:\PATH_TO_YOUR_DATASET.sav
- Rename COVS with your data (name, not label)
- Choose the ratio (1:1, 1:2...) and the caliper 
- Choose bar colors and adjust the limits of the x-axis and y-axis to the desired range
- Rename C:\PATH_TO_YOUR_FOLDER
Run the script [RStudio, SPSS (File / Open script)...]
 
Features:
* Code 1 provides a PSM through sampling without replacement. In addition, 5 plots showing PS distributions and SMD for testing the matching.
* Code 2 provides a PSM through sampling with replacement and the 5 plots.
* Code 3 provides a full PSM by sampling without replacement and a lineplot showing the SMD before and after matching. A colour assignment has been applied based on whether a covariate is included in the PS. It can be shown that PSM can also indirectly reduce the SMD of covariates not explicitly included in the PS model, due to underlying correlations or associations. 

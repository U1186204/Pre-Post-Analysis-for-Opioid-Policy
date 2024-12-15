# Pre-Post-Analysis-for-Opioid-Policy
A Data-Driven Framework Integrating Random Forest Imputation, Linear Regression, and Causal Inference to Assess Policy Impact

# Objective
This project is an exploration of opioid control policies through the blend of machine learning techniques, advanced regression methods, and causal inference strategies. By examining opioid shipment data, drug-related mortality rates, and population-level trends, we assess the effectiveness of state-level interventions. The primary focus lies on two key policy-implementing states—Florida and Washington—to evaluate changes in opioid consumption and overdose mortality post-policy implementation. This analysis employs Random Forest imputation to address missing data challenges, Pre-Post analysis to examine temporal effects, and Difference-in-Differences (DiD) to establish causal relationships by contrasting trends against control states.


# Project Schema

# Machine Learning for Missing Data
At the heart of this analysis lies the Random Forest Imputation pipeline, a cornerstone of machine learning applied to real-world data challenges. Missing overdose mortality data exhibited an exponential relationship with county population size, enabling the Random Forest algorithm to capture and model this trend with high fidelity. The model’s robustness against outliers and its capacity to identify complex patterns ensured accurate predictions even in noisy datasets.

Model training and evaluation metrics validated the predictive performance across counties, while data visualization confirmed that the predicted values adhered to the expected exponential pattern. This approach not only mitigated missing data biases but also enhanced the reliability of the broader causal analysis. 

# Pre-Post Charts 


# Pre-Post Results
- Florida: the analysis revealed a decisive, sustained reduction in both opioid shipments (measured as Morphine Milligram Equivalent, MME per capita) and overdose mortality following the 2010 policy implementation. The policy’s effects were immediate and linear, with no evidence of rebound effects. Florida’s multifaceted intervention—including strict prescription limits, real-time monitoring programs, and public health initiatives—proved to be a model of sustained policy impact.
- Washington: The 2012 policy, while initially effective, struggled to maintain its impact over time. Although MME per capita and overdose mortality declined in the immediate aftermath of policy enforcement, both metrics rebounded within a few years. This underscores the importance of addressing systemic factors, such as the rise of illicit opioids, that can erode the long-term efficacy of interventions.The Difference-in-Differences analysis validated these findings. Florida’s post-policy trends diverged significantly from its control states, highlighting the clear causal impact of its interventions. In contrast, Washington’s trends failed to exhibit meaningful divergence from its controls, pointing to limitations in the scope or execution of its policy.
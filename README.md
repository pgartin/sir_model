# sir_model
With so much uncertainty during a pandemic like the one currently going on all over the world with Covid-19, there is much information to be gained from data representing spread of the disease. This kind of information is useful for policymakers and medical workers to understand the effects of they're decisions and help efforts to mitigate the impact, but also to the individual who just wants to know what to expect. The spread of an infectious disease is highly dependent on various factors; however, a global pandemic produces enough data to show general trends. A  simple model for understanding this trend that is commonly used is the SIR-model. In this paper, I implement a trust region method to fit an SIR(D) Model to current data for the Covid-19 case counts. The reproduction number, R_0, can then be computed and compared to current value estimates such as the values ranging from 2.0 - 3.5.

## Contents
Running corona_model_trust_region.py will import covid data for the selected country and dates, then attempt a Trust Region Method to fit the SIR Model to the data. Results will be plotted.

## For more Details
See summary_paper.pdf which goes through the theory and details of this project, as well as a summary of the results.

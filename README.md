# Dependent-censoring-detection

Sample code:


import pandas as pd

from dependent_censoring _detection_function import get_final_p_value_for_dataset

df = pd.read_csv("your_data.csv")

p_value = get_final_p_value_for_dataset(

    dataset=df,
    
    quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
    
    B=100,  # Number of permutations
    
    seed=123,
    
    min_stratum_size=30,
    
    variance_threshold=0.001,
    
)

print("Final p-value:", p_value)



Interpretation (common rule):

p < 0.05: evidence against conditional independence (dependent censoring)
p >= 0.05: no strong evidence against the independence assumption

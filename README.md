# Sales-Potential-Estimation-Data-Quality-Visualization

This project processes economic indicators from two U.S. sources, enriching them with expert knowledge. 
It prioritizes features, analyzes clusters of U.S. counties, and identifies potential sales regions 
for specific products across the country.


## Project structur

Sales-Potential-Estimation-Data-Quality-Visualization/

├── notebooks_and_data/  
│   ├── data/  
│   │   ├── original_data/  
│   │   │   ├── pkl/  
│   │   │   ├── txt/  
│   │   │   └── xlsx/  
│   │   ├── processed_data/  
│   │   │   ├── batch_ids/  
│   │   │   ├── json/  
│   │   │   ├── melted_data/  
│   │   │   └── pkl/  
│   ├── my_functions/  
│   │   ├── functions_data_linking.py  
│   │   ├── functions_feature_selection.py  
│   │   └── functions_tsne_analysis.py  
│   ├── data_linking_ipynb/  
│   ├── feature_selection_ipynb/  
│   └── analysis_ipynb/  
└── README.md

## Notebooks (It makes sense to run the notebooks in the mentioned order, as they build on each other)
----------
### 1. `data_linking_ipynb`
External data sources included labor market features for industries (NAICS Code) and occupations (SOC Code) across various U.S. regional dimensions. Key datasets were the 2022 Occupational Employment and Wage Statistics (OEWS) from the Bureau of Labor Statistics and the 2021 County Business Patterns (CBP) from the Census Bureau, focusing on employee counts and establishment numbers across industries and occupations.

The primary challenge was that occupation statistics were only available and comprehensive at the state level. However, the occupation dataset provides the national distribution of occupations by NAICS industry. To address this, we used the national-level distribution of occupations for different industry sectors and the business pattern data to allocate occupation employment numbers from the state level down to individual counties within each state.

Due to the data size, the two datasets need to be manually downloaded and added to the original_data folder. For more details, please refer to the **README** in the **original_data** folder.

The **occupation_master.xlsx** file must be manually adjusted to align with your specific business case and domain field. It is essential to prioritize occupations based on relevance to your industry and the data you are working with. This ensures that the dataset accurately reflects the most important occupations for your analysis, facilitating a more targeted and meaningful application of labor market features.

The diagram shows the data linking process: 

![grafik](https://github.com/user-attachments/assets/cc4fb0f0-720d-433e-b8c0-39c1a7af8848)



### 2. `feature_selection_ipynb/`
This notebook builds the master dataframe as the foundation for further analysis. The selection of features was guided by domain experts to ensure relevance and applicability. Industries and occupations were chosen based on their significance in terms of employment and tool consumption, with a particular emphasis on sectors closely linked to manufacturing.


### 3. `Analysis_ECML_ipynb/`
This notebook analyzes the U.S. market by applying different dimensionality reduction techniques and different clustering methods for clustering across 3,233 counties. The resulting clusters are optimized based on various metrics and analyzed to identify key features. The notebook then visualizes the cluster characteristics and regional distributions. A Random Forest model is used to identify the most important features driving cluster assignments. The results highlight significant patterns and insights related to the U.S. market and its regional variations.




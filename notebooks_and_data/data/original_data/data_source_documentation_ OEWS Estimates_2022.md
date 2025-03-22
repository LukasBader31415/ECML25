# Data Source: Occupational Employment and Wage Statistics (OEWS) Tables

## 1. General Information about the Data Source

**Title:** Occupational Employment and Wage Statistics (OEWS) Tables  
**Publisher/Organization:** U.S. Bureau of Labor Statistics  
**Authors:** No specific authors listed  

## 2. URL and Access Date

**URL of the data source:**  
- [https://www.bls.gov/oes/tables.htm](https://www.bls.gov/oes/tables.htm)  
- [https://www.bls.gov/oes/special-requests/oesm22all.zip](https://www.bls.gov/oes/special-requests/oesm22all.zip)  
- `all_data_M_2022.xlsx`  

**Access Date:** October 18, 2024  

## 3. Licensing Terms

**Public Domain?**  

## 4. Data Description

The OEWS program produces employment and wage estimates for over 800 occupations. These estimates include:  
- The number of jobs in specific occupations  
- The wages paid to those jobs  

These estimates are available for:  
- The U.S. as a whole  
- Individual states  
- Metropolitan statistical areas (MSAs), metropolitan divisions, and nonmetropolitan areas  
- National occupational estimates for specific industries  

## 5. Data Format and Structure

### May 2022 OEWS Estimates  

**Source:**  
- Occupational Employment and Wage Statistics (OEWS) Survey  
- Bureau of Labor Statistics, Department of Labor  
- Website: [www.bls.gov/oes](https://www.bls.gov/oes)  
- Email: `oewsinfo@bls.gov`  

**Note:** Not all fields are available for every type of estimate.  

| **Field**         | **Description**                                                                                                               |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `area`            | U.S. (99), state FIPS code, Metropolitan Statistical Area (MSA) or New England City and Town Area (NECTA) code, or OEWS-specific nonmetropolitan area code |
| `area_title`      | Area name                                                                                                                     |
| `area_type`       | Area type: 1= U.S.; 2= State; 3= U.S. Territory; 4= Metropolitan Statistical Area (MSA) or New England City and Town Area (NECTA); 6= Nonmetropolitan Area |
| `prim_state`      | The primary state for the given area. "US" is used for the national estimates.                                                |
| `naics`           | North American Industry Classification System (NAICS) code for the given industry                                             |
| `naics_title`     | North American Industry Classification System (NAICS) title for the given industry                                            |
| `i_group`         | Industry level indicating cross-industry or NAICS sector, 3-digit, 4-digit, 5-digit, or 6-digit industry.                     |
| `own_code`        | Ownership type. Examples: 1= Federal Government; 5= Private; 123= Federal, State, and Local Government; etc.                  |
| `occ_code`        | The 6-digit Standard Occupational Classification (SOC) code or OEWS-specific code for the occupation                          |
| `occ_title`       | SOC title or OEWS-specific title for the occupation                                                                           |
| `o_group`         | SOC occupation level, e.g., major, minor, broad, or detailed levels.                                                          |
| `tot_emp`         | Estimated total employment rounded to the nearest 10 (excludes self-employed).                                                |
| `emp_prse`        | Percent relative standard error (PRSE) for the employment estimate.                                                          |
| `jobs_1000`       | Number of jobs (employment) in the given occupation per 1,000 jobs in the given area.                                         |
| `loc_quotient`    | Location quotient representing the occupation’s employment share in a given area relative to the U.S.                         |
| `pct_total`       | Percent of industry employment in the given occupation.                                                                      |
| `pct_rpt`         | Percent of establishments reporting the given occupation.                                                                    |
| `h_mean`          | Mean hourly wage                                                                                                             |
| `a_mean`          | Mean annual wage                                                                                                             |
| `mean_prse`       | Percent relative standard error (PRSE) for the mean wage estimate.                                                           |
| `h_pct10`         | Hourly 10th percentile wage                                                                                                  |
| `h_pct25`         | Hourly 25th percentile wage                                                                                                  |
| `h_median`        | Hourly median wage (or 50th percentile)                                                                                      |
| `h_pct75`         | Hourly 75th percentile wage                                                                                                  |
| `h_pct90`         | Hourly 90th percentile wage                                                                                                  |
| `a_pct10`         | Annual 10th percentile wage                                                                                                  |
| `a_pct25`         | Annual 25th percentile wage                                                                                                  |
| `a_median`        | Annual median wage (or 50th percentile)                                                                                      |
| `a_pct75`         | Annual 75th percentile wage                                                                                                  |
| `a_pct90`         | Annual 90th percentile wage                                                                                                  |
| `annual`          | "TRUE" if only annual wages are released for specific occupations.                                                           |
| `hourly`          | "TRUE" if only hourly wages are released for specific occupations.                                                           |

**Notes:**  
- `*` Wage estimate not available  
- `**` Employment estimate not available  
- `#` Wage equal to or greater than $115.00/hour or $239,200/year  
- `~` Less than 0.5% of establishments reporting  

## 6. Additional Information

**Occupational Definitions:** [https://www.bls.gov/oes/current/oes_stru.htm](https://www.bls.gov/oes/current/oes_stru.htm)  

### Reasons for Area and Statewide Employment Differences:
- Rounding  
- Inclusion of data not released separately for confidentiality/quality reasons  
- Cross-state metropolitan areas  
- Missing metropolitan or nonmetropolitan area indicators for some establishments  

## 7. Technical Documentation

- [https://www.bls.gov/oes/oes_doc.htm](https://www.bls.gov/oes/oes_doc.htm)

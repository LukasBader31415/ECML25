# Data Source: County Business Patterns: 2022

## 1. General Information about the Data Source

**Title:** County Business Patterns: 2022  
**Publisher/Organization:** United States Census Bureau  
**Authors:** No specific authors listed  

## 2. URL and Access Date

**URL of the Data Source:**
- [Dataset Overview](https://www.census.gov/data/datasets/2022/econ/cbp/2022-cbp.html)
- [Dataset Download](https://www2.census.gov/programs-surveys/cbp/datasets/2022/cbp22co.zip)
- `cbp22co.txt`

**Access Date:** October 18, 2024  

## 3. Licensing Terms

The Census Bureau has reviewed this data product to ensure appropriate access, use, and disclosure avoidance protection of the confidential source data (Project No. 7503949, Disclosure Review Board (DRB) approval number: CBDRB-FY23-0121).

## 4. Data Description

County Business Patterns (CBP) datasets are subsets of CBP data in comma-separated value (CSV) format for downloading. This series includes the number of establishments, employment during the week of March 12, first quarter payroll, and annual payroll. Corresponding record layouts and reference files are available.

## 5. Data Format and Structure

**Format:** CSV Files  

**Structure:**
```text
FIPSTATE        C       FIPS State Code
FIPSCTY         C       FIPS County Code
NAICS           C       Industry Code - 6-digit NAICS code.
EMPFLAG         C       Data Suppression Flag
                This denotes employment size class for data withheld to avoid disclosure (confidentiality)
                or withheld because data do not meet publication standards.
                A       0-19
                B       20-99
                C       100-249
                E       250-499
                F       500-999
                G       1,000-2,499
                H       2,500-4,999
                I       5,000-9,999
                J       10,000-24,999
                K       25,000-49,999
                L       50,000-99,999
                M       100,000 or More

EMP_NF          C       Total Mid-March Employees Noise Flag 
                (See all Noise Flag definitions at the end of this record layout)
EMP             N       Total Mid-March Employees with Noise
QP1_NF          C       Total First Quarter Payroll Noise Flag
QP1             N       Total First Quarter Payroll ($1,000) with Noise
AP_NF           C       Total Annual Payroll Noise Flag
AP              N       Total Annual Payroll ($1,000) with Noise
EST             N       Total Number of Establishments
N1_4            N       Number of Establishments: 1-4 Employee Size Class
N5_9            N       Number of Establishments: 5-9 Employee Size Class
N10_19          N       Number of Establishments: 10-19 Employee Size Class
N20_49          N       Number of Establishments: 20-49 Employee Size Class
N50_99          N       Number of Establishments: 50-99 Employee Size Class
N100_249        N       Number of Establishments: 100-249 Employee Size Class
N250_499        N       Number of Establishments: 250-499 Employee Size Class
N500_999        N       Number of Establishments: 500-999 Employee Size Class
N1000           N       Number of Establishments: 1,000 or More Employee Size Class
N1000_1         N       Number of Establishments: Employment Size Class: 1,000-1,499 Employees
N1000_2         N       Number of Establishments: Employment Size Class: 1,500-2,499 Employees
N1000_3         N       Number of Establishments: Employment Size Class: 2,500-4,999 Employees
N1000_4         N       Number of Establishments: Employment Size Class: 5,000 or More Employees
CENSTATE        C       Census State Code
CENCTY          C       Census County Code

NOTE: Noise Flag definitions (fields ending in _NF) are:
    G       0 to < 2% noise (low noise)
    H       2 to < 5% noise (medium noise)
    D       Withheld to avoid disclosing data for individual companies; 
            data are included in higher level totals. Employment or payroll field set to zero.
    S       Withheld because estimate did not meet publication standards. 
            Employment or payroll field set to zero.
```

## 6. General Information

**Industries Excluded from County Business Patterns:**  
CBP covers most NAICS industries, excluding:
- Crop and Animal Production (NAICS 111,112)
- Rail Transportation (NAICS 482)
- Postal Service (NAICS 491)
- Pension, Health, Welfare, and Other Insurance Funds (NAICS 525110, 525125190)
- Trusts, Estates, and Agency Accounts (NAICS 20, 5525920)
- Offices of Notaries (NAICS 541120)
- Private Households (NAICS 814)
- Public Administration (NAICS 92)

## 7. Technical Documentation

**Links:**
- [Technical Documentation Overview](https://www.census.gov/programs-surveys/cbp/technical-documentation.html)
- [Methodology](https://www.census.gov/programs-surveys/cbp/technical-documentation/methodology.html)
- [Reference Files](https://www.census.gov/programs-surveys/cbp/technical-documentation/reference.html)

## 8. Citation of the Data Source

**Recommended citation:**  
United States Census Bureau. (2023). *County Business Patterns: 2022*. Retrieved from [https://www.census.gov/data/datasets/2022/econ/cbp/2022-cbp.html](https://www.census.gov/data/datasets/2022/econ/cbp/2022-cbp.html) on October 18, 2024.

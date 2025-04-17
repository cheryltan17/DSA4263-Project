# DSA4263-Project
## Introduction
In today’s challenging job market, job seekers are now confronted with an additional obstacle: fake job postings. Fake job postings are misleading or fraudulent advertisements that do not correspond to real job opportunities (Peiris, 2024).  A survey conducted by Resume Builder involving 649 hiring managers revealed that up to 40% of companies had published fake job listings in the past year, and 30% are currently advertising positions that aren’t real (Thapa, 2024). These deceptive postings are often driven by motives such as portraying company growth or to conduct market research to gauge the talent pool without the commitment to hire (Peiris, 2024). Several hiring managers even admitted to posting fake jobs to keep their own employees on their toes, saying they want employees to feel “replaceable” so that they will work harder (Kesslen, 2025). The consequences of these fake postings extend beyond wasted effort for job seekers who spend significant time customizing their applications for non-existent roles - it also distorts the understanding of the labor market for policymakers and economists, while eroding trust in job boards, recruitment platforms, and employers (Roy, 2025). There are also privacy risks as job seekers’ personal and professional information may be misused for identity theft or spam (Peiris, 2024).

In this project, we investigate fraudulent job postings and address the following key objectives. First, we aim to gain preliminary insights into the characteristics of fake job postings through exploratory data analysis (EDA) on our selected dataset. Second, we seek to develop a practical machine learning solution capable of effectively detecting fraudulent postings. 

The dataset is obtained from Kaggle, and contains features relevant to job postings such as job title, descriptions and benefits. This job fraud dataset consists of 17880 rows and 16 columns of data. To predict whether a job posting is fraudulent, it is important to clearly define our target variable, where a fraudulent label of 1 indicates fraud, and a label of 0 indicates non-fraud.  

Kaggle Link: https://www.kaggle.com/datasets/subhajournal/job-fraud-detection/data

## Installation

1. **Clone the repository**:
```
git clone git@github.com:cheryltan17/DSA4263-Project.git
cd DSA4263-PROJECT
```

2. **(Optional but recommended) Create and activate a virtual environment:**
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install packages**
```
pip install -r requirements.txt 
```

## Project Structure
```
├── README.md <- The top-level README for developers using this project.
├── data
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── setup.py                <- Makes the project pip installable
├── requirements.txt        <- Project dependencies
├── src/                    <- Source code for the project
│   ├── __init__.py         <- Makes src a Python module
│   ├── data/               <- Scripts for data downloading/generation
│   │   └── EDA.py
    │   └── preprocessing.py
│   ├── features/           <- Feature engineering scripts
│   │   └── feature_engineering.py
    │   └── feature_selection.py
│   └── models/            
│       ├── models.py
├── notebooks/                    
│   ├── 1-preprocessing.ipynb      
│   ├── 2-EDA.ipynb
│   ├── 3-feature-engineering-and-selection.ipynb
│   └── 4-models.ipynb
```
## References
Job Fraud Detection. (2021, September 20). Kaggle. https://www.kaggle.com/datasets/subhajournal/job-fraud-detection/data

Kesslen, B. (2025, January 13). 1 of every 5 job postings is actually fake, study says. Yahoo News. https://www.yahoo.com/news/one-five-job-postings-fake-170500904.html

Peiris, S. (2024, November 28). The dark reality of fake job postings: why they exist and their impact on job seekers. Medium. https://moonlighto2.medium.com/the-dark-reality-of-fake-job-postings-why-they-exist-and-their-impact-on-job-seekers

Roy, N. (2025, February 5). The hidden cost of fake job postings on the economy. https://www.linkedin.com/pulse/hidden-cost-fake-job-postings-economy-nick-roy-2d6ac

Thapa, A. (2024, August 22). Ghost jobs: What the rise in fake job listings says about the current job market. CNBC. https://www.cnbc.com/2024/08/22/ghost-jobs-why-fake-job-listings-are-on-the-rise.html
# DSA4263-Project
# 🚀 Getting Started

## 📦 Installation

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
│   │   └── eda.py
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

## Usage
*Run from command line:*
```
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py
python src/models/predict_model.py
```

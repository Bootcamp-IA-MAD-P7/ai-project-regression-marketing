# AI-Powered Digital Marketing Revenue Forecasting

Machine Learning regression project focused on forecasting digital marketing campaign revenue before launch using planning-time campaign variables.

## Project Overview

This project aims to develop and evaluate machine learning models capable of forecasting campaign revenue based on campaign configuration, audience characteristics, channel strategy and planned spend.

The project follows a complete Data Science workflow, including data exploration, preprocessing, feature engineering, model training, optimization and deployment through a user-facing application.

## Business Problem

Marketing teams invest significant budgets across multiple digital channels. Estimating campaign revenue before launch can support:

- Budget allocation decisions
- Campaign optimization
- Launch/no-launch decisions
- Performance forecasting before media spend is committed
- Marketing ROI improvement
- Data-driven decision making

## Dataset

This project uses the **Digital Marketing Performance Dataset** available on Kaggle:

https://www.kaggle.com/datasets/alinaboulsi/digital-marketing-performance-dataset

The dataset is not included in this repository. Please download it from Kaggle and place it inside the `data/` folder.

Additional instructions can be found in `data/README.md`.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Optuna
- Matplotlib
- Seaborn
- Streamlit


## Project Roadmap

### Phase 1 – Setup & Planning
- Define business problem and project scope
- Document dataset source and usage instructions
- Configure repository structure

### Phase 2 – Data Understanding & EDA
- Load and inspect dataset
- Analyze missing values and duplicates
- Explore target variable distribution
- Create visualizations and insights

### Phase 3 – Preprocessing & Feature Engineering
- Select target variable
- Remove post-launch and identifier leakage features
- Encode categorical variables
- Build preprocessing pipeline

### Phase 4 – Modeling & Evaluation
- Train baseline regression model
- Train ensemble models
- Apply cross-validation
- Optimize with Optuna
- Evaluate using RMSE, MAE and R²
- Analyze feature importance and residuals

### Phase 5 – Productization
- Save trained model
- Build Streamlit application
- Implement feedback collection mechanism

### Phase 6 – Final Delivery
- Complete technical documentation
- Generate business insights
- Prepare final presentation

## Project Status

🚧 In Development

## Author

**Gabriela Granja**  
Marketing, Automation & AI

# PROJECT MASTER RECORD

## Project Information

| Field | Value |
|---------|---------|
| Project Name | AI-Powered Digital Marketing Revenue Forecasting |
| Project Type | Machine Learning Regression |
| Author | Gabriela Granja |
| Repository | ai-project-regression-marketing |
| Status | In Progress |
| Current Phase | Productization - Forecast Model MVP |

---

## Business Problem

Marketing teams invest significant budgets across multiple digital channels. Estimating campaign revenue before launching a campaign can improve launch decisions, planning, budget allocation, forecasting and overall marketing performance.

---

## Project Objective

Develop a machine learning solution capable of forecasting digital marketing campaign revenue before launch using only campaign planning variables.

The project aims to simulate a real-world business use case while following a complete machine learning lifecycle, from data exploration to deployment.

---

## Target Variable

### Current Target

```text
Revenue
```

### Status

Validated as the regression target.

---

## Dataset

### Selected Dataset

Digital Marketing Performance Dataset

### Source

Kaggle

### Status

Pending assessment and validation.

---

## Technology Stack

### Data Analysis

- Python
- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- Optuna

### Application

- Streamlit

### Engineering

- Docker

---

## Project Scope

### Included

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Regression Modeling
- Hyperparameter Optimization
- Model Evaluation
- Streamlit Application
- Dockerization
- Technical Documentation

### Optional (Advanced Scope)

- Database Integration
- Cloud Deployment
- Monitoring
- Automated Retraining

---

## Success Criteria

### Academic

- Functional regression model
- Proper evaluation metrics
- Complete documentation

### Portfolio

- Professional repository structure
- Business-oriented problem definition
- Deployable application
- Clear storytelling and insights

---

## Current Milestone

### Forecast Model Productization

Objectives:

- Enforce planning-time feature policy in `src/`
- Exclude post-launch performance metrics from training
- Persist the trained forecast pipeline
- Prepare the model for Streamlit integration

---

## Next Planned Deliverables

1. Streamlit MVP
2. Prediction input validation
3. Feedback or prediction logging
4. Dockerization
5. Deployment

## Working Agreements

### Documentation

- All documentation must be written in Markdown.
- All documentation must be maintained in VS Code.
- Documentation must be updated continuously throughout the project.

### Git Workflow

- New project phases require a dedicated feature branch.
- Commits should represent a complete logical unit of work.
- Commit messages must follow professional conventions.

### Project Management

- Significant decisions must be documented.
- Daily progress must be recorded.
- Major milestones must be registered in PROJECT_LOG.md.

### Development Workflow

Business Understanding → Dataset Assessment → EDA → Feature Engineering → Modeling → Application → Deployment.


### 2026-06-25

**Business Scope Update**

The project objective was redefined.

Instead of using post-launch performance metrics, the project will focus on forecasting campaign revenue before launch using only planning-time variables.

See JUN-25-business-problem-definition

#  Product Return Prediction System

##  Overview
This project builds an end-to-end Machine Learning system to predict the likelihood of a product being returned in an e-commerce setting.

The goal is to help businesses proactively identify high-risk orders and take preventive actions such as improving delivery, offering incentives, or verifying orders.

---

##  Problem Statement
Product returns lead to significant revenue loss in e-commerce.

This system predicts:
- **0 → Low Return Risk**
- **1 → High Return Risk**

using historical order and customer behavior data.

---

##  ML Approach

### Type of Problem
- Supervised Learning
- Binary Classification

---

## Features Used

- product_category  
- price  
- delivery_delay_days  
- customer_rating  
- previous_returns  
- discount_percent  
- payment_method  
- city  
- is_festive_season  
- order_month  
- order_day  
- customer_order_count_before  

---

##  Pipeline

1. Data Cleaning
2. Feature Engineering
3. Label Encoding for categorical variables
4. Train-Test Split
5. Model Training
6. Threshold Tuning (Recall Optimization)
7. Model Explainability (SHAP)
8. API Deployment using Flask

---

##  Models Used

### 1. Random Forest (Primary Model)
- Handles non-linearity well
- Robust to noise
- Balanced class handling

### 2. XGBoost (Comparison Model)
- Boosting-based approach
- Strong performance on structured data

---

##  Model Performance

| Model | Accuracy | ROC-AUC |
|------|--------|--------|
| Random Forest | ~0.92 | ~0.98 |
| XGBoost | ~0.91 | ~0.98 |

---

##  Threshold Optimization

- Default threshold (0.5) was reduced to **0.30**
- Improves **recall** for high-risk returns
- Helps catch more potential returns early

---

##  Explainability

Used SHAP (SHapley Additive Explanations) to understand feature importance.

### Key Insights:
- Customer Rating (strong negative impact)
- Previous Returns (strong positive impact)
- Delivery Delay (high influence)

---

##  API Deployment

Built using Flask.

### Run API:
```bash
python src/app.py
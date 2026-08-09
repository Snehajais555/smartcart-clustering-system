# SmartCart Clustering System

Full problem statement: see `ProblemStatment_SmartCart.pdf`

## Problem Statement
SmartCart is a growing e-commerce platform serving customers across multiple countries, with 2,240 customer records and 22 attributes covering demographics, purchase behavior, website activity, and campaign response.

Currently, SmartCart uses generic marketing and engagement strategies for all customers, without understanding distinct behavior patterns. This leads to inefficient marketing, missed opportunities to retain high-value customers, and delayed identification of churn-prone users.

This project builds an intelligent customer segmentation system using unsupervised machine learning to group customers into meaningful clusters based on purchasing behavior, engagement levels, and loyalty indicators — supporting data-driven, personalized marketing and retention decisions.

## Dataset
Each row represents a customer, with features across four groups:

**Demographics:** Year of Birth, Education, Marital Status, Income, Kidhome, Teenhome, Enrollment Date (`Dt_Customer`)

**Spending (Amount):** MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds

**Purchase Frequency:** NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth

**Feedback/Constants:** Recency (days since last purchase), Complain (1 = complained in last 2 years, 0 = no)

## Approach
1. **Preprocessing & feature engineering:** Created `Total_Spending` by combining individual product spend columns; scaled features with `StandardScaler`
2. **Dimensionality reduction:** Applied PCA (3 components) to project scaled features into a lower-dimensional space for clustering and visualization
3. **Choosing k:** Used the Elbow Method (WCSS, located via `KneeLocator`) and Silhouette Score across k=2–10 to identify the optimal number of clusters
4. **Clustering:** Compared **K-Means** (k=4) against **Agglomerative Clustering** (ward linkage, 4 clusters) on the PCA-reduced data, visualized in 3D
5. **Cluster characterization:** Analyzed income and spending patterns per cluster to profile distinct customer segments (e.g. high-income high-spenders vs. low-engagement customers) for targeted marketing

## Tech Stack
Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, kneed

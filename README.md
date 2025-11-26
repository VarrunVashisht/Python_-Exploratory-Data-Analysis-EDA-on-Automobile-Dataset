# 🚗 Exploratory Data Analysis (EDA) on Automobile Dataset

## 📌 Project Overview

This project performs **Exploratory Data Analysis (EDA)** on a real-world automobile dataset.
The goal is to understand the **factors that influence car pricing** using Python’s most important data analysis tools.

It includes:

* Cleaning & preparing messy data
* Converting incorrect data formats
* Handling missing values
* Understanding relationships between variables
* Visualizing insights using charts
* Computing statistical relationships (Pearson correlation)

This project is ideal for beginners wanting a practical, hands-on understanding of **data cleaning**, **EDA**, and **visualization**.

---

## 🧰 Tools & Libraries Used

* **Pandas** → data loading, cleaning, wrangling
* **NumPy** → handling numeric operations
* **Matplotlib** → basic charts
* **Seaborn** → advanced statistical visualizations
* **SciPy** → Pearson correlation & p-values

These are standard, industry-level Python tools used by every Data Analyst and Data Scientist.

---

## 🎯 What This Project Teaches

By going through this project, you will learn:

### ✔️ Data Cleaning

* Replacing missing values
* Converting incorrect data types
* Handling non-numeric placeholders like `"?"`
* Identifying numeric vs categorical variables

### ✔️ Exploratory Analysis

* Summary statistics (mean, median, min, max, std)
* Value counts for categorical variables
* Grouping & pivot tables
* Correlation matrix interpretation

### ✔️ Data Visualization

Multiple types of charts are included to help understand the data:

#### 📈 Regression Plots

Shows relationships between features and car price:

* Engine Size vs Price
* Highway MPG vs Price
* Peak RPM vs Price

#### 📊 Boxplots

Compare price across categories:

* Body Style
* Engine Location

#### 🗂 Heatmap

A pivot heatmap for:

* **Drive Wheels × Body Style**
* Colored by **average price**

This reveals which car types & configurations are more expensive.

#### 📉 Histograms & KDE

Visualize the distribution of car prices.

---

## 🔍 Key Insights You Can Discover

From the analysis, you can identify patterns like:

* **Bigger engines → Higher car prices**
* **Fuel-efficient cars (high MPG) → Usually cheaper**
* **Luxury body styles have higher median prices**
* **Drive-wheel types like RWD are often linked to higher prices**
* **Certain makes and body styles together create expensive combinations**

A correlation analysis also helps reveal which features have the strongest numeric relationship with the vehicle price.

---

## 📚 Learning Outcome

This project teaches you how real-world data behaves —
messy, incomplete, inconsistent — and how to clean it, visualize it, and extract insights.

If you are learning **Data Analysis**, this project builds strong confidence in:

✔ Python for Data Analysis
✔ Data Cleaning & Wrangling
✔ Visualization Skills
✔ Understanding relationships between variables
✔ Statistical reasoning (correlation + p-values)

---

## 🖼 Example Outputs

The script generates visuals such as:

* Regression lines showing feature → price relationships
* Boxplots comparing price across categories
* Heatmap visualizing combined group means
* Histogram & KDE charts for price distribution
* Correlation matrix of numeric variables

These charts help you think like a data analyst.

---

## 📁 Repository Contents

* **eda_cars.py** — Main analysis script
* **README.md** — Explanation & documentation
* *(Optional)* Plots folder if you decide to save charts

---

**Author:** 
*Varrun Vashisht*

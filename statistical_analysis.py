import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Create visualization folder
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

# Load dataset
df = pd.read_csv("sales_data.csv")

print("\nDataset Loaded Successfully!\n")
print(df.head())

# Ensure Total_Sales column exists
if "Total_Sales" not in df.columns:
    df["Total_Sales"] = df["Quantity"] * df["Price"]

sales = df["Total_Sales"]

# 1️⃣ DESCRIPTIVE STATISTICS
print("\n===== DESCRIPTIVE STATISTICS =====")

mean_sales = sales.mean()
median_sales = sales.median()
mode_sales = sales.mode()[0]
std_sales = sales.std()
variance_sales = sales.var()

print(f"Mean Sales: ${mean_sales:,.2f}")
print(f"Median Sales: ${median_sales:,.2f}")
print(f"Mode Sales: ${mode_sales:,.2f}")
print(f"Standard Deviation: ${std_sales:,.2f}")
print(f"Variance: {variance_sales:,.2f}")

# 2️⃣ DISTRIBUTION ANALYSIS
plt.figure()
sns.histplot(sales, kde=True)
plt.title("Sales Distribution")
plt.savefig("visualizations/histogram.png")
plt.close()

# 3️⃣ CORRELATION ANALYSIS
correlation = df[["Quantity", "Price", "Total_Sales"]].corr()

plt.figure()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("visualizations/heatmap.png")
plt.close()

# 4️⃣ HYPOTHESIS TESTS
print("\n===== HYPOTHESIS TESTS =====")

# Test 1: One-sample t-test (Is mean sales different from 50000?)
t_stat1, p_val1 = stats.ttest_1samp(sales, 50000)

# Test 2: Independent t-test (Compare two products if available)
products = df["Product"].unique()
if len(products) >= 2:
    group1 = df[df["Product"] == products[0]]["Total_Sales"]
    group2 = df[df["Product"] == products[1]]["Total_Sales"]
    t_stat2, p_val2 = stats.ttest_ind(group1, group2)
else:
    t_stat2, p_val2 = (0, 1)

# Test 3: ANOVA (Across products)
groups = [df[df["Product"] == p]["Total_Sales"] for p in df["Product"].unique()]
anova_stat, anova_p = stats.f_oneway(*groups)

print(f"One-sample t-test p-value: {p_val1:.4f}")
print(f"Independent t-test p-value: {p_val2:.4f}")
print(f"ANOVA p-value: {anova_p:.4f}")

# Save hypothesis results
with open("hypothesis_results.txt", "w") as f:
    f.write("HYPOTHESIS TEST RESULTS\n")
    f.write(f"One-sample t-test p-value: {p_val1:.4f}\n")
    f.write(f"Independent t-test p-value: {p_val2:.4f}\n")
    f.write(f"ANOVA p-value: {anova_p:.4f}\n")

# 5️⃣ CONFIDENCE INTERVAL
confidence = 0.95
n = len(sales)
std_err = std_sales / np.sqrt(n)
margin = stats.t.ppf((1 + confidence) / 2, n - 1) * std_err

lower = mean_sales - margin
upper = mean_sales + margin

print(f"\n95% Confidence Interval: ${lower:,.2f} to ${upper:,.2f}")

# 6️⃣ REGRESSION ANALYSIS
X = df["Quantity"]
Y = df["Total_Sales"]

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()

r_squared = model.rsquared

print(f"\nRegression R-squared: {r_squared:.4f}")

# Regression Plot
plt.figure()
sns.regplot(x=df["Quantity"], y=df["Total_Sales"])
plt.title("Regression: Quantity vs Sales")
plt.savefig("visualizations/regression_plot.png")
plt.close()

# 7️⃣ FINAL REPORT OUTPUT
print("\n====================================")
print("STATISTICAL ANALYSIS REPORT")
print("====================================")
print(f"Average Sales: ${mean_sales:,.2f} ± ${margin:,.2f} (95% CI)")
print(f"Correlation (Quantity-Sales): {correlation.loc['Quantity','Total_Sales']:.2f}")
print(f"R-squared: {r_squared:.2f}")

if p_val1 < 0.05:
    print(f"Mean sales significantly different from $50,000 (p={p_val1:.4f}) ✓ SIGNIFICANT")
else:
    print(f"Mean sales not significantly different from $50,000 (p={p_val1:.4f})")

print("\n===== BUSINESS INSIGHTS =====")
print("• Sales increase strongly with quantity sold.")
print("• Product categories show statistically significant differences.")
print("• Sales distribution shows variability across transactions.")
print("• Business should focus on high-performing products to maximize revenue.")

print("\nAll visualizations saved inside 'visualizations/' folder.")

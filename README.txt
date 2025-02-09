import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot

# Load the dataset
file_path = "C:\\Users\\sonia\\OneDrive\\Desktop\\life_expectancy_vs_school_enrollment_in_romania.csv"
df = pd.read_csv(file_path)

# Drop unnecessary column
df = df.drop(columns=['Unnamed: 0'])

# Define dependent and independent variables
y = df['Life Expectancy']
X = df[['School Enrollment']]

# Add constant term for regression
X = sm.add_constant(X)

# Perform linear regression
model = sm.OLS(y, X).fit()

# Now, get robust standard errors (HAC) by using the get_robustcov_results method
model_robust = model.get_robustcov_results(cov_type='HAC', maxlags=1)  # HAC for robust standard errors

# Print regression summary with robust standard errors
print(model_robust.summary())

# Plot the relationship between school enrollment and life expectancy
plt.figure(figsize=(8, 6))
plt.scatter(df['School Enrollment'], df['Life Expectancy'], label='Data')
plt.xlabel('School Enrollment')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy vs School Enrollment')

# Regression line
school_enrollment_range = np.linspace(df['School Enrollment'].min(), df['School Enrollment'].max(), 100)
X_pred = pd.DataFrame({'const': 1, 'School Enrollment': school_enrollment_range})
predicted_life_expectancy = model_robust.predict(X_pred)
plt.plot(school_enrollment_range, predicted_life_expectancy, color='red', label='Regression Line')

plt.legend()
plt.show()  # First plot

# Normal Probability Plot (Q-Q plot) for residuals
residuals = model_robust.resid

# Create Q-Q plot
plt.figure(figsize=(8, 6))
qqplot(residuals, line='45', ax=plt.gca())  # The '45' line is the reference line for a normal distribution
plt.title('Normal Probability Plot (Q-Q plot) of Residuals')
plt.show()  # Second plot

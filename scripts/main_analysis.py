#Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

#Load the data located in local folder and print head of data
data = pd.read_csv('U.S._Renewable_Energy_Consumption.csv')
print(data.head())

#Filter data to be used from 2010. Data in csv file date back to 1973 but are uneccessary for our analysis.
filtered_data = data[data['Year']>=2010]

#Analyzing Total Renewable Energy Trends
# Group data by 'Year' and sum up 'Total Renewable Energy'
yearly_trends = filtered_data.groupby('Year')['Total Renewable Energy'].sum()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(yearly_trends.index, yearly_trends.values, marker='o', linestyle='-', color='green')
plt.title('Total Renewable Energy Production Over Time (2010-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Renewable Energy (Quadrillion BTU)', fontsize=14)
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

#Calculate Renewable Energy Production through years and comparison with Total sum of production and Increase in percentage.
energy_2010 = yearly_trends.loc[2010]
energy_2023 = yearly_trends.loc[2023]

absolute_increase = energy_2023 - energy_2010

percentage_increase = (absolute_increase / energy_2010) * 100

print(f'Renewable Energy Production in 2010: {energy_2010:.2f} Quadrillion BTU')
print(f'Renewable Energy Production in 2023: {energy_2023:.2f} Quadrillion BTU')
print(f'Absolute Increase (2010-2023): {absolute_increase:.2f} Quadrillion BTU')
print(f'Percentage Increase (2010-2023): {percentage_increase:.2f} Quadrillion BTU')

#Build an Area Chart to show Renewable Energy Production share of each category.
energy_sources = [
    'Conventional Hydroelectric Power', 'Geothermal Energy', 'Solar Energy', 'Wind Energy', 'Biomass Energy'
]
source_trends = data.groupby('Year')[energy_sources].sum()

# Calculate Total Energy for each year
total_energy = filtered_data.groupby('Year')[energy_sources].sum().sum(axis=1)

# Calculate Contribution (%)
contribution = filtered_data.groupby('Year')[energy_sources].sum().div(total_energy, axis=0) * 100

# Plot as Stacked Area Chart
plt.figure(figsize=(16, 8))
plt.stackplot(contribution.index, contribution.values.T, labels=energy_sources, alpha=0.8)
plt.title('Contribution of Renewable Energy Sources Over Time (2010-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage Contribution (%)', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#Calculate Each Category of Renewable Energy Production and compare increase or decrease in total sum of BTU and Percentage Change from 2010 to 2023.
# List of renewable energy sources
renewable_sources = [
    'Conventional Hydroelectric Power', 
    'Geothermal Energy', 
    'Solar Energy', 
    'Wind Energy', 
    'Biomass Energy'
]

# Group data by 'Year' and sum energy for each source
source_trends = filtered_data.groupby('Year')[renewable_sources].sum()

# Initialize a dictionary to store results
trend_analysis = {}

print("Trends of Renewable Energy Sources (2010 to 2023):\n")

for source in renewable_sources:
    # Extract values for 2010 and 2023
    value_2010 = source_trends.loc[2010, source]
    value_2023 = source_trends.loc[2023, source]
    
    # Calculate absolute and percentage increase
    absolute_increase = value_2023 - value_2010
    percentage_increase = (absolute_increase / value_2010) * 100 if value_2010 != 0 else None

    # Save the results to the dictionary
    trend_analysis[source] = {
        '2010': value_2010,
        '2023': value_2023,
        'Absolute Increase': absolute_increase,
        'Percentage Increase': percentage_increase
    }

    # Print the results
    print(f"{source}:")
    print(f"  2010: {value_2010:.2f} Quadrillion BTU")
    print(f"  2023: {value_2023:.2f} Quadrillion BTU")
    print(f"  Absolute Increase: {absolute_increase:.2f} Quadrillion BTU")
    if percentage_increase is not None:
        print(f"  Percentage Increase: {percentage_increase:.2f}%")
    else:
        print(f"  Percentage Increase: Data unavailable (2010 value is zero).")
    print("-" * 40)

# Visualization of Trends for All Sources
plt.figure(figsize=(14, 8))

for source in renewable_sources:
    plt.plot(source_trends.index, source_trends[source], marker='o', linestyle='-', label=source)

plt.title('Trends of Renewable Energy Sources Over Time (2010-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Energy Production (Quadrillion BTU)', fontsize=14)
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title="Energy Sources", fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#Forecasting in Renewable Energy Generation from 2023 to 2033.
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Forecasting Total Renewable Energy
yearly_totals = filtered_data.groupby('Year')['Total Renewable Energy'].sum()

# ARIMA Model (use the most recent years for better prediction)
model = ARIMA(yearly_totals, order=(1, 1, 1))
fitted_model = model.fit()

# Forecast for 10 Years Ahead
forecast_years = np.arange(2023, 2033)
forecast_values = fitted_model.forecast(steps=10)

# Plot Historical Data and Forecast
plt.figure(figsize=(12, 6))
plt.plot(yearly_totals.index, yearly_totals.values, marker='o', label='Historical Data')
plt.plot(forecast_years, forecast_values, marker='o', linestyle='--', label='Forecast')
plt.title('Forecasting Total Renewable Energy (2023-2033)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Renewable Energy (Quadrillion BTU)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#Filter the csv file from 2010 to 2023 and save it as Updated_CSV
data = data[data['Year'] >= 2010]
filtered_data.to_csv('filtered_energy_data_2010_onwards.csv', index=False)
# Pakistans-regression-on-WDI-PENN-data(attached PDF contains all relevant charts & graphs)

The data was taken from World Bank's World Development Index(WDI file) and PENN World Table(pwt), it has been filtered for Pakistan and the missing values were imputed using Median
All the other attached datasets were exported and then cleaned in excel for their missing values and imputation.








The code is attached below:

import pandas as pd

import os
working_directory= os.getcwd()
print(working_directory)


# In[24]:


# Import the "PENN FILE" data from an Excel file

pk_data = '/Users/froggy/data/pwt1001.xlsx'
pk_data = pd.read_excel(pk_data, sheet_name='Data')
print(pk_data)
# Import the "WDI" data from an Excel file

WDI_file = '/Users/froggy/data/WDI.xlsx'
# Read the "WDI" data into a DataFrame
WDI_data = pd.read_excel(WDI_file, sheet_name='Data')
#Remove rows with missing values (NaN)
WDI_data = WDI_data.drop_duplicates()
WDI_data = WDI_data.fillna(0)
print(WDI_data)


# In[25]:


# Perform data cleaning operations as needed
pak_data = pk_data.drop_duplicates()

# Handle missing values (e.g., fill missing values with zeros)
pak_data = pk_data.fillna(0)

# Filter rows where 'Country' is equal to 'Pakistan'
pak_data = pak_data[pk_data['country'] == 'Pakistan']

# 'pak_data' now contains only the rows where 'Country' is 'Pakistan'

pak_data.head()
print(pak_data)


# Export the DataFrame to an Excel file
# pak_data.to_excel(pak_data, index=False)  # Set index to False to exclude the DataFrame index in the Excel file


# pak_data_cleaned = 'pak_data.xlsx'
# print(f"Data has been exported to {pak_data}")


# In[27]:


# Rename columns
column_renaming = pak_data.rename(columns={
    'rgdpe': 'Expenditure_Real_GDP',
    'pop': 'Population_Millions',
    'emp': 'Persons_Engaged_Millions',
    'avh': 'Avg_Annual_Hours_Worked',
    'hc': 'Human_Capital_Index',
    'rgdpna': 'Real_GDP_National_Prices',
    'rtfpna': 'TFP at constant national prices '
})
pak_data.head(10)


# In[28]:


# selecting only the renamed columns
PAK_data = [
    'Expenditure_Real_GDP',
    'Population_Millions',
    'Persons_Engaged_Millions',
    'Avg_Annual_Hours_Worked',
    'Human_Capital_Index',
    'Real_GDP_National_Prices',
]

# Create a new DataFrame with selected columns
PAK_data = pak_data[PAK_data]

# Display the resulting DataFrame
print(PAK_data)


# In[29]:


print(PAK_data.columns)


# In[30]:


merged = pd.concat ([WDI_data, PAK_data], axis=1)
merged.isnull().sum()
merged.fillna(0)
merged.dropna(inplace=True)

#Using data from merged file
merged_data = '/Users/froggy/data/Big_data.xlsx'
merged_data = pd.read_excel(merged_data, sheet_name='Biggy')
print(merged_data.columns)
print(merged_data.head(5))


# In[31]:


#FINDING CORRELATION

unemployment_column = 'Unemployment, total%'
gdp_column = 'Real_GDP_National_Prices'
selected_WDI_data = [unemployment_column, gdp_column]


# Calculate the correlation between unemployment and real GDP growth for Pakistan
correlation = merged_data[selected_WDI_data]
correlation.corr()


# In[42]:


import matplotlib.pyplot as plt

merged_data.plot()
plt.figure(figsize=(5, 4))
plt.show()

#Plotting the Histograms
plt.figure(figsize=(6, 4))
plt.hist(merged_data['Real_GDP_National_Prices'], bins=8, alpha=1)
plt.xlabel('GDP')
plt.ylabel('Frequency')
plt.title('GDP Distribution')
plt.show()


#YEARLY TRENDS:
import matplotlib.pyplot as plt
import numpy as np  # Import NumPy

# Assuming you have a DataFrame named merged_data
# You can read it from the Excel file or create it as needed

# Set the figure size for the first plot
plt.figure(figsize=(5, 4))

# Plot the data for 'Year' and 'Unemployment, male %'
plt.plot(merged_data['Year'], merged_data['Unemployment, male %'], label='Unemployment, male %')

# Add a trend line for 'Unemployment, male %'
z = np.polyfit(merged_data['Year'], merged_data['Unemployment, male %'], 1)
p = np.poly1d(z)
plt.plot(merged_data['Year'], p(merged_data['Year']), "r--")

# Repeat the process for other columns
columns_to_plot = [
    'Unemployment, total%',
    'Expenditure_Real_GDP',
    'Population (in millions)',
    'Number of persons engaged (in millions)',
    'Average annual hours worked by persons engaged',
    'Human capital index'
]

for column in columns_to_plot:
    plt.figure(figsize=(5, 4))
    plt.plot(merged_data['Year'], merged_data[column], label=column)
    z = np.polyfit(merged_data['Year'], merged_data[column], 1)
    p = np.poly1d(z)
    plt.plot(merged_data['Year'], p(merged_data['Year']), "r--")

# Show all the plots
plt.legend()
plt.show()


# In[43]:


variables = [
    'Unemployment, male %',
    'Unemployment, total%',
    'Expenditure_Real_GDP',
    'Population (in millions)',
    'Number of persons engaged (in millions)',
    'Average annual hours worked by persons engaged',
    'Human capital index',
    'Real_GDP_National_Prices',
]
for variable in variables:
    merged_data[variable] = pd.to_numeric(merged_data[variable], errors='coerce')

for i, variable in enumerate(variables):
    
    plt.subplot(3, 3, i + 1)
    plt.figure(figsize=(3, 3))

    plt.hist(merged_data[variable], bins=12, alpha=1)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title('Frequency of ' + variable)
    plt.grid()
    plt.tight_layout()

    
for i, var1 in enumerate(variables):
    for j, var2 in enumerate(variables):
        if i != j:
            x = merged_data[var1]
            y = merged_data[var2]
            
            plt.figure(figsize=(30, 30))
            plt.subplot(9, 9, i * len(variables) + j + 1)
            plt.scatter(x, y)
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title(f'Scatter Plot: {var1} vs {var2}')
            plt.grid()
            plt.tight_layout()
            
    plt.show()


# In[44]:


variables = [
    'Unemployment, male %',
    'Unemployment, total%',
    'Expenditure_Real_GDP',
    'Population (in millions)',
    'Number of persons engaged (in millions)',
    'Average annual hours worked by persons engaged',
    'Human capital index',
    'Real_GDP_National_Prices',
]

growthRates = {}
for var in variables:
    initial_value = merged_data[var].iloc[0]
    final_value = merged_data[var].iloc[-1]


    if initial_value != 0:
        growthRate = ((final_value - initial_value) / initial_value) * 100
        print(f"The growth rate for {var} is:     {growthRate}%")


# In[35]:


import matplotlib.pyplot as plt

# Assuming you have a DataFrame named merged_data
# You can read it from the Excel file or create it as needed

# Calculate the change in unemployment rate and RGDP growth rate
merged_data['Change in Total Unemployment'] = merged_data['Unemployment, total%'].diff()
merged_data['Change in Male Unemployment'] = merged_data['Unemployment, male %'].diff()
merged_data['RGDP Growth Rate'] = merged_data['Expenditure_Real_GDP'].pct_change() * 100

# Create scatterplots
plt.figure(figsize=(10, 6))

# Scatterplot for Total Unemployment vs. RGDP Growth Rate
plt.scatter(merged_data['Change in Total Unemployment'], merged_data['RGDP Growth Rate'], label='Total Unemployment')

# Scatterplot for Male Unemployment vs. RGDP Growth Rate
plt.scatter(merged_data['Change in Male Unemployment'], merged_data['RGDP Growth Rate'], label='Male Unemployment')

plt.xlabel('Change in Unemployment Rate')
plt.ylabel('RGDP Growth Rate')
plt.title('Relationship Between Unemployment and RGDP Growth Rate')
plt.legend()
plt.grid(True)

# Show the scatterplots
plt.show()

# Analyze the relationship
# Calculate the average effect of a 1 percentage point increase in the unemployment rate on output
average_effect_total_unemployment = merged_data['RGDP Growth Rate'].mean() / merged_data['Change in Total Unemployment'].mean()
average_effect_male_unemployment = merged_data['RGDP Growth Rate'].mean() / merged_data['Change in Male Unemployment'].mean()

print(f'Average effect of a 1 percentage point increase in total unemployment rate on output: {average_effect_total_unemployment}')
print(f'Average effect of a 1 percentage point increase in male unemployment rate on output: {average_effect_male_unemployment}')


# In[36]:


# Import necessary libraries
get_ipython().system('pip install numpy pandas statsmodels matplotlib')


# In[45]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named merged_data
# You can read it from the Excel file or create it as needed

# Create a DataFrame with the selected independent variables and the dependent variable
df = merged_data[['RGDP Growth Rate', 'Change in Total Unemployment', 'Change in Male Unemployment', 
                  'Population (in millions)', 'Number of persons engaged (in millions)', 
                  'Average annual hours worked by persons engaged', 'Human capital index']]
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Define your actual independent and dependent variables
X = sm.add_constant(df[['Change in Total Unemployment', 'Change in Male Unemployment', 'Population (in millions)', 
                      'Number of persons engaged (in millions)', 'Average annual hours worked by persons engaged', 
                      'Human capital index']])
Y = df['RGDP Growth Rate']

# Perform the multivariate regression
model = sm.OLS(Y, X).fit()

# Perform the multivariate regression
model = sm.OLS(Y, X).fit()

# Scatterplot of the actual data and the line of best fit
# plt.scatter(df['Change in Total Unemployment'], df['RGDP Growth Rate'], color='blue', label='Data points')
# plt.plot(df['Change in Total Unemployment'], model.predict(X), color='red', label='Line of best fit')
# plt.xlabel('Total Unemployment')
# plt.ylabel('RGDP')
# plt.title('RGDP vs Total Unemployment')
# plt.legend()
# plt.show()

# Save the regression summary to a LaTeX file
with open('output_summary.tex', 'w') as file:
    file.write(model.summary().as_latex())

# Print the summary to the console
print(model.summary())


# In[38]:


pip install seaborn


# In[49]:


plt.scatter(df['Human capital index'], df['RGDP Growth Rate'], color='blue', label='Data points')
plt.plot(df['Human capital index'], model.predict(X), color='red', label='Line of best fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y')
plt.legend()
plt.show()

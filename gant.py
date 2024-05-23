import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# Define the data for the Gantt chart
data = {
    'Task': [
        'Literature Review and Problem Definition', 
        'Data Collection and Preprocessing', 
        'Model Development and Initial Training', 
        'Model Optimization and Validation', 
        'User Evaluation and Feedback', 
        'Deployment and Integration', 
        'Documentation and Reporting'
    ],
    'Start': [
        '2024-06-01', '2024-07-01', '2024-08-01', 
        '2024-09-01', '2024-10-01', '2024-11-01', '2024-06-01'
    ],
    'End': [
        '2024-06-30', '2024-07-31', '2024-08-31', 
        '2024-09-30', '2024-10-31', '2024-11-30', '2024-11-30'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert Start and End dates to datetime
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Plotting the Gantt chart
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar for each task
for idx, row in df.iterrows():
    ax.barh(row['Task'], (row['End'] - row['Start']).days, left=row['Start'])

# Set the x-axis to date format
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Task')
plt.title('Project Gantt Chart')

plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure
plt.savefig('project_gantt_chart.png')

plt.show()

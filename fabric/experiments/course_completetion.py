from rich.console import Console
from rich.table import Table
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
	df = pd.read_csv('/home/maurits/EnergyEfficient_Scattered-Directive/Course_Completion_Prediction.csv')

	print(df.head())

	print("="*70)
	print(" DATA TYPES SUMMARY")
	print("="*70)

		# Create a detailed info dataframe
	data_info = pd.DataFrame({
			'Column': df.columns,
			'Data Type': df.dtypes.values,
			'Non-Null Count': df.notnull().sum().values,
			'Null Count': df.isnull().sum().values,
			'Null %': (df.isnull().sum().values / len(df) * 100).round(2),
			'Unique Values': df.nunique().values
	})

	print(data_info)

	print("\n" + "="*70)
	print(" DATA TYPES DISTRIBUTION")
	print("="*70)
	print(df.dtypes.value_counts())

	print("="*70)
	print("NUMERICAL FEATURES - DESCRIPTIVE STATISTICS")
	print("="*70)

	numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	print(f"\n Number of Numerical Features: {len(numerical_cols)}")

	stats_df = df[numerical_cols].describe().T
	# Create a rich table
	console = Console()
	table = Table(title="Numerical Features Statistics")

	table.add_column("Feature", style="cyan")
	for col in stats_df.columns:
			table.add_column(col)

	# Calculate thresholds for each column
	for col in stats_df.columns:
			q75 = stats_df[col].quantile(0.75)
			q25 = stats_df[col].quantile(0.25)
			
	for idx, row in stats_df.iterrows():
			row_values = [str(idx)]
			for col_name, val in row.items():
					q75 = stats_df[col_name].quantile(0.75)
					q25 = stats_df[col_name].quantile(0.25)
					
					# Highlight top 25% in green, bottom 25% in red
					if val >= q75:
							row_values.append(f"[bold green]{val:.2f}[/bold green]")
					elif val <= q25:
							row_values.append(f"[bold red]{val:.2f}[/bold red]")
					else:
							row_values.append(f"{val:.2f}")
			table.add_row(*row_values)

	console.print(table)

	print("="*70)
	print(" CATEGORICAL FEATURES - VALUE COUNTS")
	print("="*70)

	categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
	print(f"\nNumber of Categorical Features: {len(categorical_cols)}")

	for col in categorical_cols:
			if col not in ['Student_ID', 'Name', 'Enrollment_Date']:
					print(f"\n{'='*50}")
					print(f" {col}")
					print('='*50)
					print(df[col].value_counts())
	print("="*70)
	print(" CATEGORICAL FEATURES - VALUE COUNTS")
	print("="*70)

	categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
	print(f"\n📝 Number of Categorical Features: {len(categorical_cols)}")

	for col in categorical_cols:
			if col not in ['Student_ID', 'Name', 'Enrollment_Date']:
					print(f"\n{'='*50}")
					print(f" {col}")
					print('='*50)
					print(df[col].value_counts())

	print(df.head().T)

	# Prepare features
feature_cols = ['Age', 'Login_Frequency', 'Average_Session_Duration_Min', 'Video_Completion_Rate',
									'Discussion_Participation', 'Time_Spent_Hours', 'Days_Since_Last_Login',
									'Notifications_Checked', 'Peer_Interaction_Score', 'Assignments_Submitted',
									'Assignments_Missed', 'Quiz_Attempts', 'Quiz_Score_Avg', 'Project_Grade',
									'Progress_Percentage', 'Rewatch_Count', 'Payment_Amount', 'App_Usage_Percentage',
									'Reminder_Emails_Clicked', 'Support_Tickets_Raised', 'Satisfaction_Rating',
									'Course_Duration_Days', 'Instructor_Rating']

# Encode categorical features
le = LabelEncoder()
df_encoded = df.copy()
df_encoded['Gender_Encoded'] = le.fit_transform(df['Gender'])
df_encoded['Education_Encoded'] = le.fit_transform(df['Education_Level'])
df_encoded['Employment_Encoded'] = le.fit_transform(df['Employment_Status'])
df_encoded['Device_Encoded'] = le.fit_transform(df['Device_Type'])
df_encoded['Internet_Encoded'] = le.fit_transform(df['Internet_Connection_Quality'])
df_encoded['Level_Encoded'] = le.fit_transform(df['Course_Level'])
df_encoded['Category_Encoded'] = le.fit_transform(df['Category'])
df_encoded['Payment_Encoded'] = le.fit_transform(df['Payment_Mode'])

# Add encoded features
all_features = feature_cols + ['Gender_Encoded', 'Education_Encoded', 'Employment_Encoded',
                                'Device_Encoded', 'Internet_Encoded', 'Level_Encoded',
                                'Category_Encoded', 'Payment_Encoded']

# X = df_encoded[all_features]
# y = (df_encoded['Completed'] == 'Completed').astype(int).unsqueeze(1)

# print(y)
# print(df_encoded.head().T)
# df_encoded.to_csv('/home/maurits/EnergyEfficient_Scattered-Directive/python/hfl-train-model/datasets/courseData.csv')

alpha = 1.5
sizes = np.random.pareto(alpha, 20) + 1
# Normalize to sum to 100000
sizes = [int(s * 100000 / sum(sizes)) for s in sizes]

print(f"Min rows: {min(sizes)}, Max rows: {max(sizes)}, Mean: {np.mean(sizes):.0f}")
print(sizes)

offset = 0
for i, num_rows in enumerate(sizes):
	df_encoded.iloc[offset:(offset+num_rows)].to_csv(f'/home/maurits/EnergyEfficient_Scattered-Directive/python/hfl-train/datasets/client{i+1}.csv')
	offset += num_rows

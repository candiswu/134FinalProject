import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

internships_df = pd.read_csv('data/internships_clean.csv')

# 1. Check missing values in each column
missing_values = internships_df.isnull().sum()
print("Missing Values per Column:\n", missing_values)

# 2. Check unique values in key columns
unique_values = internships_df.nunique()
print("\nUnique Values per Column:\n", unique_values)

# 3. Distribution of Job Types
plt.figure(figsize=(8, 5))
sns.countplot(data=internships_df, x='Job.Type')
plt.title('Distribution of Job Types')
plt.xlabel('Job Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 4. Count of Skills
skills_columns = ['Skill.1', 'Skill.2', 'Skill.3', 'Skill.4', 'Skill.5', 'Skill.6']
all_skills = internships_df[skills_columns].values.flatten()

skills_counter = Counter(skill for skill in all_skills if pd.notnull(skill))
skills_df = pd.DataFrame(skills_counter.items(), columns=['Skill', 'Count']).sort_values(by='Count', ascending=False)

count_threshold = 5
filtered_skills_df = skills_df[skills_df['Count'] >= count_threshold]

plt.figure(figsize=(12, 8))
sns.barplot(data=filtered_skills_df, x='Count', y='Skill')
plt.title('Counts of Each Skill (Filtered)')
plt.xlabel('Count')
plt.ylabel('Skill')
plt.yticks(fontsize=7) 
plt.show()

# 5. Count of Perks
perks_columns = ['Perk.1', 'Perk.2', 'Perk.3', 'Perk.4', 'Perk.5', 'Perk.6']
all_perks = internships_df[perks_columns].values.flatten()

perks_counter = Counter(perk for perk in all_perks if pd.notnull(perk))
perks_df = pd.DataFrame(perks_counter.items(), columns=['Perk', 'Count']).sort_values(by='Count', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=perks_df, x='Count', y='Perk')
plt.title('Counts of Each Perk (Filtered)')
plt.xlabel('Count')
plt.ylabel('Perk')
plt.show()

# 6. Count of Locations
location_counts = internships_df['Location'].value_counts().reset_index()
location_counts.columns = ['Location', 'Count']

location_count_threshold = 2

filtered_location_counts = location_counts[location_counts['Count'] >= location_count_threshold]

plt.figure(figsize=(12, 8))
sns.barplot(data=filtered_location_counts, x='Count', y='Location')
plt.title('Counts of Each Location (Filtered)')
plt.xlabel('Count')
plt.ylabel('Location')
plt.show()

# 7. Correlation Matrix for Skills
skills_data = internships_df[skills_columns].values.tolist()
skills_data = [[skill for skill in job_skills if pd.notnull(skill)] for job_skills in skills_data]

mlb = MultiLabelBinarizer()
skills_matrix = mlb.fit_transform(skills_data)
skills_df_binary = pd.DataFrame(skills_matrix, columns=mlb.classes_)

skill_counts = skills_df_binary.sum().sort_values(ascending=False)
top_skills = skill_counts.head(10).index.tolist()

top_skills_df_binary = skills_df_binary[top_skills]

top_skills_corr = top_skills_df_binary.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(top_skills_corr, cmap='coolwarm', center=0, annot=True, fmt=".2f")
plt.title("Correlation Plot for Top 10 Skills")
plt.show()


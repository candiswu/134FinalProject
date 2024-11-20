import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/members_clean.csv')
#(df.head())
# column names: 
#print(df.columns)

# Histogram of Grad Year
plt.figure(figsize=(8,6))
sns.histplot(df['grad_year'], bins=5, discrete = True, palette = 'mako') 
plt.xlabel('Graduation Year')
plt.ylabel('Number of Members')
plt.title('Distribution of Graduation Years')
plt.show()

#print("Unique values in 'major':", df['major'].unique())
#print(df['major'].value_counts())

# Distribution of Majors 
def categorize_major(major):
    if "statistics & data science," in major.lower() or ", statistics & data science" in major.lower():
        return "Double Major with Statistics & Data Science"
    elif "financial math & statistics," in major.lower() or ", financial math & statistics" in major.lower():
        return "Double Major with Financial Math & Statistics"
    elif "electrical engineering" in major.lower():
        return "Electrical Engineering"
    elif "undeclared" in major.lower():
        return "Undeclared"
    else:
        return major.strip()

# Apply the categorization
df['major_cleaned'] = df['major'].apply(categorize_major)
major_counts = df['major_cleaned'].value_counts().reset_index()
major_counts.columns = ['Major', 'Count']
major_counts['Major'] = major_counts['Major'].where(major_counts['Count'] >= 5, other='Other')
major_counts = major_counts.sort_values(by='Count', ascending=True)
grouped_major_counts = major_counts.groupby('Major', as_index=False).sum()
grouped_major_counts = grouped_major_counts.sort_values(by='Count', ascending=False)
plt.figure(figsize=(12, 10))
sns.barplot(hue='Major', y='Count', data=grouped_major_counts, palette="mako")
plt.xlabel('Major')
plt.ylabel('Count')
plt.title('Distribution of Majors (Grouped)')
plt.tight_layout() 
plt.show()
#print(df['major_cleaned'].value_counts())

# Correlation between Programming Languages
df_binary = df.replace({'Yes': 1, 'No': 0})
languages_df = df_binary[["Java","Python","R","SQL","C/C++","JavaScript/TypeScript"]]
correlation_matrix = languages_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Programming Languages")
plt.show()

# Correlation between DS Libraries 
libraries_df = df_binary[["NumPy","Pandas","SciPy","Scikit-learn","PyTorch","Tensorflow"]]
lib_correlation_matrix = libraries_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(lib_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Libraries")
plt.show()

# Correlation between Courses Taken 
courses_df = df_binary[["Math 4A","CS 9","Calculus/Math 3A/Math 3B/AP Calc",
"Math 6A","Math 8/PSTAT 8/CS 40","PSTAT 10",
"PSTAT 120A","PSTAT 120B","PSTAT 126","PSTAT 160A/160B",
"Math 117","PSTAT 131","Math 108A","CS 165A/165B","PSTAT 135"]]
course_correlation_matrix = courses_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(course_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Courses Taken")
plt.show()

# Intern Job Count Boxplot with Graduation Year 
plt.figure(figsize=(12, 6))
sns.boxplot(x='grad_year', y='intern_job_count', data=df, palette="mako")
plt.title('Intern Job Count by Graduation Year')
plt.xlabel('Graduation Year')
plt.ylabel('Intern Job Count')
plt.tight_layout()
plt.show()

# Relationship between Grad_Year and Language
df_melted = df.melt(id_vars='grad_year', 
                    value_vars=['Python', 'R', 'SQL', 'Java', 'C/C++', 'JavaScript/TypeScript'], 
                    var_name='Language', 
                    value_name='Proficient')

df_melted = df_melted[df_melted['Proficient'] == 'Yes']

plt.figure(figsize=(12, 6))
sns.countplot(data=df_melted, x='grad_year', hue='Language', palette='mako')
plt.title('Count of Language Proficiency by Graduation Year')
plt.xlabel('Graduation Year')
plt.ylabel('Number of Students Proficient in Language')
plt.legend(title='Programming Language')
plt.tight_layout()
plt.show()


# Returning vs New Members Bar Plot
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='returning_member', palette="viridis")
plt.title('Returning vs New Members')
plt.xlabel('Member Status')
plt.ylabel('Count')
plt.show()

# Gender Count Bar Plot
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='gender', palette="rocket")
plt.title('Gender Count')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Transfer Student and Courses Plot
transfer_courses_df = df.loc[:, ['transfer'] + [col for col in df.columns if col.startswith("Math") or col.startswith("PSTAT") or col.startswith("CS")]]
transfer_courses_counts = transfer_courses_df.melt(id_vars=['transfer'], var_name='Course', value_name='Taken').query("Taken == 'Yes'").groupby(['transfer', 'Course']).size().unstack()

transfer_courses_counts.plot(kind="bar", stacked=True, colormap="coolwarm", width=0.8, figsize=(14, 8))
plt.title('Transfer Students and Courses Taken')
plt.xlabel('Transfer Status')
plt.ylabel('Number of Courses Taken')
plt.legend(title='Course', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Major vs Minor by Year Plot
major_order = [
    'Statistics & Data Science', 
    'Computer Science/Computer Engineering', 
    'Other', 
    'Mathematics/Applied Mathematics', 
    'Financial Math & Statistics', 
    'Economics', 
    'Electrical Engineering', 
    'Mechanical Engineering'
]

custom_palette = sns.color_palette("tab20", len(major_order))

plt.figure(figsize=(10, 8))
sns.countplot(data=df, x='grad_year', hue='major', palette=custom_palette, dodge=False, hue_order=major_order)
plt.title('Major Distribution by Graduation Year')
plt.xlabel('Graduation Year')
plt.ylabel('Count')
plt.legend(title='Major', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

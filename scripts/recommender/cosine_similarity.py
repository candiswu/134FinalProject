import pandas as pd
import nltk 
import re 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')
# nltk.download('stopwords')

# reading in student data
student_df = pd.read_csv('data/Cleaned_Member_Registration_Dataset.csv')

# reading in job data
job_df = pd.read_csv('data/internships_clean.csv')

# categorizing student internship job count 
def categorize_internship_count(count):
    if count == 0: 
        return "No internship experience"
    elif count == 1: 
        return "At least one internship experience"
    elif count == 2:
        return "At least two internship experiences"
    else: 
        return "Advanced internship experience"

student_df['intern_experience_text'] = student_df['intern_job_count'].apply(categorize_internship_count)

# categorizing grad year 
def categorize_by_graduation_year(row, current_year=2024):
    years_left = row['grad_year'] - current_year
    
    if years_left == 4:
        return 'Freshman'
    elif years_left == 3:
        return 'Sophomore'
    elif years_left == 2:
        return 'Junior'
    elif years_left == 1:
        return 'Senior'
    elif years_left == 0:
        return 'Graduate'
    else:
        return 'None'

# applying categorization function
student_df['year'] = student_df.apply(categorize_by_graduation_year, axis=1)

# combining student_df columns 
features = ['year', 'major', 'minor', 'intern_experience_text', 'Career.Goal', 'internship_or_full_time', 'Prefer.Remote', 'Top.desired.location', 'Top.desired.state', 'Industry.Preferences', 'Data_Science_Technologies', 'Data.Science.Skills', 'Merged_Languages', 'Packages']

student_df['text'] = ''
for feature in features:
    student_df[feature] = student_df[feature].fillna('')  # Fill NA values
    student_df['text'] += student_df[feature] + ' '       # Concatenate into 'text'

student_text = student_df[['ID', 'text']].copy() 
print(student_text.head())

# combining job_df columns
job_features = ['Company', 'Job.Title', 'Location', 'Job.Type', 'Experience', 'Skill.1', 'Skill.2', 'Skill.3','Skill.4', 'Skill.5', 'Skill.6', 'ExperienceQualifications', 'ClassYearQualifications']

job_df['text'] = ''
for feature in job_features:
    job_df[feature] = job_df[feature].fillna('')  
    job_df['text'] += job_df[feature] + ' '
job_df['text'] = job_df['text'].str.strip()

job_text = job_df[['text']].copy()# new job text with combined text 
print(job_text.head())

# tokenizing and preprocessing function 
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)  # tokenize
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]  # Remove stopwords and stem
    return ' '.join(tokens)  # join tokens back into a string

# apply preprocessing
student_text['preprocessed'] = student_text['text'].apply(preprocess)
job_text['preprocessed'] = job_text['text'].apply(preprocess)

# combining all text 
all_text = pd.concat([student_text['preprocessed'], job_text['preprocessed']], ignore_index=True)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)

student_tfidf = tfidf_matrix[:len(student_text)]  
job_tfidf = tfidf_matrix[len(student_text):]   

# looking at features 
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print(tfidf_df.head())

#### Cosine Similarity stuff: 

# Calculate Cosine Similarity
cosine_sim_matrix = cosine_similarity(student_tfidf, job_tfidf)

# Extract student IDs as a list or array
student_ids = student_df['ID'].astype(str).values  # Ensure student IDs are strings

# Extract job titles as a list or array
job_titles = job_df['Job.Title'].values  # Replace 'Job.Title' with the actual column name

# Convert cosine similarity matrix to a DataFrame for easier manipulation
cosine_sim_df = pd.DataFrame(
    cosine_sim_matrix,
    index=student_ids,  # Use student IDs as the index
    columns=job_titles  # Use job titles as the columns
)

# Define the give_recommendation function
def give_recommendation(student_id, top_n=10):
    if student_id not in cosine_sim_df.index:
        return f"No recommendations available for student ID: {student_id}"
    
    # Get similarity scores for the student and sort them
    scores = cosine_sim_df.loc[student_id]
    similar_jobs = scores.sort_values(ascending=False).iloc[:top_n]
    
    # Return a DataFrame of recommendations
    return pd.DataFrame({
        'Top Recommended Jobs': similar_jobs.index,
        'Similarity Score': similar_jobs.values
    })

# Example Usage
# Replace 'student_1' with an actual student ID from your dataset
recommendations = give_recommendation('421') # score 0.391746 
print(recommendations)


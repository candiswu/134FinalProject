# Job Recommendation System for Data Science UCSB
## Introduction
Let’s be honest—job hunting sucks. It’s like online dating but worse, because at least bad dates come with free food (well, except for Ryan, who loses money). That’s why we decided to band together and build this job recommendation system, turning our shared suffering into a creative outlet 👌.

We began this project by looking for data. We were able to get access to the **Data Science UCSB member registration dataset**, compiled from Google Form surveys during club sign-ups. This dataset gave us valuable insights into member skillsets, interests, majors, graduation years, and more—perfect for tailoring recommendations to our members’ unique needs.

Next, we found a **job dataset** on Kaggle, which included predictors such as company, job title, experience requirements, locations, skills, and perks. By combining **machine learning techniques** with **domain knowledge**, we aim to make job hunting a little less painful for everyone.

## Methodology
### Data Cleaning & Exploratory Data Analysis (EDA) 📊
We started by cleaning the datasets:
- Handling missing values
- Removing duplicates and irrelevant columns
- Factoring categorical variables

Next, we explored key trends and relationships through EDA to inform the recommendation system and uncovered several useful trends:
- Most members are Data Science majors.
- The average internship count increases with school year progression.
- Surprisingly, CS skills are preferred in Data Science job postings over actual Data Science skills.
We also resolved anomalies, such as skewed graduation year data that initially made trends appear counterintuitive.

## Recommendation System 🔍
### Preprocessing and Vectorization
1. Data Preprocessing:
     - Combined relevant columns into a single `text` column
     - Converted text to lowercase, removed punctuation, and tokenized using `nltk`
     - Applied the **Porter Stemmer** and removed stop words
2. TF-IDF Vectorization:
     - Used `sklearn` to compute Term Frequency-Inverse Document Frequency (TF-IDF) scores
     - Split TF-IDF matrix into student and job datasets for further analysis

## Cosine Similarity & Recommendation ML Model
Our system implements a **content-based filtering** approach to match students with relevant internships.

1. Cosine Similarity: Quantifies how closely a student’s profile matches a job posting based on TF-IDF vectors
2. Heuristic Scoring: Factors in skills, experience, and location preferences with different weights
3. Machine Learning (Random Forest Regressor): Labeled heuristic scores were used as a target variable to train a **Random Forest model**
Features like `student_id` and `internship_id` were **one-hot encoded**
The final output ranks internship recommendations for each student, displaying the **top opportunities** based on predicted relevance scores.

## Results & Next Steps 🎯

Our job recommendation system successfully generated personalized internship opportunities for Data Science UCSB club members.

While the model captured general trends, some challenges remain:
- Data limitations: The lack of detailed predictors required us to create synthetic data to align with project goals.
### Next Steps:
1. Enhancing Data Collection: Update next year’s member registration form to include more relevant survey questions.
2. Integration: Incorporate the recommendation system into the Data Science UCSB Discord bot, which already features a weekly LinkedIn scraper for personalized job alerts.
3. Improving Accuracy: Explore advanced machine learning models for more robust recommendations.
By evolving this system, we hope to make job hunting less painful!

## Resources 📚
### Datasets:
- Internship Listings: [Kaggle Dataset - Internships](https://www.kaggle.com/datasets/sulphatet/internships-list)
- Member Data: Club sign-up Google Form responses
#### Tools & Libraries:
- Python: Web Scraping, Data Cleaning, EDA, Recommendation Engine
- nltk: Text Preprocessing
- sklearn: TF-IDF Vectorization, Machine Learning Models
- Random Forest Regressor: Scoring Predictions

## Slides 🤓

Link to Presentation: https://www.canva.com/design/DAGYCIRK8kM/VXOS9eD7dNIKjVYKU-PdEA/view?utm_content=DAGYCIRK8kM&utm_campaign=designshare&utm_medium=link&utm_source=editor

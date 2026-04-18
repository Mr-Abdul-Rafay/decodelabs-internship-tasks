# Project 3: AI Recommendation Engine

A console-based recommendation system that suggests career paths based on a user's technical skills. The project uses content-based filtering with TF-IDF vectorization and cosine similarity to rank job roles against a user's selected skills.

## Overview

This project simulates a simple recommendation engine for tech career guidance.

The program:

- stores a small dataset of job roles and their associated skills
- asks the user to choose their top 3 skills
- converts both job roles and the user profile into TF-IDF vectors
- compares the user vector with each role vector using cosine similarity
- returns the top matching career recommendations

## Features

- interactive onboarding survey
- case-insensitive skill matching
- TF-IDF weighting for skill importance
- cosine similarity scoring
- top-N recommendation ranking
- explanation of why the top result was selected
- clear console output for learning and demonstration purposes

## Tech Stack

- Python 3
- Standard Library only
  - `math`

No external packages are required.

## Project Structure

```text
Ai Project 3/
|-- recommendation_engine.py
|-- README.md
```

## How It Works

### 1. Data Model

The system defines several job roles such as:

- Cloud Architect
- Data Scientist
- DevOps Engineer
- Backend Developer
- Frontend Developer
- Machine Learning Engineer
- Security Analyst
- Mobile Developer

Each role contains a list of relevant technical skills.

### 2. Onboarding Survey

The user selects 3 skills from the available skills database. These skills form the initial user profile.

### 3. TF-IDF Vectorization

The recommendation engine calculates:

- Term Frequency (TF): whether a skill appears in a role
- Inverse Document Frequency (IDF): how unique a skill is across all roles
- TF-IDF: the weighted importance of each skill

Rare skills receive higher importance than common skills.

### 4. Cosine Similarity

The engine compares the user profile vector with every job-role vector using cosine similarity:

```text
similarity = (A · B) / (|A| × |B|)
```

The higher the score, the closer the user's skills are to that job role.

### 5. Ranking

All roles are sorted by similarity score, and the top recommendations are displayed with match confidence percentages.

## How To Run

Open a terminal in the project folder:

```powershell
cd "C:\Users\HOME\Desktop\Ai Project 3"
```

Run the program:

```powershell
$env:PYTHONIOENCODING="utf-8"
python recommendation_engine.py
```

### Why `PYTHONIOENCODING` is needed on Windows

This script prints emoji and special characters. In some Windows PowerShell setups, running the script without UTF-8 output enabled can cause a `UnicodeEncodeError`.

If your terminal already supports UTF-8 correctly, you can also try:

```powershell
python recommendation_engine.py
```

## Example Run

```text
📊 DATABASE STATISTICS:
  - Total Job Roles: 8
  - Total Unique Skills: 37

Please select your top 3 skills:
  Skill #1: Python
  Skill #2: SQL
  Skill #3: Linux
```

The program then:

- calculates TF-IDF scores
- builds vectors
- computes similarity with every job role
- shows the top recommended career paths

## Learning Objectives

This project demonstrates core recommendation system concepts:

- content-based filtering
- feature representation with vectors
- weighting with TF-IDF
- similarity measurement with cosine similarity
- ranking and recommendation explanation

It is a good beginner-friendly project for understanding how recommendation engines work without using external machine learning libraries.

## Functions Included

Key functions in [`recommendation_engine.py`](/c:/Users/HOME/Desktop/Ai%20Project%203/recommendation_engine.py:1):

- `get_all_unique_skills()`
- `onboarding_survey()`
- `calculate_tf()`
- `calculate_idf()`
- `calculate_tfidf_vector()`
- `display_vector_sample()`
- `cosine_similarity()`
- `calculate_all_similarities()`
- `get_top_n_recommendations()`
- `display_recommendations()`
- `explain_recommendation_logic()`

## Strengths

- simple and easy to understand
- no third-party dependencies
- demonstrates real recommendation-system logic
- useful for portfolio or academic submission

## Limitations

- uses a small hardcoded dataset
- only accepts 3 user skills
- does not learn from user feedback
- does not persist users or recommendations
- is fully content-based and does not use collaborative filtering

## Possible Improvements

- load job roles from a JSON or CSV file
- allow weighted user preferences
- support more than 3 skills
- add a graphical interface
- export recommendations to a file
- remove emoji for full terminal compatibility
- add unit tests for similarity and TF-IDF calculations

## Author

DecodeLabs AI Engineer

## License

This project is intended for learning, demonstration, and academic practice unless your course or organization requires a separate license.

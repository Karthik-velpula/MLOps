# Google Colab Notebooks

## Week 1  [Skill_Gap_Prediction]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LV5fbj5rikuBG_4C1UhvNNaNEocOLMPP)

Or click here:
https://colab.research.google.com/drive/1LV5fbj5rikuBG_4C1UhvNNaNEocOLMPP


## Week 2 [Data_Preprocessing]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KKHWIQq9NXkja-FCZ7ttCi6mOZu5MCLL)

Or click here:
https://colab.research.google.com/drive/1KKHWIQq9NXkja-FCZ7ttCi6mOZu5MCLL


## Week 3 [Feast]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15KXxvectMH5phe4sJiXw5ceXV_l4gjNU?usp=sharing)

Or click here:
[https://colab.research.google.com/drive/15KXxvectMH5phe4sJiXw5ceXV_l4gjNU](https://colab.research.google.com/drive/15KXxvectMH5phe4sJiXw5ceXV_l4gjNU?usp=sharing)







# Curriculum-Industry Skill Feature Store Using Feast (Skill Gap Dataset)

## Student Details

| Field | Details |
|-------|---------|
| Name | V Venkata Karthik|
| Register Number | 231FA04509 |
| Section | 15 |
| Repository Name | 231FA04509-MLOps-Feast-SkillGap |

---

## 1. Problem Statement

The curriculum-industry skill-gap problem focuses on identifying whether specific skills are aligned with industry requirements or if there exists a gap between curriculum coverage and industry demand.

This project uses a skill gap dataset containing information about various technical and professional skills, their curriculum coverage, industry demand, job frequency, expert importance ratings, and a gap status target indicating whether a skill gap exists.

**The objective of this assignment** is to convert the processed skill gap dataset into a simple Feast-based feature store.

**Feast is used to:**
- Define reusable features
- Retrieve historical features for machine-learning training
- Materialize features into an online store
- Retrieve online features for inference
- Use the same feature definitions for prediction

---

## 2. Dataset

The dataset used in this project is `skill_gap_dataset_realistic.csv`, which contains 100 records of skill assessments.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Number of records | 100 |
| Number of original columns | 6 |
| Entity/Identifier | Skill_Name |
| Feast Entity | skill |
| Feast Join Key | skill_name |
| Target | Gap_Status |
| Target Encoding | Aligned/Gap |
| Missing Values | 0 in all original columns |

### Original Dataset Columns

| Column | Description |
|--------|-------------|
| Skill_Name | Name of the skill |
| Curriculum_Coverage | How well the skill is covered in curriculum (1-5 scale) |
| Industry_Demand | Industry demand level for the skill (1-5 scale) |
| Job_Frequency | Frequency of job postings requiring this skill (1-10 scale) |
| Expert_Importance | Importance rating by industry experts (1-5 scale) |
| Gap_Status | Whether there is a gap between curriculum and industry (Aligned/Gap) |

The dataset contains 25 unique skills with multiple assessments each, totaling 100 records.

---

## 3. Project Workflow
Original Skill Gap Dataset
|
v
Data Preprocessing
|
v
Feature Engineering
|
v
Parquet Feature Dataset
|
v
Feast Data Source
|
v
Feast FeatureView
|
+------+------+
| |
v v
Historical Features Materialization
| |
v v
Model Training SQLite Online Store
|
v
Online Feature Retrieval
|
v
Prediction

text

---

## 4. Feature Engineering

The following feature engineering steps were performed before creating the Feast feature dataset:

### 4.1 Skill Name Encoding

```python
df["skill_encoded"] = df["Skill_Name"].astype("category").cat.codes
Example:

Python -> 21

Java -> 14

C++ -> 2

SQL -> 23

4.2 Industry Gap Calculation
python
df["industry_gap"] = df["Industry_Demand"] - df["Curriculum_Coverage"]
Example:

For Generative AI: Industry Demand (5) - Curriculum Coverage (1) = 4 (Large Gap)

For Python: Industry Demand (5) - Curriculum Coverage (5) = 0 (Perfect Alignment)

4.3 Job Importance Calculation
python
df["job_importance"] = df["Job_Frequency"] * df["Expert_Importance"]
Example:

Python: 10 * 5 = 50

C++: 7 * 4 = 28

4.4 Feature Renaming
Curriculum_Coverage -> curriculum_coverage

Industry_Demand -> industry_demand

Job_Frequency -> job_frequency

Expert_Importance -> expert_importance

4.5 Timestamp Creation
python
base_time = pd.Timestamp("2026-01-01", tz="UTC")
df["event_timestamp"] = base_time + pd.to_timedelta(df.index, unit="s")
df["created_timestamp"] = df["event_timestamp"] + pd.Timedelta(seconds=1)
5. Difference Between Original Dataset and Feature Dataset
The original dataset contains 6 raw columns.

The Feast feature dataset contains a transformed and enhanced set of features:

skill_name (join key)

event_timestamp

created_timestamp

skill_encoded

curriculum_coverage

industry_demand

job_frequency

expert_importance

industry_gap (engineered)

job_importance (engineered)

The target gap_status is maintained separately as the label.

6. Feast Entity
The Feast entity is skill with the join key skill_name:

python
skill = Entity(
    name="skill",
    join_keys=["skill_name"],
    description="Skill in the skill gap dataset"
)
7. Feast Data Source
python
skill_gap_source = FileSource(
    name="skill_gap_source",
    path="data/skill_gap_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)
8. Feast FeatureView
Name: skill_gap_features

Features Stored
Feature	Data Type	Meaning
skill_encoded	Int64	Encoded skill identifier
curriculum_coverage	Int64	Curriculum coverage rating (1-5)
industry_demand	Int64	Industry demand rating (1-5)
job_frequency	Int64	Job posting frequency (1-10)
expert_importance	Int64	Expert importance rating (1-5)
industry_gap	Int64	Gap between industry demand and curriculum coverage
job_importance	Int64	Product of job frequency and expert importance
Configuration
Property	Value
Name	skill_gap_features
Entity	skill
Online	True
TTL	50000 days
9. Feature Service
python
skill_gap_service = FeatureService(
    name="skill_gap_service",
    features=[skill_gap_feature_view]
)
10. Feast Configuration
Feast version: 0.64.0

Offline Store: File-based (Parquet)

Online Store: SQLite

Provider: Local

feature_store.yaml
yaml
project: skill_gap_project
registry: data/registry.db
provider: local
offline_store:
  type: file
online_store:
  type: sqlite
  path: data/online_store.db
11. Feast Registration
bash
feast apply
Output:

text
Created project skill_gap_project
Created entity skill
Created feature view skill_gap_features
Created feature service skill_gap_service
Created sqlite table skill_gap_project_skill_gap_features
12. Historical Feature Retrieval
python
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=feature_service
).to_df()
First Five Records
skill_name	skill_encoded	curriculum_coverage	industry_demand	job_frequency	expert_importance	industry_gap	job_importance	gap_status
Python	21	5	5	10	5	0	50	Aligned
Java	14	5	4	9	5	-1	45	Aligned
C++	2	5	3	7	4	-2	28	Aligned
SQL	23	4	5	9	5	1	45	Aligned
DBMS	7	5	4	8	5	-1	40	Aligned
13. Machine-Learning Model
Features Used
python
feature_columns = [
    "skill_encoded",
    "curriculum_coverage",
    "industry_demand",
    "job_frequency",
    "expert_importance",
    "industry_gap",
    "job_importance"
]
Model Configuration
python
DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
Data Split
Training records: 80

Testing records: 20

Model Accuracy
text
Accuracy: 80.0 %
14. Materialization
bash
feast materialize \
    2026-01-01T00:00:00 \
    2026-01-02T00:00:00
Output:

text
Materializing 1 feature views
from 2026-01-01 00:00:00+00:00
to 2026-01-02 00:00:00+00:00
into the sqlite online store.
15. Online Feature Retrieval
python
online_features = store.get_online_features(
    features=feature_service,
    entity_rows=[{"skill_name": "Python"}]
).to_dict()
Results for Python
skill_name	curriculum_coverage	job_importance	skill_encoded	industry_demand	industry_gap	expert_importance	job_frequency
Python	5	50	21	4	-1	5	10
Final Prediction
text
Predicted gap status: Aligned
16. Additional Online Predictions
skill_name	predicted_gap_status
Python	Aligned
Java	Aligned
C++	Aligned
SQL	Aligned
17. Required Analysis Questions
17.1 What is the entity in your Feast implementation?
The Feast entity is skill with the join key skill_name.

17.2 List the features stored in your FeatureView.
skill_encoded

curriculum_coverage

industry_demand

job_frequency

expert_importance

industry_gap

job_importance

17.3 Explain how one feature was calculated.
The industry_gap feature is calculated as:

text
industry_gap = Industry_Demand - Curriculum_Coverage
Positive values indicate a gap exists, negative values indicate alignment, and zero indicates perfect alignment.

17.4 What is the difference between your original dataset and the feature dataset?
The original dataset has 6 columns, while the feature dataset contains 10 columns including engineered features (industry_gap, job_importance) and timestamp fields.

17.5 What is the purpose of the offline store?
The offline store keeps historical feature data and is used to retrieve features for model training.

17.6 What is the purpose of the online store?
The online store contains materialized features and is used for fast feature retrieval during prediction.

17.7 What is the purpose of feast apply?
feast apply registers the Feast definitions (entity, feature view, feature service) with the feature store.

17.8 What does materialization do?
Materialization transfers feature values from the offline store to the online store for a specified time range.

17.9 What is the advantage of retrieving features through Feast?
Feast provides centralized and reusable feature definitions, ensuring consistency between training and prediction.

17.10 State two limitations of your current dataset.
Limited skill coverage: Only 25 unique skills

Subjective ratings: Based on assessments rather than objective job data

17.11 State two ways your feature store could be improved.
Real-time data sources: Incorporate job postings and industry surveys

Temporal features: Track how skills evolve over time

18. Lab2 vs Lab3 Comparison
Aspect	Lab2 (Preprocessing)	Lab3 (Feast)
Dataset	skill_gap_dataset_realistic.csv	skill_gap_dataset_realistic.csv
Features	Original 6 columns	7 Feast features + engineered
Preprocessing	SimpleImputer, StandardScaler, OneHotEncoder	Preprocessed dataset
Model	Logistic Regression	Decision Tree
Training records	80	80
Testing records	20	20
Baseline accuracy	76.47%	-
Final accuracy	80.0%	80.0%
Feature store	No	Yes
19. Training and Serving Consistency
text
                 Feast FeatureView
                       |
             +---------+---------+
             |                   |
             v                   v
 get_historical_features   get_online_features
             |                   |
             v                   v
       Training Data         Prediction Data
             |                   |
             +---------+---------+
                       |
                       v
                    ML Model
20. Results Summary
Component	Result
Dataset size	100 records
Original columns	6
Unique skills	25
Feast version	0.64.0
Entity	skill
Join key	skill_name
FeatureView	skill_gap_features
FeatureService	skill_gap_service
Offline store	Parquet
Online store	SQLite
ML model	Decision Tree Classifier
Model accuracy	80.0%
Test skill	Python
Final prediction	Aligned
21. Technologies Used
Python

Google Colab

Pandas

NumPy

Scikit-learn

Apache Parquet

Feast 0.64.0

SQLite

GitHub

22. Conclusion
This project demonstrates a complete basic Feast feature-store workflow for the curriculum-industry skill-gap problem.

The original 6-column skill gap dataset was preprocessed and transformed into an enhanced feature dataset containing encoded skill identifiers, curriculum coverage, industry demand, job frequency, expert importance, and engineered features (industry_gap and job_importance).

The engineered data was stored in Parquet and registered with Feast through a FileSource and FeatureView. The skill entity and skill_name join key were created, and the skill_gap_features FeatureView was successfully registered using feast apply.

Historical features were retrieved using get_historical_features() and used to train a Decision Tree classifier achieving 80.0% accuracy.

The features were materialized into a SQLite online store and retrieved using get_online_features(). For Python, the retrieved online features produced a final gap status prediction of Aligned.

Thus, the implementation demonstrates how Feast can provide a consistent feature pipeline between historical model training and online model inference.

Key Feast Commands
Apply Feast Definitions
bash
feast apply
Materialize Features
bash
feast materialize \
    2026-01-01T00:00:00 \
    2026-01-02T00:00:00
Verify Feast Version
bash
feast --version
Expected version:

text
Feast 0.64.0
Final Project Outcome
#	Task	Status
1	Dataset preprocessing	✅
2	Feature engineering	✅
3	Feast entity creation	✅
4	Feast FileSource creation	✅
5	FeatureView creation	✅
6	FeatureService creation	✅
7	feast apply	✅
8	Historical feature retrieval	✅
9	Machine-learning model training	✅
10	Model accuracy (80.0%)	✅
11	Feature materialization	✅
12	SQLite online store	✅
13	Online feature retrieval	✅
14	Online gap status prediction	✅
15	Training-serving feature consistency	✅
Final Result: Feast-based skill gap feature store successfully implemented.

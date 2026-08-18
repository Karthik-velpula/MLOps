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







# 231FA04509-MLOps-Feast-SkillGap

## Student Details

| Field | Details |
| :--- | :--- |
| **Name** | V Venkata Karthik |
| **Register Number** | 231FA04509 |
| **Section** | 15 |
| **Repository Name** | 231FA04509-MLOps-Feast-SkillGap |

---

## 1. Problem Statement

The curriculum-industry skill-gap problem focuses on identifying whether students possess the academic, technical, and industry-oriented skills required for employment.
This project uses a curriculum-industry skill alignment dataset containing student academic information, technical skills, professional indicators, internship experience, certifications, projects, placement information, and an industry-readiness target.
The objective of this assignment is to convert the processed curriculum-industry dataset into a simple Feast-based feature store.
Feast is used to:
* Define reusable features
* Retrieve historical features for machine-learning training
* Materialize features into an online store
* Retrieve online features for inference
* Use the same feature definitions for prediction

The complete workflow is demonstrated in the Lab3 Feast notebook.

---

## 2. Dataset

The dataset used in this project was created in the previous curriculum-industry skill-gap activity and then preprocessed in Lab2.

### Dataset Statistics

| Property | Value |
| :--- | :--- |
| **Number of records** | 100 |
| **Number of original columns** | 25 |
| **Entity/Identifier** | Student_ID |
| **Feast Entity** | student |
| **Feast Join Key** | student_id |
| **Target** | Industry_Readiness |
| **Target Encoding** | Not Ready = 0, Ready = 1 |
| **Missing Values** | 0 in all original columns |

### Original Dataset Columns

| Column | Description |
| :--- | :--- |
| **Student_ID** | Unique student identifier |
| **Gender** | Student gender |
| **Semester** | Current semester |
| **CGPA** | Cumulative Grade Point Average |
| **Programming** | Programming skill score |
| **DSA** | Data Structures and Algorithms skill score |
| **DBMS** | Database Management Systems skill score |
| **OS** | Operating Systems skill score |
| **CN** | Computer Networks skill score |
| **OOP** | Object-Oriented Programming skill score |
| **Python** | Python skill score |
| **Java** | Java skill score |
| **SQL** | SQL skill score |
| **Web_Development** | Web development skill score |
| **Cloud** | Cloud computing skill score |
| **AI_ML** | Artificial Intelligence and Machine Learning skill score |
| **Communication** | Communication skill score |
| **Aptitude** | Aptitude skill score |
| **Teamwork** | Teamwork skill score |
| **Problem_Solving** | Problem-solving skill score |
| **Internship** | Internship experience |
| **Certifications** | Number of certifications |
| **Projects_Completed** | Number of completed projects |
| **Placement_Status** | Placement status |
| **Industry_Readiness** | Industry-readiness target |

The Lab3 notebook shows that the original dataset contains 100 rows and 25 columns, with all columns containing 100 non-null values.

---

## 3. How the Dataset Entries Were Created

Each row represents a student profile.
The dataset combines:
* Academic information
* Technical skill scores
* Soft skills
* Internship experience
* Certifications
* Projects
* Placement status
* Industry-readiness status

The dataset was created to represent the relationship between curriculum-level skills and industry-readiness indicators.

---

## 4. Project Workflow

The project follows the following end-to-end workflow:

```text
Original Curriculum-Industry Dataset
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
          |             |
          v             v
 Historical Features  Materialization
          |             |
          v             v
   Model Training   SQLite Online Store
                        |
                        v
                Online Feature Retrieval
                        |
                        v
                    Prediction
5. Feature Engineering
The Lab3 notebook performs feature engineering on the preprocessed dataset before creating the Feast feature dataset.

5.1 Student ID
The original Student_ID values such as S001, S002, etc. are converted to integer student_id values.
Example:
S001 -> 1
S002 -> 2
S003 -> 3
5.2 Gender Encoding
The Gender column is encoded as:
Male -> 0
Female -> 1
The resulting feature is: gender_encoded

5.3 Skill Count
skill_count counts how many of the following six technical skills have a score greater than 5:
Programming
DSA
DBMS
OS
CN
OOP
The calculation is:

Plaintext
skill_count =
    (Programming > 5) +
    (DSA > 5) +
    (DBMS > 5) +
    (OS > 5) +
    (CN > 5) +
    (OOP > 5)
This produces an integer representing the number of selected technical skills in which the student scored above 5.

5.4 Placement Encoding
The original Placement_Status is converted into:
Placed -> 1
Not Placed -> 0
The resulting Feast feature is: is_placed

5.5 CGPA and Programming
The notebook converts:
CGPA -> cgpa
Programming -> programming
Both features are stored as Float32.

5.6 Industry Readiness Target
The target is encoded as:
Not Ready -> 0
Ready -> 1
and stored separately as: industry_readiness

This target is used for model training and is not included as a FeatureView feature.

6. Difference Between Original Dataset and Feature Dataset
The original dataset contains 25 raw columns, including academic, technical, soft-skill, internship, certification, project, placement, and target information.
The Feast feature dataset is a smaller, model-ready dataset containing:
student_id
event_timestamp
created_timestamp
semester
gender_encoded
cgpa
programming
skill_count
is_placed
The target: industry_readiness is kept separately as the label for historical feature retrieval and model training.
Therefore, the feature dataset is a transformed and reduced representation of the original dataset designed specifically for reusable feature-store and machine-learning operations.

7. Feast Entity
The Feast entity is: student with the join key: student_id.

The entity is defined as:

Python
student = Entity(
    name="student",
    join_keys=["student_id"],
    description="CSE student"
)
The entity uniquely identifies each student and allows Feast to associate feature values with the correct student.

8. Feast Data Source
The Feast data source is a FileSource:

Python
curriculum_source = FileSource(
    name="curriculum_source",
    path="data/curriculum_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)
The engineered feature data is stored in: data/curriculum_features.parquet.
The event_timestamp is used for point-in-time historical feature retrieval.

9. Feast FeatureView
The FeatureView is: curriculum_features. It is associated with the student entity.

Features Stored in the FeatureView
Feature	Data Type	Meaning
semester	Int64	Student's semester
gender_encoded	Int64	Encoded gender
cgpa	Float32	Student CGPA
programming	Float32	Programming skill score
skill_count	Int64	Number of selected technical skills scoring above 5
is_placed	Int64	Placement status encoded as 1/0
FeatureView Configuration
Property	Value
Name	curriculum_features
Entity	student
Online	True
TTL	50000 days
10. Feature Service
The notebook creates a FeatureService named: industry_readiness_service.
It contains: curriculum_features.
The FeatureService is used for both historical and online feature retrieval.

11. Feast Configuration
The project uses:
Feast version: 0.64.0
The local Feast configuration uses:
Offline Store: File-based offline store
Offline Data: Parquet
Online Store: SQLite
Provider: Local
The configuration follows the local-development approach required for this assignment.

12. Feast Registration Using feast apply
The following command is executed in Lab3:

Bash
feast apply
The command completed successfully.
The output confirms that Feast created/registered:
Project: curriculum_industry_skill_alignment
Entity: student
FeatureView: curriculum_features
FeatureService: industry_readiness_service
SQLite online table: curriculum_industry_skill_alignment_curriculum_features
The entity listing confirms:
student CSE student
The FeatureView listing confirms:
curriculum_features {'student'} FeatureView Yes
Therefore, the assignment requirement for Feast registration using feast apply is demonstrated.

13. Historical Feature Retrieval
Historical features are retrieved using:

Python
store.get_historical_features(
    entity_df=entity_df,
    features=feature_service
).to_df()
The entity dataframe contains:
student_id
event_timestamp
industry_readiness
First Five Label Records
student_id	event_timestamp	industry_readiness
1	2025-01-01 00:00:01+00:00	0
2	2025-01-01 00:00:02+00:00	1
3	2025-01-01 00:00:03+00:00	0
4	2025-01-01 00:00:04+00:00	0
5	2025-01-01 00:00:05+00:00	0
The resulting historical training data contains:
student_id, event_timestamp, industry_readiness, semester, gender_encoded, cgpa, programming, skill_count, is_placed.

First Five Historical Feature Records
student_id	semester	gender_encoded	cgpa	programming	skill_count	is_placed	industry_readiness
1	7	1	8.43	3.0	3	0	0
2	7	1	8.88	5.0	4	1	1
3	7	0	8.68	3.0	3	0	0
4	8	0	7.76	7.0	4	0	0
5	8	0	8.84	9.0	5	0	0
14. Purpose of the Offline Store
The offline store contains historical feature data.
In this project, the feature data is stored in Parquet format and is used by Feast for historical feature retrieval.
The offline store is useful for:
Creating training datasets
Historical feature retrieval
Point-in-time correct feature retrieval
Reusing feature definitions for model training
15. Machine-Learning Model
The historical Feast features are used as input to a Decision Tree Classification model.
The model-training features are:

Python
feature_columns = [
    "semester",
    "gender_encoded",
    "cgpa",
    "programming"
]
The target is: industry_readiness.

The data is divided into training and testing sets using:

Python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
Therefore:
Training records: 80
Testing records: 20
The final model used for the main evaluation is:

Python
DecisionTreeClassifier(
    max_depth=1,
    min_samples_split=15,
    min_samples_leaf=8,
    random_state=42
)
16. Model Accuracy
The actual Lab3 output is:
Accuracy: 95.0 %

Therefore:
Model Accuracy = 95.0%

The model was trained using features retrieved through Feast's historical feature retrieval process.
For comparison, the earlier preprocessing notebook reported:
Baseline accuracy: 0.90
Accuracy after preprocessing: 0.95
The preprocessing notebook used 80 training records and 20 testing records.

17. Materialization into the Online Store
The following Feast command is executed in Lab3:

Bash
feast materialize \
    2025-01-01T00:00:00 \
    2025-01-02T00:00:00
The command successfully reports:
Materializing 1 feature views from 2025-01-01 00:00:00+00:00 to 2025-01-02 00:00:00+00:00 into the sqlite online store.

The materialized FeatureView is: curriculum_features.
Therefore, the materialization requirement is demonstrated.

18. Purpose of the Online Store
The online store contains materialized feature values and allows features to be retrieved quickly for inference.
In this project, SQLite is used as the local online store.
The online store is useful for:
Fast feature retrieval
Serving features during prediction
Providing the latest materialized feature values
Connecting a trained model to online inference
19. Online Feature Retrieval
The notebook retrieves online features for: student_id = 25 using:

Python
online_features = store.get_online_features(
    features=feature_service,
    entity_rows=[
        {"student_id": 25}
    ]
).to_dict()
Actual Output
Python
{
    'student_id': [25],
    'is_placed': [0],
    'semester': [7],
    'programming': [9.0],
    'gender_encoded': [0],
    'skill_count': [3],
    'cgpa': [7.059999942779541]
}
Displayed as a dataframe
student_id	is_placed	semester	programming	gender_encoded	skill_count	cgpa
25	0	7	9.00	0	3	7.06
This demonstrates online feature retrieval using get_online_features().

20. Final Online Prediction
The final model is trained on the complete historical dataset:

Python
final_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
final_model.fit(X, y)
The online features for student 25 are then passed to the model.
The actual notebook output is:
Predicted survival: 0

In the context of this project, the prediction represents:
0 = Not Ready
1 = Ready
The notebook labels the printed output as "Predicted survival" due to reused code wording, but the actual target in this project is industry_readiness.

Student 25 Final Prediction
Property	Result
Student ID	25
Predicted Industry Readiness	Not Ready
Encoded Prediction	0
21. Additional Online Feature Output
The notebook also retrieves online features for students 10, 20, 30, and 40.

Online Features
student_id	is_placed	semester	programming	gender_encoded	skill_count	cgpa
10	0	7	10.0	1	4	6.73
20	0	7	7.0	0	5	8.50
30	0	8	4.0	0	3	6.96
40	0	7	10.0	0	3	9.32
Final Online Predictions
student_id	predicted_industry_readiness
10	0
20	0
30	0
40	0
Where:
0 = Not Ready
1 = Ready
22. Required Analysis Questions
22.1 What is the entity in your Feast implementation?
The Feast entity is student. Its join key is student_id, which uniquely identifies each student.
The entity definition is:

Python
student = Entity(
    name="student",
    join_keys=["student_id"],
    description="CSE student"
)
22.2 List the features stored in your FeatureView.
The curriculum_features FeatureView stores:
semester
gender_encoded
cgpa
programming
skill_count
is_placed
22.3 Explain how one feature was calculated.
The skill_count feature is calculated by counting how many of six selected technical skills have a score greater than 5.
The six skills are: Programming, DSA, DBMS, OS, CN, OOP.
For each skill, a value greater than 5 contributes 1; otherwise it contributes 0.
Calculation:

Plaintext
skill_count = (Programming > 5) + (DSA > 5) + (DBMS > 5) + (OS > 5) + (CN > 5) + (OOP > 5)
For example, if four of the six skills are greater than 5, the student's skill_count is 4.

22.4 What is the difference between your original dataset and the feature dataset?
The original dataset contains 25 raw columns.
The Feast feature dataset contains a smaller set of transformed, model-ready features: student_id, event_timestamp, created_timestamp, semester, gender_encoded, cgpa, programming, skill_count, is_placed.
The target industry_readiness is maintained separately as the label.
The feature dataset is therefore a transformed representation designed for feature-store usage rather than a direct copy of the original dataset.

22.5 What is the purpose of the offline store?
The offline store keeps historical feature data and is used to retrieve features for model training.
In this project, the historical feature data is stored in Parquet format.

22.6 What is the purpose of the online store?
The online store contains materialized features and is used for fast feature retrieval during prediction.
This project uses SQLite as the local online store.

22.7 What is the purpose of feast apply?
feast apply registers the Feast definitions with the feature store.
In this project, it successfully created/registered the student entity, curriculum_features FeatureView, industry_readiness_service FeatureService, and SQLite online table.

22.8 What does materialization do?
Materialization transfers feature values from the offline/historical source into the online store for a specified time range.
In this project, materialization was performed from 2025-01-01 00:00:00 to 2025-01-02 00:00:00 into the SQLite online store.

22.9 What is the advantage of retrieving features through Feast instead of manually calculating them separately during training and prediction?
Feast provides centralized and reusable feature definitions.
The same FeatureView and FeatureService can be used for:
Historical Retrieval -> Model Training
Online Retrieval -> Model Prediction
This reduces duplicated feature-engineering code and helps maintain consistency between training and inference.
It also provides a standard mechanism for managing historical and online feature data.

22.10 State two limitations of your current dataset.
Small Dataset: The dataset contains only 100 student records. A dataset of this size may not represent the complete diversity of students and may limit the generalization of the machine-learning model.
Limited Real-World Industry Evidence: The current dataset contains student-level curriculum and skill indicators but does not include a large collection of continuously updated real-world industry evidence such as job-posting skill requirements, employer assessments, or industry trend data.
22.11 State two ways your feature store could be improved when more curriculum and industry evidence becomes available.
Add Real Industry Evidence: The feature store can incorporate job descriptions, employer skill requirements, internship evaluations, placement outcomes, industry certifications, and current technology-demand information.
Add Continuous Feature Updates: The feature store can be updated as curriculum versions and industry requirements change. New evidence can be incorporated into the feature pipeline so that the features remain relevant to current industry expectations.
23. Training and Serving Consistency
One important advantage demonstrated by this project is that Feast provides the same feature definitions for training and serving.

Plaintext
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
This reduces the risk of using different feature calculations during model training and prediction.

24. Results Summary
Component	Result
Dataset size	100 records
Original columns	25
Feast version	0.64.0
Entity	student
Join key	student_id
FeatureView	curriculum_features
FeatureService	industry_readiness_service
Offline data	Parquet
Online store	SQLite
Historical retrieval	Successful
feast apply	Successful
Materialization	Successful
Online retrieval	Successful
ML model	Decision Tree Classifier
Model accuracy	95.0%
Final test student	Student 25
Final prediction	Not Ready (0)
25. Technologies Used
Python
Google Colab
Pandas
NumPy
Scikit-learn
Apache Parquet
Feast 0.64.0
SQLite
Jupyter Notebook
GitHub
26. Conclusion
This project demonstrates a complete basic Feast feature-store workflow for the curriculum-industry skill-gap problem.
The original 25-column student dataset was preprocessed and transformed into a smaller feature dataset containing reusable academic, demographic, technical, skill-count, and placement features.
The engineered data was stored in Parquet and registered with Feast through a FileSource and FeatureView.
The student entity and student_id join key were created, and the curriculum_features FeatureView was successfully registered using feast apply.
Historical features were retrieved using get_historical_features() and used to train a Decision Tree classifier.
The model achieved an accuracy of 95.0% on the 20-record test set.
The features were then materialized into a SQLite online store and retrieved using get_online_features().
For student 25, the retrieved online features produced a final industry-readiness prediction of: Not Ready (0).

Thus, the implementation demonstrates how Feast can provide a consistent feature pipeline between historical model training and online model inference and can serve as a foundation for a larger curriculum-industry skill alignment system.

Key Feast Commands
Apply Feast Definitions

Bash
feast apply
Materialize Features

Bash
feast materialize \
    2025-01-01T00:00:00 \
    2025-01-02T00:00:00
Verify Feast Version

Bash
feast --version
Expected version: Feast 0.64.0

Final Project Outcome
The project successfully demonstrates:
Dataset preprocessing
Feature engineering
Feast entity creation
Feast FileSource creation
FeatureView creation
FeatureService creation
feast apply
Historical feature retrieval
Machine-learning model training
95% model accuracy
Feature materialization
SQLite online store
Online feature retrieval
Online industry-readiness prediction
Training-serving feature consistency
Final Result: Feast-based curriculum-industry skill feature store successfully implemented.

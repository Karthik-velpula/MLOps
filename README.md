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
| **Name** | Srinivas Yasarapu |
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

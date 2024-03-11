# Fake News Detection with BERT

![Header Image](https://github.com/user/AquaPredictor/blob/main/HeaderImage.png)

## Table of Contents

1. [Introduction](#introduction)
   1. [Problem Statement](#problem-statement)
   2. [Data](#data)
   3. [Tasks](#tasks)
2. [Data Preparation: Acquisition, Analysis, Cleaning, and Preprocessing](#data-preparation)
   1. [Data Acquisition](#data-acquisition)
   2. [Data Analysis](#data-analysis)
   3. [Data Cleaning](#data-cleaning)
   4. [Data Splitting](#data-splitting)
   5. [Data Tokenization](#data-tokenization)
   6. [Converting Tokenized Data to Tensors](#converting-tokenized-data-to-tensors)
3. [Fine-Tuning BERT Models: Model Training, Evaluation & Predictions](#fine-tuning-bert-models)
   1. [BERT Model with Layers Freeze](#bert-model-with-layers-freeze)
   2. [BERT Model with All Layers](#bert-model-with-all-layers)
4. [Results](#results)
5. [Conclusions](#conclusions)
6. [Contributions](#contributions)

## 1. Introduction <a name="introduction"></a>

### 1.1 Problem Statement <a name="problem-statement"></a>

In the digital age, the proliferation of false information and fake news has become a significant societal challenge, impacting public opinion, social harmony, and even political landscapes. The project aims to develop sophisticated deep learning models, particularly based on BERT (Bidirectional Encoder Representations from Transformers), to accurately distinguish between real and fake news articles.

### 1.2 Data <a name="data"></a>

The project utilizes the WELFake dataset obtained from Kaggle, comprising 72,134 news articles labeled as real or fake. This dataset serves as a comprehensive corpus for training and evaluating machine learning models aimed at detecting misinformation in textual data.

### 1.3 Tasks <a name="tasks"></a>

The primary objective is to train a deep learning model to recognize linguistic patterns and contextual subtleties associated with false news articles. By leveraging advanced natural language processing techniques, particularly BERT, the project aims to develop a powerful tool for automating the identification process, thereby combating the spread of misinformation in digital ecosystems.

## 2. Data Preparation: Acquisition, Analysis, Cleaning, and Preprocessing <a name="data-preparation"></a>

### 2.1 Data Acquisition <a name="data-acquisition"></a>

The dataset was sourced from Kaggle, ensuring its relevance and suitability for the task of fake news detection. With 72,134 samples, the dataset provides a substantial foundation for analysis and model training.

### 2.2 Data Analysis <a name="data-analysis"></a>

Exploratory data analysis revealed a balanced distribution of true and fake news articles within the dataset, enabling robust model training without bias towards any class. The analysis also provided insights into the linguistic characteristics and patterns prevalent in fake news articles.

### 2.3 Data Cleaning <a name="data-cleaning"></a>

Data cleaning procedures involved removing irrelevant columns, handling missing values, and ensuring data consistency. Visualization techniques aided in identifying and addressing data anomalies effectively.

### 2.4 Data Splitting <a name="data-splitting"></a>

The dataset was split into training and validation sets to facilitate model training and evaluation, ensuring reliable performance metrics.

### 2.5 Data Tokenization <a name="data-tokenization"></a>

Text sequences were tokenized to convert them into numerical representations suitable for input to the deep learning model, leveraging techniques such as BERT tokenizer.

### 2.6 Converting Tokenized Data to Tensors <a name="converting-tokenized-data-to-tensors"></a>

Tokenized sequences and labels were converted into tensors for efficient processing by the deep learning model, ensuring compatibility with PyTorch framework.

## 3. Fine-Tuning BERT Models: Model Training, Evaluation & Predictions <a name="fine-tuning-bert-models"></a>

### 3.1 BERT Model with Layers Freeze <a name="bert-model-with-layers-freeze"></a>

The BERT model was fine-tuned with initial layers frozen, followed by training and evaluation to assess model performance. Hyperparameter tuning and cross-validation techniques were employed to optimize model parameters.

### 3.2 BERT Model with All Layers <a name="bert-model-with-all-layers"></a>

An alternative approach involved fine-tuning the BERT model with all layers trainable, allowing for comprehensive exploration of the model's architecture. Evaluation metrics such as precision, recall, and F1-score were computed to gauge model efficacy.

## 4. Results <a name="results"></a>

The machine learning algorithm trained for detecting fake

 news demonstrated promising results, achieving an accuracy of over 70% on the test dataset. A Streamlit application was developed to provide a user-friendly interface for users to input news articles and determine their authenticity.

## 5. Conclusions <a name="conclusions"></a>

The project represents a significant step towards combating the proliferation of misinformation in digital ecosystems. By leveraging advanced natural language processing techniques, particularly based on BERT, the project demonstrates the feasibility of automating the detection of fake news articles with high accuracy.

## 6. Contributions <a name="contributions"></a>

The project underscores the collaborative efforts of the team members in data acquisition, analysis, model training, and evaluation. Each team member's contribution played a crucial role in achieving the project's objectives, thereby advancing the field of fake news detection through data science and deep learning techniques.

---
© 2024 Moh Jaiswal. All Rights Reserved.

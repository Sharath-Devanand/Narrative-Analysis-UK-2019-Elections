# Narrative Analysis UK 2019 Elections

## Project Background

This project aimed to classify narratives within tweets related to the 2019 UK General Election. Working as a team of four, we annotated tweets, engineered multiple preprocessing pipelines, experimented with prompt-based LLM classification, and trained several ML models to compare performance. The ultimate goal was to evaluate the efficacy of different classification strategies in identifying underlying super narratives in political discourse.


## Analysis

### Annotations

 - Each team member annotated a total of 600 tweets (3 sets of 200) using the annotation tool TeamWare. Each tweet was labeled with a primary super narrative and a corresponding confidence level (1–5). Additionally, a secondary super narrative and its confidence level were recorded where applicable, allowing for richer context and overlap in narrative interpretation.

- Each person also annotated one set from another teammate. The agreement between annotators was measured using Cohen’s Kappa Score to assess annotation consistency.


### Exploratory Analysis

- Super Narrative Distribution - Visualized how frequently each super narrative appeared across annotated tweets.

- Annotator Confidence Levels - Compared individual annotators' confidence in their labels.

- Confidence by Narrative - Investigated which narratives were more difficult to label with high certainty.


### Data Processing

To evaluate the impact of preprocessing on classification accuracy, we applied and compared five processing techniques:

1. Hashtag removal
2. Mention removal (or substitution)
3. Stopword and punctuation removal
4. Tokenisation
5. Emoji to text (in Pipeline 2)

Used LLaMA2 via ollama for classification, outputting binary labels (1 = relevant, 0 = not relevant). Iteratively tuned prompts for improved classification accuracy and minimal hallucination. Designed binary relevance prompts to assess the presence of a super narrative in a given tweet. Two pipelines were designed to evaluate the efficiency of preprocessing on Large language models.

Pipeline Design 1
Raw Data → Stopword & Punctuation Removal → Mention Removal → Tokenisation

Pipeline Design 2
Raw Data → Stopword & Punctuation Removal → Mention Removal → Hashtag Removal → Tokenisation → Emoji-to-Text


### Modelling

#### Elementary Models

Implemented machine learning pipeline (TF-IDF → Classifier) across 10 traditional models:

Models Tested: Naive Bayes, Logistic Regression, SVM, Random Forest, K - Nearest Neighbour, Multi-Layer Perceptron, XGBoost, AdaBoost, Voting Classifier, Stacking Classifier

Top 3 Models Based on F1 Score:
1. Support Vector Machine
2. Multi-Layer Perceptron
3. Stacked Classifier (Logistic Regression + Support Vector Machine + Multi-Layer Perceptron)

Additional Experiments:
1. Binary chaining with TF-IDF + Logistic Regression
2. Hyperparameter tuning with Grid Search for top models

#### Large Language Models (LLMs)

1. Distilled Embeddings: Used Birch (Distilled BERT Embeddings)
2. Standard Embeddings: Used BERT (Base BERT Embeddings)

Result: LLM-based methods outperformed traditional models on average F1 score across super narratives

## Technical Tool Used

- Python - Pandas, Numpy, Scikit-learn, NLTK, Transformers, Matplotlib, Seaborn


## Credits

Team Members - Gaurav Dhande, Munesh Khan, Shweta Kadke, Sharath Devanand
Supervisor - Carolina Scarton, Ibrahim Abu
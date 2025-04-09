# Narrative Analysis UK 2019 Elections

## Project Background

This project analyzes political narratives embedded within tweets from the 2019 UK General Election. Conducted by a team of four postgraduate students from the University of Sheffield, the aim was to annotate, classify, and evaluate super narratives using a combination of Large Language Models (LLMs) and traditional machine learning classifiers. The work is both a study in computational social science and a practical application of NLP pipelines.

## Analysis

### Annotations

 - Each team member annotated a total of 600 tweets (3 sets of 200) using the annotation tool TeamWare. Each tweet was labeled with a primary super narrative and a corresponding confidence level (1–5). Additionally, a secondary super narrative and its confidence level were recorded where applicable, allowing for richer context and overlap in narrative interpretation.

- Each person also annotated one set from another teammate. The agreement between annotators was measured using Cohen’s Kappa Score to assess annotation consistency.


### Data Processing

To evaluate the impact of preprocessing on classification accuracy, we applied and compared five processing techniques:

1. Hashtag removal
2. Mention removal (or substitution)
3. Stopword and punctuation removal
4. Tokenisation
5. Emoji to text

Used LLaMA2 via ollama for classification, outputting binary labels (1 = relevant, 0 = not relevant). Iteratively tuned prompts for improved classification accuracy and minimal hallucination. Designed binary relevance prompts to assess the presence of a super narrative in a given tweet. Two pipelines were designed to evaluate the efficiency of preprocessing on Large language models.


### Modelling

#### Elementary Models

Top 3 models (out of 10) based on F1 Score:
1. Support Vector Machine
2. Multi-Layer Perceptron
3. Stacked Classifier (Logistic Regression + Support Vector Machine + Multi-Layer Perceptron)

Additional Experiments:
1. Binary chaining with TF-IDF + Logistic Regression
2. Hyperparameter tuning with Grid Search for top models

#### Large Language Models (LLMs)

1. Distilled Embeddings: Used Birch (Distilled BERT Embeddings)
2. Standard Embeddings: Used BERT (Base BERT Embeddings)


### Insights

- The Support Vector Machine (SVM) consistently outperformed other models in overall classification metrics, particularly in identifying dominant narratives such as "Political hate and polarisation" and "Distrust in institutions."

- Manual double annotation significantly improved narrative interpretation, especially in cases where tweets contained subtle or multi-layered messaging. The shared annotation process also brought clarity and alignment among team members on how each narrative should be interpreted.

- Despite applying an ensemble approach through a Stacking Classifier, no significant improvement over individual models was observed. This suggests that the base classifiers used in the ensemble may not have been complementary or that the meta-model requires further optimization.

- A substantial class imbalance was found in the dataset. Narratives such as “Anti-Elites,” “Gender-related,” “Geopolitics,” and “Migration-related” were underrepresented and resulted in poor precision and recall. These findings indicate that models were unable to effectively learn patterns for these narratives.

- Some narratives, such as “None” and “Political hate and polarisation,” exhibited better performance. This is likely due to the presence of distinctive linguistic patterns or keywords that made them easier for models to identify.

- The complexity of certain narratives, such as “Distrust in institutions” or “Geopolitics,” may be due to the subtle language in tweets, which current feature extraction techniques failed to capture adequately. This suggests the need for more context-aware models or advanced embeddings.

- Annotator subjectivity played a significant role in performance discrepancies. Inconsistent interpretations, particularly in minority classes, may have led the models to learn incorrect patterns. A systematic adjudication process or expert review is recommended for future annotation efforts.

- Confidence scores were helpful in quantifying annotator certainty and can serve as an additional signal for training or weighting during model development.


## Technical Tool Used

- Python - Pandas, Numpy, Scikit-learn, NLTK, Transformers, Matplotlib, Seaborn


## Credits

Team Members - Gaurav Dhande, Munesh Khan, Shweta Kadke, Sharath Devanand
Supervisor - Carolina Scarton, Ibrahim Abu
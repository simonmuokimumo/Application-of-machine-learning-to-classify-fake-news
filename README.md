# APPLICATION OF MACHINE LEARNING TO CLASSIFY FAKE NEWS using Python
In this project, I collected two datasets; one containing fake news articles and the other genuine news articles, both focusing on Kenyan politics.
Afterward, I trained various machine learning models. Then through comparison analysis, I identified the best performing model (Support Vector Machine model: Accurancy of 96.36%), which I deployed and then I validated the prototype by collecting reviews from users.
# I USED THE FOLLOWING STEPS TO DEVELOP THE FAKE NEWS PROTOTYPE:
1. Data collection.
In this step, I collected fake news article dataset and true news articles dataset.
2. Data exploration.
In this phase, I visualized the distribution of fake and true articles datasets, created wordcloud for both the fake news dataset and true news dataset and did feature selection using chi-square algorithm to investgate feature importance.
3.  Data preparation.
Here I performed data preparation by implementing tokenization, removing stop words, addressing missing values, stemming and splitting the data into training, testing and validation sets. Additionally, I performed vectorization to convert the textual data into numerical representations.
4.  Machine learning modelling.
In this phase, I developed the following machine learning models; Logistic Regression, Naïve Bayes, Random Forest, Decision Trees, Gradient Boosting Classifier, Artificial Neural Network and Support Vector Machine.
5. Model Evaluation.
To analyze the performance of these models, I used various evaluation metrics such as accuracy, precision, recall, F1-Score, Area Under ROC Curve, sensitivity and specificity.
6. Deployment and Validation.
I developed the web application prototype using python and the Support Vector Machine model then I deployed it using Streamlit. To evaluate the performance of the prototype, I collected user reviews through a form embedded within the web application. The analysis of the received reviews indicated positive feedback regarding the functionality of the prototype.

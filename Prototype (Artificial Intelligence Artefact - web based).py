import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
pickle_in = open("SVM.pkl", "rb")
model = pickle.load(pickle_in)

# Load the vectorizer
with open("vectorization.pkl", "rb") as file:
    vectorization = pickle.load(file)

# Define the label mapping
label_mapping = {0: "Real News", 1: "Fake News"}

#evaluating the prototype
def save_review_to_csv(review):
    # Create or load existing CSV file
    try:
        df = pd.read_csv('reviews.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Review'])

    # Append new review to the DataFrame
    df = pd.concat([df, pd.DataFrame({'Review': [review]})], ignore_index=True)

    # Save DataFrame to CSV file
    df.to_csv('reviews.csv', index=False)



# Create a function to predict the sentiment of a text
def predict_sentiment(text):
    features = vectorization.transform([text]).toarray()
    prediction = model.predict(features)
    label = label_mapping.get(prediction[0], "Unknown")
    return label

# Create a title for the website
st.title('Fake News Detection Prototype')

# Create a text input field for the user to enter the text to be analyzed
text = st.text_input('Enter the title of the article to be analyzed:')




# If the user enters text, predict the sentiment and display the result
if text:
    prediction = predict_sentiment(text)
    st.write('Prediction:', prediction)

#st.title('Prototype validation')
#st.text("Give reviews about the system on the link below:")
#st.write("[LINK](https://forms.gle/x8rsWbDnKSDJ4nf1A)")


# evaluating the prototype

# Create form
st.title('Prototype Validation')
st.write('Please provide your feedback or review about the prototype.')
# Create text areas for user reviews
review1 = st.text_area('Enter Your First Name:')
review2 = st.text_area('Question 1: How easy is it to interact with the prototype?')
review3 = st.text_area('Question 2: Are there any technical issues or bugs encountered while using the prototype?')
review4 = st.text_area('Question 3: Which aspects of the prototype should be improved?')
review5 = st.text_area('Question 4: Does the prototype fulfill the intended purpose?')
review6 = st.text_area('Question 5: How can you rate the overall functionality of the prototype from 1-10?')
# Submit button
if st.button('Submit'):
    save_review_to_csv(review1)
    save_review_to_csv(review2)
    save_review_to_csv(review3)
    save_review_to_csv(review4)
    save_review_to_csv(review5)
    save_review_to_csv(review6)
    # Confirmation message
    st.write('Thank you for your feedback!')
import streamlit as st
import pickle

# Load models
with open('svm_linear_model.pkl', 'rb') as file:
    svm_linear = pickle.load(file)

with open('svm_rbf_model.pkl', 'rb') as file:
    svm_rbf = pickle.load(file)

with open('naive_bayes_model.pkl', 'rb') as file:
    nb_model = pickle.load(file)

# Load pre-fitted vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit app title
st.title("Email Spam Classifier")

# Email input
email_input = st.text_area("Enter the email content:")

# Model selection
model_choice = st.selectbox("Select a model:", ["Linear SVM", "RBF SVM", "Naive Bayes"])

# Predict button
if st.button("Classify Email"):
    if email_input.strip():
        # Transform the input email using the vectorizer
        email_tfidf = vectorizer.transform([email_input])

        # Model prediction
        if model_choice == "Linear SVM":
            prediction = svm_linear.predict(email_tfidf)[0]
        elif model_choice == "RBF SVM":
            prediction = svm_rbf.predict(email_tfidf)[0]
        else:
            prediction = nb_model.predict(email_tfidf)[0]

        # Display result
        if prediction == 1:
            st.success("The email is classified as SPAM.")
        else:
            st.success("The email is classified as NOT SPAM.")
    else:
        st.error("Please enter the email content.")

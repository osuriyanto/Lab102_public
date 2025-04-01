import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    model = joblib.load("sentiment_pipeline.pkl")

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    predicted_sentiment = ""
    if st.button("Predict"):
        input_series = pd.Series([userinput])  # convert to Series for compatibility
        predicted_sentiment = model.predict(input_series)[0]  # get the prediction (0 or 1)
        if predicted_sentiment == 1:
            output = 'positive 👍'
        else:
            output = 'negative 👎'
        sentiment=f'Predicted sentiment of "{userinput}" is {output}.'
        st.success(sentiment)

if __name__ == "__main__":
    run()
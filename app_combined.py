
import streamlit as st
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup
st.set_page_config(page_title="Unified Mental Health Assistant", layout="centered")
st.title("🧠 Your Mental Health Assistant")
st.markdown("Type anything — your feelings, questions, or upload survey data. I'm here for you. 💬")

# State management
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Intent model setup
intent_data = {
    "text": [
        "I feel stressed", "I'm anxious", "Hello", "Hi",
        "I'm sad", "Feeling low", "I'm overwhelmed", "Good morning"
    ],
    "intent": [
        "stress", "anxiety", "greeting", "greeting",
        "sadness", "sadness", "stress", "greeting"
    ]
}
df_intent = pd.DataFrame(intent_data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_intent['text'])
clf = LogisticRegression()
clf.fit(X, df_intent['intent'])

responses = {
    "greeting": ["Hi there! How can I support you today?"],
    "stress": ["Stress is tough. Let’s take a deep breath together."],
    "anxiety": ["It’s okay to feel anxious. I'm here with you."],
    "sadness": ["I’m sorry you feel this way. Want to talk more about it?"],
    "default": ["Tell me more, I'm listening."]
}

recommendations = {
    "stress": [
        {"title": "5-Minute Meditation", "url": "https://www.youtube.com/watch?v=inpok4MKVLM"},
        {"title": "Stretch & Relax", "url": "https://www.youtube.com/watch?v=qHJ992N-Dhs"}
    ],
    "anxiety": [
        {"title": "Anxiety Relief Music", "url": "https://www.youtube.com/watch?v=ZToicYcHIOU"},
        {"title": "Grounding Exercise", "url": "https://www.youtube.com/watch?v=KZXT7L4s0bY"}
    ],
    "sadness": [
        {"title": "Uplifting Music", "url": "https://www.youtube.com/watch?v=UfcAVejslrU"},
        {"title": "Talk on Depression", "url": "https://www.youtube.com/watch?v=XiCrniLQGYc"}
    ],
    "default": [
        {"title": "Mental Health Playlist", "url": "https://www.youtube.com/playlist?list=PLFzWFredxyJlR9L1_LPODw_JH6XkUnYVX"}
    ]
}

movie_recs = {
    "joy": ["Tamil: Oh My Kadavule", "Malayalam: Ustad Hotel"],
    "hopeful": ["Tamil: Raja Rani", "Malayalam: Charlie"],
    "sadness": ["Tamil: Vaaranam Aayiram", "Malayalam: Kumbalangi Nights"],
    "anger": ["Tamil: 96", "Malayalam: Bangalore Days"],
    "neutral": ["Tamil: Mersal", "Malayalam: Premam"]
}

def detect_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return "joy"
    elif polarity > 0.1:
        return "hopeful"
    elif polarity < -0.5:
        return "sadness"
    elif polarity < -0.1:
        return "anger"
    else:
        return "neutral"

def handle_input(user_input):
    if any(x in user_input.lower() for x in ["upload", "survey", ".csv"]):
        return "Please use the file uploader below to submit your survey CSV.", "upload"

    crisis_keywords = ["suicide", "kill myself", "end it", "hopeless", "give up"]
    if any(kw in user_input.lower() for kw in crisis_keywords):
        return "[⚠️ Crisis Detected] Please seek immediate professional help.", "crisis"

    input_vec = vectorizer.transform([user_input])
    intent = clf.predict(input_vec)[0] if clf.predict_proba(input_vec).max() > 0.4 else "default"
    sentiment = TextBlob(user_input).sentiment.polarity
    emotion = detect_emotion(user_input)
    reply = random.choice(responses.get(intent, responses["default"]))

    st.session_state.chat_log.append((user_input, reply))

    return f"{reply}\n🧠 Detected Emotion: `{emotion}`\n🎬 Suggested Feel-Good Movie: {random.choice(movie_recs[emotion])}", intent

user_input = st.text_input("💬 You:", "")
if user_input:
    response, intent = handle_input(user_input)
    st.markdown(f"**🤖 Assistant:** {response}")

if st.checkbox("📜 Show Chat Log"):
    for u, r in st.session_state.chat_log:
        st.markdown(f"**You:** {u}")
        st.markdown(f"**Bot:** {r}")

if st.checkbox("📈 Show Mood Over Time") and st.session_state.chat_log:
    df_log = pd.DataFrame(st.session_state.chat_log, columns=["user_input", "bot_response"])
    df_log["sentiment"] = df_log["user_input"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_log["sentiment_score"] = df_log["sentiment"].map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df_log["time"] = range(len(df_log))
    st.line_chart(df_log.set_index("time")["sentiment_score"])

st.header("📊 Upload Survey CSV")
uploaded_file = st.file_uploader("Upload your Google Form survey (.csv)", type="csv")
if uploaded_file:
    survey_df = pd.read_csv(uploaded_file)
    st.write("📝 Survey Preview", survey_df.head())

    if "Rate your overall mood (1–5)" in survey_df.columns:
        avg = survey_df["Rate your overall mood (1–5)"].mean()
        st.metric("🌤️ Average Mood Score", round(avg, 2))
        fig, ax = plt.subplots()
        sns.histplot(survey_df["Rate your overall mood (1–5)"], bins=5, ax=ax)
        st.pyplot(fig)

    if "What are you struggling with lately?" in survey_df.columns:
        text = " ".join(survey_df["What are you struggling with lately?"].dropna())
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.subheader("🧠 Common Concerns")
        st.image(wc.to_array())

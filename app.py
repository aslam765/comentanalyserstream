import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    layout="centered"
)

st.title("🎥 YouTube Sentiment Analyzer")
st.write("Analyze YouTube comments sentiment")

# ---------------- API KEY ----------------
API_KEY = st.secrets["YOUTUBE_API_KEY"]

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

# ---------------- EXTRACT VIDEO ID ----------------
def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    elif "watch?v=" in url:
        return url.split("v=")[1].split("&")[0]

    elif "/shorts/" in url:
        return url.split("/shorts/")[1].split("?")[0]

    elif "/live/" in url:
        return url.split("/live/")[1].split("?")[0]

    return None


# ---------------- GET COMMENTS ----------------
def get_comments(video_id):
    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50
        )

        response = request.execute()

        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)

    except Exception as e:
        st.error(f"Error fetching comments: {e}")

    return comments


# ---------------- ANALYZE COMMENTS ----------------
def analyze_comments(comments):
    positive = 0
    negative = 0
    neutral = 0

    comments = comments[:50]

    try:
        results = model(comments)

        for result in results:
            label = result["label"].lower()
            score = result["score"]

            # Neutral detection
            if score < 0.60:
                neutral += 1
            elif "pos" in label:
                positive += 1
            else:
                negative += 1

    except Exception as e:
        st.error(f"Analysis Error: {e}")

    return positive, negative, neutral


# ---------------- UI ----------------
url = st.text_input("Enter YouTube Video URL")

if st.button("Analyze"):

    if not url:
        st.warning("Please enter a YouTube URL")

    else:
        with st.spinner("Analyzing comments..."):

            video_id = extract_video_id(url)

            if not video_id:
                st.error("Invalid YouTube URL")

            else:
                comments = get_comments(video_id)

                if not comments:
                    st.error("No comments found")

                else:
                    pos, neg, neu = analyze_comments(comments)

                    st.success("Analysis Completed!")

                    # Metrics
                    col1, col2, col3 = st.columns(3)

                    col1.metric("😊 Positive", pos)
                    col2.metric("😡 Negative", neg)
                    col3.metric("😐 Neutral", neu)

                    # Pie Chart
                    fig, ax = plt.subplots()

                    ax.pie(
                        [pos, neg, neu],
                        labels=["Positive", "Negative", "Neutral"],
                        autopct="%1.1f%%"
                    )

                    ax.set_title("Sentiment Distribution")

                    st.pyplot(fig)

                    # Total comments analyzed
                    st.write(
                        f"Total Comments Analyzed: {len(comments)}"
                    )

import streamlit as st
import ollama
import pandas as pd
import re

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Emotion Detection",
    page_icon="😊",
    layout="centered"
)

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- TITLE ----------------

st.title("😊 AI Emotion Detection from Text")
st.write("Enter your text and let AI identify the emotion.")

st.divider()

# ---------------- TEXT INPUT ----------------

text = st.text_area(
    "📝 Enter your text",
    placeholder="Example: I am very happy because I got selected for my dream job!",
    height=120
)

# ---------------- DETECT EMOTION ----------------

if st.button("🔍 Detect Emotion", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text.")

    else:

        prompt = f"""
You are an AI emotion detection system.

Analyze the following text:

{text}

Choose exactly ONE emotion from:
Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral.

Also provide a confidence percentage between 0 and 100.

Give the response exactly in this format:

Emotion: Happy
Confidence: 95%
Reason: The person is expressing happiness.
Suggestion: Keep enjoying this positive moment.
"""

        try:

            response = ollama.chat(
                model="gemma3:latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            result = response["message"]["content"]

            # ---------------- EXTRACT DATA ----------------

            emotion_match = re.search(
                r"Emotion:\s*(.*)",
                result,
                re.IGNORECASE
            )

            confidence_match = re.search(
                r"Confidence:\s*(\d+)",
                result,
                re.IGNORECASE
            )

            reason_match = re.search(
                r"Reason:\s*(.*)",
                result,
                re.IGNORECASE
            )

            suggestion_match = re.search(
                r"Suggestion:\s*(.*)",
                result,
                re.IGNORECASE
            )

            emotion = (
                emotion_match.group(1).strip()
                if emotion_match
                else "Unknown"
            )

            confidence = (
                confidence_match.group(1)
                if confidence_match
                else "N/A"
            )

            reason = (
                reason_match.group(1).strip()
                if reason_match
                else "No reason available."
            )

            suggestion = (
                suggestion_match.group(1).strip()
                if suggestion_match
                else "No suggestion available."
            )

            # ---------------- EMOTION ICON ----------------

            icons = {
                "Happy": "😊",
                "Sad": "😢",
                "Angry": "😠",
                "Fear": "😨",
                "Surprise": "😲",
                "Disgust": "🤢",
                "Neutral": "😐"
            }

            icon = icons.get(emotion, "🎭")

            # ---------------- DISPLAY RESULT ----------------

            st.divider()

            st.subheader("🎯 Emotion Result")

            st.metric(
                label="Detected Emotion",
                value=f"{icon} {emotion}"
            )

            if confidence != "N/A":
                st.progress(
                    int(confidence) / 100
                )
                st.write(f"📊 Confidence: **{confidence}%**")

            st.info(f"💬 **Reason:** {reason}")

            st.success(f"💡 **AI Suggestion:** {suggestion}")

            # ---------------- SAVE HISTORY ----------------

            st.session_state.history.append(
                {
                    "Text": text,
                    "Emotion": emotion,
                    "Confidence": confidence
                }
            )

        except Exception as e:

            st.error("Unable to connect to Ollama.")

            st.write(
                "Make sure Ollama is running and "
                "gemma3:latest is installed."
            )

            st.code(str(e))


# ---------------- HISTORY ----------------

if st.session_state.history:

    st.divider()

    st.subheader("📝 Emotion History")

    df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    # ---------------- CHART ----------------

    st.subheader("📈 Emotion Statistics")

    emotion_counts = (
        df["Emotion"]
        .value_counts()
    )

    st.bar_chart(emotion_counts)

    # ---------------- DOWNLOAD ----------------

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download History",
        data=csv,
        file_name="emotion_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ---------------- CLEAR HISTORY ----------------

    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()
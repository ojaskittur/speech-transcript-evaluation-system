import streamlit as st
import json
from scorer import IntroductionScorer

st.set_page_config(page_title="Intro Scorer", layout="wide")

st.title(" Introduction Scoring System")
st.write("Enter your speech transcript and duration to get a detailed rubric score.")

with st.form("score_form"):
    transcript = st.text_area("Transcript", height=200, placeholder="Hello, my name is...")
    duration = st.number_input("Duration (seconds)", min_value=0, value=0)
    submitted = st.form_submit_button("Analyze Score")

if submitted and transcript:
    with st.spinner("Analyzing... (Loading AI models might take a moment)"):
        scorer = IntroductionScorer(transcript, duration)
        results = scorer.calculate_overall_score()
        
        st.metric(label="Total Score", value=f"{results['Total Score']} / 100")
        
        st.subheader("Detailed Breakdown")
        breakdown = results['Breakdown']
        
        for category, data in breakdown.items():
            with st.expander(f"{category} (Score: {data['score']})"):
                st.write(f"**Feedback:** {data['feedback']}")
                st.progress(data['score'] / (data.get('max', 10) if data.get('max') else 15))

        st.subheader("Raw JSON Data")
        st.json(results)
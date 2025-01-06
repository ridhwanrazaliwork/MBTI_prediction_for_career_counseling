import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from joblib import load
import nltk
import pypdf
from mbti_descriptions import mbti_descriptions #Import description
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Page config
st.set_page_config(layout="wide", page_title="MBTI Predictor")

# CSS styling
st.markdown(
    """
    <style>
        body {
            background-color: #ffffff;
            color: #000000;
            font-family: sans-serif;
        }
        .st-expander {
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 10px;
        }
       .stButton>button {
            color: white;
            background-color: #581845;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stSuccess {
            background-color: #e6f7f2;
            color: #28a745;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            font-weight: bold;

        }
         .stError {
            background-color: #ffe6e6;
            color: #dc3545;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            font-weight: bold;

        }
        .mbti-section {
             background-color: #f0f0f0;
            border-radius: 5px;
             padding: 10px;
            margin-top: 10px;
        }
        .mbti-header {
             background-color: #e0e0e0;
             border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }
        .careers-box {
             background-color: #f8f8f8;
             border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('vader_lexicon')


# Load the saved models
model_ie = load('logistic_regression_IE.pkl')
model_ns = load('logistic_regression_NS.pkl')
model_ft = load('logistic_regression_FT.pkl')
model_jp = load('logistic_regression_JP.pkl')

# Load the saved CountVectorizer and TfidfTransformer
cntizer = load('count_vectorizer.pkl')
tfizer = load('tfidf_transformer.pkl')

# Function to preprocess the input text
def preprocess_text(text):
    text = text.lower()
    # remove url links
    pattern = re.compile(r'https?://[a-zA-Z0-9./-]*/[a-zA-Z0-9?=_.]*[_0-9.a-zA-Z/-]*')
    text= re.sub(pattern, ' ', text)

    pattern2=re.compile(r'https?://[a-zA-Z0-9./-]*')
    text= re.sub(pattern2, ' ', text)
    
    #removing special characters and numbers from texts.
    pattern = re.compile('\W+')
    text= re.sub(pattern, ' ', text)
    pattern = re.compile(r'[0-9]')
    text= re.sub(pattern, ' ', text)
    pattern = re.compile(r'[_+]')
    text= re.sub(pattern, ' ', text)
    
    # removing extra spaces from texts.
    pattern = re.compile('\s+')
    text= re.sub(pattern, ' ', text)
    
    #remove stop words
    remove_words = stopwords.words("english")
    text=" ".join([w for w in text.split(' ') if w not in remove_words])
    
    #remove mbti personality words from text
    mbti_words =  ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
    text=" ".join([w for w in text.split(' ') if w not in mbti_words])
    
    #Lemmatization (grouping similar words)
    lemmatizer = WordNetLemmatizer()
    text=" ".join([lemmatizer.lemmatize(w) for w in text.split(' ')])
    return text

def predict_mbti(text):
    processed_text = preprocess_text(text)

    # Transform the input text using the fitted vectorizer and tfidf.
    # No refitting needed here
    X_cnt = cntizer.transform([processed_text])
    X_tfidf = tfizer.transform(X_cnt).toarray()
    
    # Make predictions using the loaded models
    pred_ie = model_ie.predict(X_tfidf)[0]
    pred_ns = model_ns.predict(X_tfidf)[0]
    pred_ft = model_ft.predict(X_tfidf)[0]
    pred_jp = model_jp.predict(X_tfidf)[0]
    
    # Convert predictions to MBTI type
    mbti_type = ""
    b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

    mbti_type += b_Pers_list[0][pred_ie]
    mbti_type += b_Pers_list[1][pred_ns]
    mbti_type += b_Pers_list[2][pred_ft]
    mbti_type += b_Pers_list[3][pred_jp]
    
    return mbti_type

def get_mbti_description(mbti_type, workplace_values, interests_skills):
    """Retrieves the description for a given MBTI type."""
    base_data = mbti_descriptions.get(mbti_type, {
        "overview": "No description available.",
        "strengths": "No strengths information.",
        "weaknesses": "No weakness information.",
        "work_style": "No work style information.",
        "communication_style": "No communication style information.",
         "careers": {
          "default":["No career information available"]
         }
    })
    if "careers" in base_data:
       
       career_key= "default"
       for key in base_data["careers"].keys():
            if key != "default" and (key in workplace_values or key in interests_skills):
                 career_key= key
                 break
       
       return {
            "overview": base_data["overview"],
            "strengths": base_data["strengths"],
            "weaknesses": base_data["weaknesses"],
            "work_style": base_data["work_style"],
            "communication_style": base_data["communication_style"],
            "careers": base_data["careers"][career_key]
           }
    else:
        return base_data

def analyze_text_input(text, workplace_values, team_role, interests_skills, scenario1_reflection, scenario2_reflection):
    """Analyzes the user's text input and returns relevant insights."""

    all_text = f"{text} {' '.join(workplace_values)} {team_role} {' '.join(interests_skills)} {scenario1_reflection} {scenario2_reflection}"
    processed_text=preprocess_text(all_text)
    word_counts = Counter(processed_text.split())
    
    # Extract some keywords
    keywords = ["innovative", "structured", "leading", "teamwork", "technology", "strategic","creative", "analytical", "helping", "communication", "organized"]
    keyword_counts = {keyword: word_counts[keyword] for keyword in keywords if keyword in word_counts}

    skill_counts = {skill: word_counts[skill.lower()] for skill in interests_skills if skill.lower() in word_counts}
    team_role_str=f"Prefers {team_role}"
    
    #Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(processed_text)
    
    if sentiment_scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores["compound"] <= -0.05:
      sentiment = "Negative"
    else:
      sentiment = "Neutral"
    
    return {
        "keyword_counts": keyword_counts,
        "skill_counts": skill_counts,
        "team_role":team_role_str,
        "sentiment": sentiment
    }

# Streamlit app
def main():
    st.title("MBTI Personality Predictor")
    st.header("Provide input in order to predict your MBTI type")

    def populate_dummy_input():
      st.session_state.dummy_text_input = "I am a person who enjoys being innovative and coming up with novel approaches, and prefer structured and goal oriented settings."
      st.session_state.dummy_workplace_values = ["Innovation", "Structure"]
      st.session_state.dummy_team_role = "Take a leading role"
      st.session_state.dummy_interests_skills = ["Technology", "Strategic Planning"]
      st.session_state.dummy_scenario1_reflection = "I would try to create a clear plan and then delegate tasks to team members, while keeping the vision in mind."
      st.session_state.dummy_scenario2_reflection = "I would gather more evidence, and see if I can run small scale test cases to prove the efficiency of this new approach."

    if st.button("Populate Dummy Input"):
      populate_dummy_input()

    with st.expander("Core Input", expanded = True):
      text_input = st.text_area(
          "Describe your personality, work preferences, and tendencies. For example, you can mention how you interact with others, your problem-solving approach, and how you typically make decisions.",
           placeholder = "Start here by describing how you are",
           key = "text_input",
           value = st.session_state.get("dummy_text_input", "")
          )
    with st.expander("Preferences and Skills"):
        workplace_values = st.multiselect(
          "Which of the following do you value most in the workplace?",
          ["Teamwork", "Autonomy", "Innovation", "Structure", "Recognition", "Growth"],
          key = "workplace_values",
          default = st.session_state.get("dummy_workplace_values", [])
        )
        team_role=st.radio(
         "When you are in a team, do you prefer to:",
          ["Take a leading role", "Contribute as a team member", "Provide technical expertise"],
          key = "team_role",
          index = ["Take a leading role", "Contribute as a team member", "Provide technical expertise"].index(st.session_state.get("dummy_team_role","Contribute as a team member")) if st.session_state.get("dummy_team_role", None) else 1
          )
        interests_skills = st.multiselect(
            "Select your interests or skills:",
            ["Art", "Science", "Technology", "Helping People", "Strategic Planning",
             "Problem-solving", "Creativity", "Communication", "Organization"],
            key = "interests_skills",
            default = st.session_state.get("dummy_interests_skills",[])
          )
    with st.expander("Scenario Reflections"):
      st.subheader("Scenario 1")
      scenario1 = st.text_input(
           "You are assigned to lead a project with a tight deadline and conflicting team opinions. What would be your action and approach?",
           disabled=True, key = "scenario1")
      scenario1_reflection = st.text_area(
          "Your reflection:",
           height=80, key = "scenario1_reflection",
           value = st.session_state.get("dummy_scenario1_reflection", "")
       )

      st.subheader("Scenario 2")
      scenario2 = st.text_input(
             "You are presented with a novel approach that can solve a problem efficiently. However, there is no historical data, and this idea is considered novel. What would be your approach?",
             disabled=True, key = "scenario2")
      scenario2_reflection = st.text_area(
             "Your reflection:",
            height=80, key = "scenario2_reflection",
            value = st.session_state.get("dummy_scenario2_reflection", "")
         )
    with st.expander("Resume Upload (Optional)"):
        uploaded_file = st.file_uploader("Upload a Resume (TXT/PDF)", type=["txt","pdf"])
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                     pdf_reader = pypdf.PdfReader(uploaded_file)
                     text_input=""
                     for page_num in range(len(pdf_reader.pages)):
                         page = pdf_reader.pages[page_num]
                         text_input += page.extract_text()
                elif uploaded_file.type == "text/plain":
                     text_input = uploaded_file.read().decode("utf-8")
            except Exception as e:
                 st.error(f"An error occurred: {e}")

    st.sidebar.header("Additional Insights")
    if st.button("Predict MBTI"):
            combined_text=f"{text_input} {' '.join(workplace_values)} {team_role} {' '.join(interests_skills)} {scenario1_reflection} {scenario2_reflection}"
            predicted_mbti = predict_mbti(combined_text)
            st.markdown(f'<div class="mbti-header"> <b>Predicted MBTI Type:</b></div>', unsafe_allow_html=True)
            st.title(f"{predicted_mbti}")
            
            description = get_mbti_description(predicted_mbti, workplace_values, interests_skills)
            st.markdown(f'<div class="mbti-section"> <b>MBTI Type Insights:</b></div>', unsafe_allow_html=True)
            st.markdown(f"**Overview:** {description['overview']}")
            st.markdown(f"**Strengths:** {description['strengths']}")
            st.markdown(f"**Potential Weaknesses:** {description['weaknesses']}")
            st.markdown(f"**Preferred Work Style:** {description['work_style']}")
            st.markdown(f"**Communication Style:** {description['communication_style']}")

            st.markdown(f'<div class="careers-box"> <b>Career Recommendations:</b></div>', unsafe_allow_html=True)
            for career in description['careers']:
                st.markdown(f"- {career}")
            text_analysis_result = analyze_text_input(text_input, workplace_values, team_role, interests_skills, scenario1_reflection, scenario2_reflection)

            st.sidebar.markdown("**Keywords Mentioned:**")
            for keyword, count in text_analysis_result["keyword_counts"].items():
               st.sidebar.markdown(f"- {keyword}: {count}")
            st.sidebar.markdown("**Skills Selected:**")
            for skill, count in text_analysis_result["skill_counts"].items():
               st.sidebar.markdown(f"- {skill}: {count}")
            st.sidebar.markdown(f"**Team Preference**")
            st.sidebar.markdown(f"- {text_analysis_result['team_role']}")
            st.sidebar.markdown(f"**Overall Sentiment:** {text_analysis_result['sentiment']}")

if __name__ == "__main__":
    main()
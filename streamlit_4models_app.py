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
from mbti_descriptions import mbti_descriptions, malaysia_profile #Import description
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

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
    """
    Retrieves the description for a given MBTI type and adapts the careers
    section based on workplace values and interests/skills.
    """
    # Fetch the base description for the MBTI type
    base_data = mbti_descriptions.get(mbti_type, {
        "overview": "No description available.",
        "strengths": "No strengths information.",
        "weaknesses": "No weakness information.",
        "work_style": "No work style information.",
        "communication_style": "No communication style information.",
        "careers": {
            "default": ["No career information available"]
        }
    })

    # Check if careers are available in the data
    if "careers" in base_data:
        # Default career key
        career_key = "default"
        # Find a specific career key matching workplace_values or interests_skills
        for key in base_data["careers"].keys():
            if key != "default" and (key in workplace_values or key in interests_skills):
                career_key = key
                break

        # Return the description with the appropriate career key
        return {
            "overview": base_data["overview"],
            "strengths": base_data["strengths"],
            "weaknesses": base_data["weaknesses"],
            "work_style": base_data["work_style"],
            "communication_style": base_data["communication_style"],
            "careers": base_data["careers"][career_key],
            "career_details": base_data.get("career_details", {})
        }

    # If no "careers" field, return the base data
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
            if "career_details" in description:
              st.markdown(f'<div class="careers-box"> <b>Career Details:</b></div>', unsafe_allow_html=True)
              for career in description['careers']:
                 if career in description['career_details']:
                    st.markdown(f"**{career}**: Tasks: {description['career_details'][career]['tasks']}. Skills:{', '.join(description['career_details'][career]['skills'])}. Salary Range: {description['career_details'][career]['salary_range']}")
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
    
    with st.sidebar.expander("Information"):
        info_tab, training_data_tab = st.tabs(["About", "Training Data"])
        with info_tab:
            st.markdown(f"People from Malaysia are likely to be slightly more **Introverted** than Extraverted (+1.17%), slightly more **Intuitive** than Observant (+0.71%), significantly more **Feeling** than Thinking (+18.52%), slightly more **Prospecting** than Judging (+4.30%), and significantly more **Turbulent** than Assertive (+11.74%).")
            st.markdown("Top MBTI types in Malaysia:")
            for i, type in enumerate(malaysia_profile["top_mbti_types"][:10]):
                st.markdown(f"{i+1}. {type}")
            st.markdown(
                """
                **About MBTI:**
                The Myers-Briggs Type Indicator (MBTI) is a self-report questionnaire designed to indicate different psychological preferences in how people perceive the world and make decisions.

                **How to Use This Tool:**
                This tool helps you explore your potential MBTI personality type by analyzing your written responses.

                1. Start by describing your personality.
                2. Select your workplace values and skills.
                3. Reflect on the scenarios presented.
                4. Click `Predict MBTI` to get results.

                **Limitations:**
                The accuracy of the MBTI result and career recommendations depends on the quality and amount of the information that you provide. These results should be used as a guide only, and not as a final solution. If you are unsure on any result, please reach out to career counselors.

                **Data Privacy:**
                 This app does not collect any of your input data. All the processing happens locally, and nothing is saved on the server.

                 **Contact Information:**
                 If you have any questions, please reach out to: contact@email.com
                """
            )
        with training_data_tab:
           mbti_df= pd.read_csv("mbti_1.csv")
           mbti_counts= mbti_df['type'].value_counts()
           fig, ax = plt.subplots(1,3,figsize=(12,4))
           ax[0].bar(mbti_counts.index, mbti_counts.values)
           ax[0].set_xlabel('MBTI Type')
           ax[0].set_ylabel('Number of Posts')
           ax[0].set_title('Distribution of MBTI Types in Training Dataset')
           for tick in ax[0].get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right') # Rotate x-axis labels to avoid overlap

           roles = list(malaysia_profile["roles"].keys())
           role_counts = list(malaysia_profile["roles"].values())
           ax[1].bar(roles, role_counts)
           ax[1].set_xlabel("Roles")
           ax[1].set_ylabel("Percentage")
           ax[1].set_title("Roles Distribution in Malaysia")
            
           strategies = list(malaysia_profile["strategies"].keys())
           strategy_counts = list(malaysia_profile["strategies"].values())
           ax[2].bar(strategies, strategy_counts)
           ax[2].set_xlabel("Strategies")
           ax[2].set_ylabel("Percentage")
           ax[2].set_title("Strategies Distribution in Malaysia")
           for tick in ax[2].get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right') # Rotate x-axis labels to avoid overlap
            
           plt.tight_layout()
           st.pyplot(fig)

           st.markdown("**Trait Comparisons (Malaysia vs World):**")
           traits = malaysia_profile["trait_comparisons"]["traits"]
           malaysia_values=malaysia_profile["trait_comparisons"]["malaysia_values"]
           world_values = malaysia_profile["trait_comparisons"]["world_values"]
            
           comparison_df = pd.DataFrame(
            {
               'Trait': traits,
               'Malaysia': malaysia_values,
               'World': world_values
           })
           st.dataframe(comparison_df)

if __name__ == "__main__":
    main()
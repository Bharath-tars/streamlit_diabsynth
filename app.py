# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
df = pd.read_csv("diabetes.csv")

# Customizing Streamlit theme for dark mode
st.markdown(
    """
    <style>
        body {
            color: white;
            background-color: #1E1E1E;
        }
        .sidebar .sidebar-content {
            background-color: #1E1E1E;
        }
        .Widget>label {
            color: white;
        }
        .stButton>button {
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Page 1: Project Description
def project_description():
    st.title("Predictive Health Analytics for Diabetes Risk Assessment and Personalized Reporting")

    # Our Mission
    st.header("Our Mission")
    st.write("Welcome to DiabSynth, the frontier of predictive health analytics, where the synergy of algorithms and empathy "
             "transforms the very fabric of diabetes care. Within our consortium, trailblazing minds, visionary engineers, "
             "and dedicated healthcare aficionados unite with unwavering determination. Our collective mission is nothing short "
             "of redefining the boundaries of healthcare technology, ushering in a new era in diabetes management that transcends the ordinary.")

    # Our Vision
    st.header("Our Vision")
    st.write("Our vision transcends the confines of algorithms and raw data. DiabSynth aspires to manifest a world where individuals "
             "don't merely receive healthcare insights but hold the knowledge and tools to seize control of their well-being. In our "
             "revolutionary pursuit, we propel beyond the conventional boundaries, determined to bridge the formidable gap between "
             "sophisticated machine learning techniques and pragmatic, user-friendly medical reporting.")

# Page 2: Team Members Details
def team_member_details():
    st.title("Team Members")
    
    team_members = [
        {"name": "Sudarsanam Bharath", "role": "Team Lead", "bio": "Hey everyone, I'm Bharath, a skilled multitasker capable of getting the work done before due with high accuracy and consistency. I'm a tech enthusiast with a primary goal of learning almost every technology possible that the market requires. Revolutionary in thought and code, I spearhead our machine learning endeavor. With a background in Full Stack Development, Machine Learning & Advanced algorithms, UI&UX Design, and API's Development, I bring a visionary approach to decoding the complexities of diabetes analytics.", "image": "bharath.jpg"},
        {"name": "Pooja Chinta", "role": "Data Insights Specialist", "bio": "I am Pooja, an undergraduate at MLR Institute of Technology, donning the role of a Data Insights Specialist in this project. With a keen focus on unraveling the stories hidden within vast datasets, I bring a meticulous approach to Data Analytics, Data Extraction, and Data Preprocessing. My expertise lies in transforming raw data into actionable insights, navigating the complexities of diverse datasets to extract meaningful narratives.", "image": "pooja.jpg"},
        {"name": "Yenuganti Sai Kumar", "role": "Research Coordinator", "bio": "I'm honored to take on the pivotal role of Research Coordinator in the DiabSynth project. As a dedicated and insightful team member, my focus is on delving into the latest advancements in predictive health analytics. I take the lead in ensuring our team is well-informed about cutting-edge research, industry trends, and regulatory developments, contributing valuable insights to shape DiabSynth's innovative approach.", "image": "sai_kumar.jpg"},
        {"name": "Talari Lakshmi", "role": "User Experience Advocate", "bio": "I, as the User Experience Advocate within the DiabSynth project, am dedicated to championing the end-user perspective. My role ensures that DiabSynth's reports go beyond technical accuracy to resonate with users, contributing to a positive and user-friendly experience. My advocacy for user-centric design principles is integral to DiabSynth's mission of providing actionable insights for proactive diabetes management.", "image": "lakshmi.jpg"}
    ]

    for member in team_members:
        st.subheader(member["name"])
        st.write(f"**Role:** {member['role']}")
        st.write(member["bio"])

# Page 3: Diabetes Prediction
def diabetes_prediction():
    st.title('Diabetes Checkup')
    st.sidebar.header('Patient Data')
    st.subheader('Training Data Stats')
    st.write(df.describe())

    # X AND Y DATA
    x = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # FUNCTION
    def user_report():
        pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
        glucose = st.sidebar.slider('Glucose', 0, 200, 120)
        bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
        skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
        insulin = st.sidebar.slider('Insulin', 0, 846, 79)
        bmi = st.sidebar.slider('BMI', 0, 67, 20)
        dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
        age = st.sidebar.slider('Age', 21, 88, 33)

        user_report_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skinthickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        report_data = pd.DataFrame(user_report_data, index=[0])
        return report_data

    # PATIENT DATA
    user_data = user_report()
    st.subheader('Patient Data')
    st.write(user_data)

    # MODEL
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    user_result = rf.predict(user_data)

    # VISUALIZATIONS
    st.title('Visualised Patient Report')

    # COLOR FUNCTION
    color = 'red' if user_result[0] == 1 else 'blue'

    # Age vs Pregnancies
    st.header('Pregnancy count Graph (Others vs Yours)')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
    ax2 = sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 20, 2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_preg)

    # Add similar blocks for other visualizations...

    # OUTPUT
    st.subheader('Your Report: ')
    output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
    st.title(output)
    st.subheader('Accuracy: ')
    st.write(str(accuracy_score(y_test, rf.predict(x_test)) * 100) + '%')

# Page 4: Contact Us

def contact_us():
    st.title("Contact Us")
    st.write("Feel free to reach out to us with any questions or feedback!")

    # Simple contact form
    name = st.text_input("Your Name:")
    email = st.text_input("Your Email:")
    message = st.text_area("Message:")

    if st.button("Submit"):
        # Perform any desired action with the form data (e.g., send an email)
        st.success("Ticket Raised!!.. we will get back to you asap")

# Main App
def main():
    st.sidebar.title("DiabSynth Navigator")
    page = st.sidebar.radio("Choose a Path", ["Project Description", "Home", "About", "Contact", "Blogs and Reference", "Diabetes Prediction"])
# Page routing
    if page == "Project Description":
        	project_description()
    # Include your Home page content here
    elif page == "About":
    		st.markdown("[Click Here for About:](https://diabsynth.netlify.app/homepage/about)")
    # Include your About page content here
    elif page == "Diabetes Prediction":
    		diabetes_prediction()
    # Include your Diabetes Prediction page content here
    elif page == "Contact":
    		st.markdown("[Click Here for Contact:](https://diabsynth.netlify.app/homepage/contact)")
    # Include your Contact page content here
    elif page == "Blogs and Reference":
    		st.markdown("[Click Here for Blogs and Reference:](https://diabsynth.netlify.app/homepage/faq)")
    elif page == "Project Description":
        	project_description()
    elif page == "Home":
    		st.markdown("[Click Here for Home:](https://diabsynth.netlify.app/homepage/index.html)")
    elif page == "Team Members":
        	team_member_details()
    elif page == "Diabetes Prediction":
        	diabetes_prediction()
    elif page == "Contact Us":
        	contact_us()
if __name__ == "__main__":
    main()

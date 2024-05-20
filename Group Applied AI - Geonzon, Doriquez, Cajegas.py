from copyreg import pickle
import streamlit as st
import pickle


st.set_page_config(page_title="Inappropriate Speech Detector", layout="centered")


with st.sidebar:
    st.subheader("About")
    st.markdown("<h1 style='text-align: center; margin-top: 50px;'>Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; margin-top: 10px;'>This tool detects hateful language in text input.</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Upload your text and click 'Detect Hateful Language' to get started.</h4>", unsafe_allow_html=True)


st.title("Inappropriate Speech Detector")
st.markdown("<h5 style='text-align: center; margin-top: 50px;'>Enter Text Here</h5>", unsafe_allow_html=True)


speech = st.text_input("Enter your text:", key="speech_input")


if st.button("Detect Hateful Language"):
    if not speech:
        st.error("Please Enter a Valid Text")
    else:
        
        model = pickle.load(open('hate_speech_model.pkl',"rb"))
        cv = pickle.load(open("cv.pkl","rb"))

        
        data = cv.transform([speech]).toarray()

        
        pred = model.predict(data)

        if pred == "Offensive Language":
            st.error("Offensive Language Detected")
            
            col1, col2 = st.columns([1, 1])  
            with col1:
                st.image("hate_speech_image.gif", width=300, use_column_width=True)

            with col2:
                st.write("Offensive Language detected.")
                st.write("Why would you even write such...thing...")

            st.info("This speech contains offensive language. Please be cautious when using such language.")
            
            
            st.markdown("---")
            
        elif pred == "No Hate and Offensive language":
            st.success("No Hate and Offensive Language was detected")
            
            col1, col2 = st.columns([1, 1])  
            with col1:
                st.image("happy.gif", width=300, use_column_width=True)

            with col2:
                st.write("No hate speech detected.")
                st.write("Always spread positivity and good energy :>")

            st.info("This speech does not contain hate or offensive language. Continue spreading positivity!")
            
            st.markdown("---")
            
        elif pred == "Hate Speech":
            st.error("Hateful Speech Detected")
            
            col1, col2 = st.columns([1, 1])  
            with col1:
                st.image("giphy.gif", width=300, use_column_width=True)

            with col2:
                st.write("Hateful Speech detected.")
                st.write("Why would you even write such...thing...")

            st.info("This speech contains offensive language. Please be cautious when using such language.")
            st.markdown("---")

        else:
            st.warning("Not Found")

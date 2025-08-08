# app_final.py
import streamlit as st
from src.predict import predict_message


# Sidebar
st.sidebar.title("About")
st.sidebar.info("SpamLens: Where AI inspects every message")
st.sidebar.info("AI-powered spam detection using Naive Bayes.")

# Main content
st.title("📧 SpamLens (AI-Powered Spam Detection) ")
st.markdown("Enter a message below to check if it's spam.")

message = st.text_area("Message:", placeholder="Type your message here...")

if st.button("🎯 Try Sample Spam"):
    message = "Congratulations! You've won $1000. Click now to claim!"
    st.session_state.message = message

if st.button("🔍 Analyze"):
    if not message.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            pred, prob = predict_message(message)
        
        # Display result with style
        if pred == 1:
            st.markdown("<h3 style='color:red; text-align:center;'>🚨 SPAM DETECTED</h3>", unsafe_allow_html=True)
            st.error(f"This message is likely **spam**.")
            st.progress(int(prob * 100))
            st.write(f"**Confidence:** {prob:.2f}")
        else:
            st.markdown("<h3 style='color:green; text-align:center;'>✅ SAFE</h3>",unsafe_allow_html=True)
            st.success(f"This message is likely **not spam**.")
            st.progress(int((1 - prob) * 100))
            st.write(f"**Confidence:** {1 - prob:.2f}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
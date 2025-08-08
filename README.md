# 📧 SpamLens -AI-Powered Spam Detection

SpamLens is an AI-powered spam detection app that looks deep into the messages to decide whether they’re spam or safe using **Naive Bayes** and **TF-IDF**.  
Built from scratch using Python, scikit-learn, and Streamlit.

> 🔍 **Try it live**: [Streamlit App](https://Spamlens.streamlit.app)  
> 💻 **Code**: [GitHub Repository](https://github.com/Sandeep-060/SpamLens)

---

## 🎯 What It Does

- Takes a message as input
- Uses a trained AI model to predict if it's **spam** or **not spam**
- Shows prediction with **confidence score**
- Built with clean, modular code


---

## 🧰 Tech Stack

| Tool | Purpose |
|------|--------|
| `Python` | Core language |
| `Pandas` | Load and clean data |
| `scikit-learn` | Train Naive Bayes model |
| `TF-IDF Vectorizer` | Convert text to numbers |
| `Streamlit` | Build the web interface |
| `joblib` | Save and load the trained model |

---
## 📂 Run Locally  
```bash
git clone https://github.com/Sandeep-060/SpamLens.git
cd ai-spam-detector
pip install -r requirements.txt
streamlit run app_final.py
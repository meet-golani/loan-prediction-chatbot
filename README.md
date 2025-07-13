# 🏦 Loan Eligibility Q&A Chatbot + Prediction Tool

A **Streamlit**-based intelligent assistant that allows users to:
- Ask natural language questions about a loan dataset using Retrieval-Augmented Generation (RAG)
- Instantly see suggested FAQs
- Upload a test dataset CSV and receive **loan approval predictions** using a trained **Random Forest Classifier**

> 📌 Built with 💻 Python, 🤖 Transformers, 🎯 scikit-learn, and 🧠 SentenceTransformers.

---

## 🚀 Features

- 🔍 **Q&A Chatbot** – Ask questions like:
  - “How many loans were approved for male applicants?”
  - “What is the average loan amount for self-employed people?”
  - “Tell me about married applicants with high income.”
- 💡 Suggested Questions panel on startup
- 🧠 **RAG Pipeline**: Retrieves relevant data and answers using `google/flan-t5-base`
- 📊 **Loan Prediction**: Upload test CSVs and get predictions via a Random Forest model
- 📥 **Downloadable Results**: Save predictions as `.csv`

---

## 🧪 Files in This Repository

| File | Description |
|------|-------------|
| `app.py` | Streamlit frontend combining Q&A + prediction |
| `model.py` | ML model training script using RandomForestClassifier |
| `Training Dataset.csv` | Dataset used for training + RAG context |
| `Test Dataset.csv` | Sample input file for predictions |
| `Result.mp4` | (Optional) Demo video of the app in action |

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the GitHub repository
git clone https://github.com/meet-golani/loan-prediction-chatbot.git
cd loan-prediction-chatbot

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Mac/Linux

# 3. Install all required dependencies
pip install --upgrade pip
pip install streamlit pandas scikit-learn sentence-transformers transformers faiss-cpu

# 4. Run the Streamlit app locally
streamlit run app.py

# 5. Initialize Git and push to GitHub (if you're uploading it)
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/meet-golani/loan-prediction-chatbot.git
git pull origin main --allow-unrelated-histories
git push -u origin main
```
📸 UI Highlights
Chatbot with suggested questions
Text input for custom queries
Prediction table + CSV download
Dark theme layout

🧠 Model Details
ML Algorithm: RandomForestClassifier (sklearn)
Text Embedding: all-MiniLM-L6-v2 (via sentence-transformers)
Answer Generation: google/flan-t5-base (via transformers.pipeline)

✍️ Authors
Meet Golani
Built during internship for data science and ML integration using Streamlit + Transformers.

📄 License
This project is under the Apache 2.0 License.

Thanks for checking out the Loan Prediction Chatbot!
Feel free to ⭐ star the repo or fork it to add your own enhancements.
If you found this helpful, connect with me on GitHub or leave feedback!
Happy Building! 🚀

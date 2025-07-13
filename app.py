import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model():
    train_df = pd.read_csv("Training Dataset.csv")  # or "train.csv"
    train_df.dropna(inplace=True)
    
    X = train_df.drop(columns=["Loan_ID", "Loan_Status"])
    y = train_df["Loan_Status"]

    X = pd.get_dummies(X)
    y = LabelEncoder().fit_transform(y)
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    return model, X.columns

model, train_columns = train_model()


import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import numpy as np
import faiss

# Load data
df = pd.read_csv("Training Dataset.csv")
docs = df.astype(str).apply(lambda row: ' | '.join(row), axis=1).tolist()

# Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, show_progress_bar=True)
dimension = doc_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Load model
qa_model = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=128)

# Precomputed stats
summary_stats = {
    "approved_males": df[(df["Gender"] == "Male") & (df["Loan_Status"] == "Y")].shape[0],
    "average_loan_self_employed": round(df[df["Self_Employed"] == "Yes"]["LoanAmount"].mean(), 1),
    "total_approved": df[df["Loan_Status"] == "Y"].shape[0],
}

# Streamlit config
st.set_page_config(page_title="Loan RAG Chatbot", layout="centered")
st.title("📊 Loan Eligibility Q&A Chatbot")
st.markdown("Ask a question related to the loan dataset (e.g., approvals, income, education).")

# Suggested questions
suggested_questions = [
    "What is the number of approvals?",
    "How many loans were approved for male applicants?",
    "What is the average loan amount for self-employed people?",
    "Tell me about married applicants with high income.",
]

with st.expander("💡 Suggested Questions"):
    for q in suggested_questions:
        if st.button(q):
            st.session_state.query = q

# Input box
query = st.text_input("🔍 Enter your question:", key="query")

# RAG functions
def retrieve(query, k=20):
    query_vec = embedder.encode([query])
    _, I = index.search(np.array(query_vec), k)
    return [docs[i] for i in I[0]]

def generate_answer(query, context):
    max_context_length = 400
    context = context[:max_context_length]
    prompt = f"""Use the data below to answer the question:\n{context}\n\nQuestion: {query}"""
    result = qa_model(prompt)[0]["generated_text"]
    return result

def rag_pipeline(query):
    q = query.lower()

    if "approved" in q and "male" in q:
        return f"{summary_stats['approved_males']} loans were approved for male applicants."

    if "average loan amount" in q and "self-employed" in q:
        return f"The average loan amount for self-employed people is ₹{summary_stats['average_loan_self_employed']}."

    if "number of approvals" in q or ("how many" in q and "approved" in q):
        return f"There are {summary_stats['total_approved']} approved loans."

    if q.strip() in ["hi", "hello", "hey"]:
        return "👋 Hello! Ask me something about loan approvals, income, or education from the dataset."

    relevant_docs = retrieve(query, k=20)
    context = "\n".join(relevant_docs)
    return generate_answer(query, context)

# Run pipeline
if query:
    with st.spinner("Thinking..."):
        answer = rag_pipeline(query)
    st.success("Answer:")
    st.markdown(f"**{answer}**")

st.sidebar.header("📄 Predict Loan Approvals on Test Data")

uploaded_file = st.sidebar.file_uploader("Upload test dataset (CSV)", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    
    # Preprocess test data
    test_processed = pd.get_dummies(test_df)
    test_processed = test_processed.reindex(columns=train_columns, fill_value=0)
    
    # Predict
    predictions = model.predict(test_processed)
    test_df["Loan_Status_Prediction"] = ["Y" if pred == 1 else "N" for pred in predictions]

    # Show result
    st.subheader("🔮 Prediction Results")
    st.dataframe(test_df)

    # Download button
    csv = test_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("/content/Training Dataset.csv")

# Combine into a single string per row for embedding
docs = df.astype(str).apply(lambda row: ' | '.join(row), axis=1).tolist()

# Preprocessing: Add a grouped summary dictionary
loan_data = pd.read_csv("/content/Training Dataset.csv")

summary_stats = {
    "approved_males": loan_data[(loan_data["Gender"] == "Male") & (loan_data["Loan_Status"] == "Y")].shape[0],
    "average_loan_self_employed": loan_data[loan_data["Self_Employed"] == "Yes"]["LoanAmount"].mean()
}

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model (lightweight)
model = SentenceTransformer("all-MiniLM-L6-v2")  # 40MB, good performance

# Convert to vectors
doc_embeddings = model.encode(docs, show_progress_bar=True)

# Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings[0].shape[0])
index.add(np.array(doc_embeddings))

from transformers import pipeline

# Load a lightweight generative model (FLAN-T5)
qa_model = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=128)

def generate_answer(query, context):
    prompt = f"""Use the data below to answer the question:\n{context}\n\nQuestion: {query}"""
    result = qa_model(prompt)[0]["generated_text"]
    return result

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query, k=20):  # Increase from 5 to 20 or even 50
    query_vec = embedder.encode([query])
    _, I = index.search(query_vec, k)
    return [docs[i] for i in I[0]]

def rag_pipeline(query):
    q = query.lower()

    if "approved" in q and "male" in q:
        count = loan_data[(loan_data["Gender"] == "Male") & (loan_data["Loan_Status"] == "Y")].shape[0]
        return f"{count} loans were approved for male applicants."

    if "average loan amount" in q and "self-employed" in q:
        avg = loan_data[loan_data["Self_Employed"] == "Yes"]["LoanAmount"].mean()
        return f"The average loan amount for self-employed people is ₹{round(avg, 1)}."

    # Else use retrieval + LLM
    relevant_docs = retrieve(query, k=20)
    context = "\n".join(relevant_docs)
    return generate_answer(query, context)


from transformers import pipeline

qa_model = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=128)


def generate_answer(query, context):
    # Truncate context if too long
    max_context_length = 400  # safe buffer under 512
    context = context[:max_context_length]

    prompt = f"""Use the data below to answer the question:\n{context}\n\nQuestion: {query}"""
    result = qa_model(prompt, max_new_tokens=128)[0]["generated_text"]
    return result


query1 = "How many loans were approved for male applicants?"
query2 = "What is the average loan amount for self-employed people?"

print("Q1:", query1)
print("A1:", rag_pipeline(query1))
print("\nQ2:", query2)
print("A2:", rag_pipeline(query2))

print(rag_pipeline("What documents are needed for loan approval?"))
print(rag_pipeline("What is the trend in applicant education and loan approval?"))
print(rag_pipeline("How many loans were approved for male applicants?"))  # Static
print(rag_pipeline("What is the average loan amount for self-employed people?"))  # Static

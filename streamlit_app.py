import streamlit as st
from PyPDF2 import PdfReader
import docx
import requests

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ''
    return text

# Function to extract text from a DOCX
def extract_text_from_docx(doc_file):
    doc = docx.Document(doc_file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Function to interact with Hugging Face API for summarization
def summarize_text_hf(text):
    api_key = st.secrets["huggingface_api_key"]  # Get Hugging Face API key securely
    model = "facebook/bart-large-cnn"  # Summarization model

    url = f"https://api-inference.huggingface.co/models/{model}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 512,  # Control the max length of the summary
            "min_length": 100,
            "do_sample": False
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        summary = response.json()[0].get('summary_text', 'Sorry, I could not summarize the document.')
    else:
        summary = f"Error: {response.status_code}, {response.text}"
    
    return summary

# Function to interact with Hugging Face API for Q&A based on the summarized document
def generate_answer_hf(summary, question):
    api_key = st.secrets["huggingface_api_key"]  # Get Hugging Face API key securely
    model = "bigscience/bloom"  # Generative model

    url = f"https://api-inference.huggingface.co/models/{model}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    prompt = f"Context: {summary}\nQuestion: {question}\nAnswer:"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,  # You can adjust this to control the length of the answer
            "temperature": 0.5
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        generated_text = response.json()[0].get("generated_text", "Sorry, I could not find an answer.")
    else:
        generated_text = f"Error: {response.status_code}, {response.text}"
    
    return generated_text

# Streamlit app
def main():
    st.title("Chat App with Document Support (Using Hugging Face)")

    st.write("Upload a PDF or DOCX file, then ask questions based on its summarized content.")
    
    # Upload PDF or DOCX
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=['pdf', 'docx'])
    
    if uploaded_file is not None:
        # Extract text based on file type
        if uploaded_file.name.endswith('.pdf'):
            document_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            document_text = extract_text_from_docx(uploaded_file)
        
        st.write("Document uploaded successfully.")
        
        # Summarize the document
        st.write("Summarizing the document...")
        summary = summarize_text_hf(document_text)
        st.write("Summary of the document:", summary)
        
        # Chat functionality
        st.write("Now you can ask questions related to the document summary.")
        
        question = st.text_input("Enter your question:")
        
        if question:
            # Answering the question based on the summary
            answer = generate_answer_hf(summary, question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()

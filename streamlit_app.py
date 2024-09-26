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

# Function to split large document into chunks to handle token limits
def split_text_into_chunks(text, max_chunk_size=512):
    words = text.split()
    chunks = [' '.join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size)]
    return chunks

# Function to interact with Hugging Face API for generative models (larger models)
def generate_answer_hf(text, question):
    api_key = st.secrets["huggingface_api_key"]  # Get Hugging Face API key securely
    model = "bigscience/bloom"  # You can replace this with a larger generative model if needed

    url = f"https://api-inference.huggingface.co/models/{model}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # Send the request in chunks if the document is large
    chunks = split_text_into_chunks(text)

    all_answers = []
    
    for chunk in chunks:
        prompt = f"Context: {chunk}\nQuestion: {question}\nAnswer:"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,  # You can adjust this to control the length of the answer
                "temperature": 0.7  # Increase for more diverse answers, decrease for more focused
            }
        }

        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            generated_text = response.json()[0].get("generated_text", "Sorry, I could not find an answer.")
            all_answers.append(generated_text)
        else:
            all_answers.append(f"Error: {response.status_code}, {response.text}")
    
    # Combine all chunk answers into one cohesive answer
    return ' '.join(all_answers)

# Streamlit app
def main():
    st.title("Chat App with Document Support (Using Hugging Face)")

    st.write("Upload a PDF or DOCX file, then ask questions based on its content.")
    
    # Upload PDF or DOCX
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=['pdf', 'docx'])
    
    if uploaded_file is not None:
        # Extract text based on file type
        if uploaded_file.name.endswith('.pdf'):
            document_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            document_text = extract_text_from_docx(uploaded_file)
        
        st.write("Document uploaded successfully.")
        
        # Chat functionality
        st.write("Now you can ask questions related to the document.")
        
        question = st.text_input("Enter your question:")
        
        if question:
            # Answering the question based on document
            answer = generate_answer_hf(document_text, question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()

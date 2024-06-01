import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Model and tokenizer loading
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# Function to preprocess PDF files
def file_preprocessing(files):
    final_texts = ""
    for file in files:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        for text in texts:
            final_texts += text.page_content
    return final_texts

# Function to generate summarization
def llm_pipeline(content):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500, 
        min_length=50
    )
    result = pipe_sum(content)
    result = result[0]['summary_text']
    return result

# Function to display the combined PDF and summarization
def display_combined_pdf_and_summary(files, summary):
    st.info("Uploaded Files")
    for file in files:
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    st.info("Summarization Complete")
    st.success("Combined Summary:")
    st.write(summary)

# Streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization")

    uploaded_files = st.file_uploader("Upload your PDF files", type=['pdf'], accept_multiple_files=True)

    if uploaded_files is not None:
        if st.button("Summarize"):
            file_paths = []
            for uploaded_file in uploaded_files:
                filepath = "data/"+uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                file_paths.append(filepath)
            
            # Process chunks and combine results
            combined_content = ""
            for file_path in file_paths:
                content = file_preprocessing([file_path])
                combined_content += content
            
            summary = llm_pipeline(combined_content)
            display_combined_pdf_and_summary(file_paths, summary)

if __name__ == "__main__":
    main()

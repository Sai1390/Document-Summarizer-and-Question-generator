# Document-Summarizer-and-Question-generator

# Overview
Navigating the extensive and complex content of PDF documents poses a significant challenge, particularly in contexts such as corporate reports, legal documents, and academic research. This Document Summarizer & Question Generator tool addresses these challenges by leveraging state-of-the-art natural language processing models to extract insights from PDF documents, enhancing the user's capacity to access and absorb information effectively.

Table of Contents
Introduction
Technologies Used
Features
Installation
Usage
Evaluation
Contributing
License
Contact
# Introduction
The Document Summarizer & Question Generator tool is designed to assist users in efficiently navigating the dense and occasionally complicated structure of PDF documents. This tool provides concise summaries and generates relevant questions based on the content of the documents, using advanced natural language processing algorithms to ensure that users understand the required content properly.

# Technologies Used
Natural Language Processing (NLP) Models

FLAN T5: Used for document summarization, finetuned on the DialogSum dataset.
GPT-3.5-turbo: Utilized for generating contextually relevant questions.
Metrics

Rouge Metrics: Used for evaluating the performance of the summarization and question generation models, including rouge-1, rouge-2, rouge-L, and rouge-Lsum.
# Features
Document Summarization: Provides concise summaries of PDF documents using the finetuned FLAN T5 model.
Question Generation: Produces relevant questions based on the content of the documents using GPT-3.5-turbo.
User-friendly Interface: Easy-to-use interface for uploading PDFs and viewing summaries and questions.

# Upload a PDF Document:
Use the provided interface to upload a PDF document that you want to summarize and generate questions from.

# View Summaries and Questions:
The application will display a concise summary of the PDF document and generate relevant questions based on the content.

# Evaluation
The FLAN T5 model demonstrates robust summarization capabilities when finetuned on the DialogSum dataset, achieving exceptional performance across various rouge metrics. The GPT-3.5-turbo model excels in generating contextually relevant questions, also evaluated through rouge scores.

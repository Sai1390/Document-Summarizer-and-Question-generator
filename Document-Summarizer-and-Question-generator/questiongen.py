import streamlit as st
import os
import json
import pandas as pd
import traceback
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import PyPDF2

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2

# Load environment variables
load_dotenv()

import os
from langchain_openai import ChatOpenAI

# Get the OpenAI API key from the environment variable
OPENAI_API_KEY = "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"

# Function to generate quiz and evaluate
def generate_and_evaluate_quiz(text, number, subject, tone):
    # get_openai_callback is so important to setup token usage tracking in LangChain
    TEXT = text
    NUMBER = number
    SUBJECT = subject
    TONE = tone
    RESPONSE_JSON = '{"1": {"question": "What is the main contribution of the paper?", "options": {"a": "Question answering methods", "b": "Neural network training", "c": "Data collection from CQA websites", "d": "Question generation approaches"}, "correct_answer": "Data collection from CQA websites"}, "2": {"question": "How many question generation approaches are proposed in the paper?", "options": {"a": "One", "b": "Two", "c": "Three", "d": "Four"}, "correct_answer": "Two"}, "3": {"question": "What is the main advantage of using CQA websites for training data collection?", "options": {"a": "Low quality data", "b": "High cost", "c": "Real user-generated content", "d": "Limited information"}, "correct_answer": "Real user-generated content"}, "4": {"question": "How is question topic selection performed in the generation-based method?", "options": {"a": "Based on pre-defined rules", "b": "Using entity recognition", "c": "Attention mechanism", "d": "Random selection"}, "correct_answer": "Attention mechanism"}, "5": {"question": "What does the improvement on MS MARCO dataset indicate?", "options": {"a": "Better question generation performance", "b": "Quality of questions from Bing search log", "c": "Crowdsourced data", "d": "No impact on QA systems"}, "correct_answer": "Quality of questions from Bing search log"}}'
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.5)

    PROMPT_TEMPLATE = """
    Text: {text}
    You are an expert MCQ maker. Given the above text, it is your job to 
    create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide.
    Ensure to make {number} MCQs
    ### RESPONSE_JSON
    {response_json}
    """
    
    quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=PROMPT_TEMPLATE
    )
    quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)
    
    EVALUATION_PROMPT_TEMPLATE = """
    You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students.
    You need to evaluate the complexity of the questions and give a complete analysis of the quiz. Only use at max 50 words for complexity if the quiz is not at per with the cognitive and analytical abilities of the students,
    update the quiz questions which needs to be change the tone such that it perfectly fits the students ability.
    Quiz_MCQs:
    {quiz}

    Check from an expert English Writer of the above quiz.
    """
    
    quiz_evaluation_prompt = PromptTemplate(
        input_variables=["subject", "quiz"],
        template=EVALUATION_PROMPT_TEMPLATE
    )

    # Connect the both chain to get the actual output using Sequential Chain
    quiz_evaluation_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

    generate_evaluate_chains = SequentialChain(chains=[quiz_chain, quiz_evaluation_chain], input_variables=["text", "number", "subject", "tone", "response_json"], output_variables=["quiz", "review"], verbose=True)
    
    with get_openai_callback() as cb:
        response = generate_evaluate_chains(
            {
            "text": TEXT, 
            "number": NUMBER,
            "subject": SUBJECT,
            "tone": TONE,
            "response_json": json.dumps(RESPONSE_JSON)
            }
        )
    quiz_response = response['quiz']
    quiz_dict = json.loads(quiz_response)

    return quiz_dict

# Main function for Streamlit app
def main():
    st.title("MCQ Generator and Evaluator")

    # File uploader for uploading PDF or text file
    uploaded_file = st.file_uploader("Upload PDF or text file", type=["pdf", "txt"])

    if uploaded_file is not None:
        # Read text from the uploaded file
        if uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")  # Read text file
        elif uploaded_file.type == "application/pdf":
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Read text from the PDF file
            text = ""
            with open("uploaded_file.pdf", "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
        else:
            st.error("Unsupported file format. Please upload a PDF or text file.")
            return

        # Input fields
        number = st.number_input("Number of MCQs", min_value=1, value=5)
        subject = st.text_input("Subject", "English")
        tone = st.text_input("Tone", "Simple")

        # Button to generate quiz and evaluate
        if st.button("Generate and Evaluate Quiz"):
            # Generate and evaluate quiz
            try:
                quiz = generate_and_evaluate_quiz(text, number, subject, tone)
                st.success("Quiz generated and evaluated successfully!")
 
                # Display quiz and review
                st.subheader("Generated Quiz")
                for question_num, details in quiz.items():
                    st.write(f"Question {question_num}: {details['question']}")
                    st.write("Options:")
                    for option, text in details['options'].items():
                        st.write(f"{option}: {text}")
                    st.write(f"Correct Answer: {details['correct_answer']}")
                    st.write("")  # Add a blank line for separation

                    # st.subheader("Review")
                    # st.write(review)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.write(traceback.format_exc())

if __name__ == "__main__":
    main()

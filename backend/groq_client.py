# the limits for llama3-70b-8192 are 30 requests per minute & 6,000 tokens per minute.	Also max 14,400 requests per day.
import base64
import mimetypes
import os
from groq import Groq



class GroqClient:

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def extract_text_from_file(self, notes_path):
        # extracts text from file
        with open(notes_path, 'r') as file:
            return file.read()

    def generate_initial_questions(self, notes_path):
        notes = self.extract_text_from_file(notes_path)
        initial_questions = []

        completion = self.client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Take these notes and generate 4 questions based off of them. Make 75% of the questions open ended and make 25% of the questions closed ended. Try to make the questions mainly based off notes. No Multiple choice. List questions 1 to 20 with no headers and after the end of each question add '*'. Notes: " + notes
                }
            ],
            temperature=1,
            max_tokens=6000,
            top_p=1,
            stream=True,
            stop=None,
        )

        questions_string = ''
        for chunk in completion:

            # Get the content from the chunk
            token = chunk.choices[0].delta.content

            if token is None:
                continue

            questions_string += token
            questions_string = questions_string.strip() # removes leading/trailing whitespace and newline characters

        initial_questions = questions_string.split("*") # seperates questions into array elements

        # print("\n-----------------------------\n")
        # for question in initial_questions:
        #     print(question)
        #
        # print(f"Q: {initial_questions[2]} |")

        return initial_questions

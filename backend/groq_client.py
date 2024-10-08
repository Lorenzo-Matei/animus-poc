# the limits for llama3-70b-8192 are 30 requests per minute & 6,000 tokens per minute.	Also max 14,400 requests per day.
import base64
import mimetypes
import os
from groq import Groq



class GroqClient:

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.conversation_history = []  # keeps context of conversation
        # FYI the issue with stacking context is that llama has a cap of 6000 tokens
        # per request. The longer the responses become the greater a chance it will start cutitng off context



    def extract_text_from_file(self, notes_path):
        # extracts text from file
        with open(notes_path, 'r') as file:
            return file.read()

    def generate_initial_questions(self, notes_path):
        notes = self.extract_text_from_file(notes_path)

        prompt = {
                    "role": "user",
                    "content": "Take these notes and generate 4 questions based off of them. Make 75% of the questions open ended and make 25% of the questions closed ended. Only list the questions and dont add your own descriptions of the type of question. Try to make the questions mainly based off notes. No Multiple choice. List questions 1 to 4 with no headers and after the end of each question add '*'. Notes: " + notes
                }
        self.conversation_history.append(prompt)
        response = self.send_prompt()
        initial_questions = self.extract_questions(response)

        return initial_questions

    def extract_questions(self, response):
        questions_string = ''
        for chunk in response:

            # Get the content from the chunk
            token = chunk.choices[0].delta.content

            if token is None:
                continue

            questions_string += token
            questions_string = questions_string.strip() # removes leading/trailing whitespace and newline characters
        initial_questions = questions_string.split("*") # seperates questions into array elements
        initial_questions.pop(-1)  # removes empty string last element ''

        return initial_questions

    def evaluate_questions(self, test):
        prompt = {
            "role": "user",
            "content": "evaluate the answers to the questions asked previously and assign a score of 0 to 100. Format your response with: 'question #:', 'score: ', and 'improve answer' then add a '*' at the end of each 'improve answer'. Make sure to keep the 'improve answer' short and comprehensive. Here is every question and answer in a dictionary corresponding to each index" + str(test)
        }

        self.conversation_history.append(prompt)
        response = self.send_prompt()

        return response


    def split_response(self, response):
        response_string = ''

        for chunk in response:
            token = chunk.choices[0].delta.content

            if token is None:
                continue

            response_string += token
            response_string = response_string.strip()

        response_line = response_string.split('*') # creates array questions with evaluations
        # response_line.pop(-1)

        return response_line


    def print_response(self, questions):
        for q in questions:
            print(q)



    def send_prompt(self, prompt=None):
        completion = self.client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=self.conversation_history,
            temperature=1,
            max_tokens=6000,
            top_p=1,
            stream=True,
            stop=None,
        )

        return completion


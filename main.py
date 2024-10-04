import datetime
import os
from tokenizers import Tokenizer
import torch
import numpy
from backend.groq_client import GroqClient


def get_notes_path():
    study_notes_path = "" # /Users/macbook/Documents/sql_vs_nosql.txt

    while os.path.exists(study_notes_path) == False:
        if study_notes_path.lower() == "quit":
            return

        print("please input the path to your study notes or 'quit' to exit:")
        study_notes_path = "/Users/macbook/Documents/sql_vs_nosql.txt"

    return study_notes_path


def main():
    groq_client = GroqClient()


    notes_path = get_notes_path()
    # count_file_tokens(notes_path) # needs key to authenticate

    # feeds notes into LLama to generate questions
    groq_client.generate_initial_questions(notes_path)


if __name__ == '__main__':
    main()

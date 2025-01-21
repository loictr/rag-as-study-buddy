import json
import random

QUESTIONS_PATH = 'questions/questions_for_demo.json'

class QuestionsRepository:

    current_index = 0

    def __init__(self):
        self.questions_all = []
        with open(QUESTIONS_PATH, 'r') as f:
            self.questions_all = json.loads( '\n'.join([q for q in f if not q.strip().startswith('//')]))

    def get_question(self):
        question = self.questions_all[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.questions_all)
        return question
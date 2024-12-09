import json
import random
from common import QUESTIONS_PATH

class QuestionsRepository:

    def __init__(self):
        self.questions_all = []
        with open(QUESTIONS_PATH, 'r') as f:
            self.questions_all = json.load(f)

    def get_question(self):
        return random.choice(self.questions_all)
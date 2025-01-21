from answer_analyst_stepped import AnswerAnalystStepped
import unittest

class AnswerAnalystSteppedTest(unittest.TestCase):

    def test_evaluate_answer(self):
        analyst = AnswerAnalystStepped()
        
        # Test basic evaluation
        question = "What is Python?"
        user_answer = "Python is a programming language"
        result = analyst.evaluate_answer(question, user_answer)
        assert isinstance(result, str)
        assert len(result) > 0

        # # Test with empty answer
        # question = "What is Python?"
        # user_answer = ""
        # result = analyst.evaluate_answer(question, user_answer)
        # assert isinstance(result, str)
        # assert len(result) > 0

        # # Test with empty question
        # question = ""
        # user_answer = "Some answer"
        # result = analyst.evaluate_answer(question, user_answer)
        # assert isinstance(result, str)
        # assert len(result) > 0

        # # Test with special characters
        # question = "What is Python?!@#$%^&*()"
        # user_answer = "Python is a programming language!@#$%^&*()"
        # result = analyst.evaluate_answer(question, user_answer)
        # assert isinstance(result, str)
        # assert len(result) > 0

    def test_evaluator_initialization(self):
        analyst = AnswerAnalystStepped()
        assert analyst._chain_full is not None

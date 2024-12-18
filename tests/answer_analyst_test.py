from answer_analyst import AnswerAnalyst
import unittest

class AnswerAnalystTest(unittest.TestCase):

    def test_evaluate_answer():
        evaluator = AnswerAnalyst()
        
        # Test basic evaluation
        question = "What is Python?"
        user_answer = "Python is a programming language"
        result = evaluator.evaluate_answer(question, user_answer)
        assert isinstance(result, str)
        assert len(result) > 0

        # Test with empty answer
        question = "What is Python?"
        user_answer = ""
        result = evaluator.evaluate_answer(question, user_answer)
        assert isinstance(result, str)
        assert len(result) > 0

        # Test with empty question
        question = ""
        user_answer = "Some answer"
        result = evaluator.evaluate_answer(question, user_answer)
        assert isinstance(result, str)
        assert len(result) > 0

        # Test with special characters
        question = "What is Python?!@#$%^&*()"
        user_answer = "Python is a programming language!@#$%^&*()"
        result = evaluator.evaluate_answer(question, user_answer)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_evaluator_initialization():
        evaluator = AnswerAnalyst()
        assert evaluator._chain is not None
        assert evaluator._prompt_template is not None
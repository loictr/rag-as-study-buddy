from pprint import pprint
from ragas import EvaluationDataset, SingleTurnSample, evaluate
import json
import logging
import os

# Local imports
from common import EMBEDDING_MODEL, LLM_MODEL 
from answer_analyst import AnswerAnalyst
import argparse


class RagasLauncher:

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def launch(self, question_id_filter=None, case_id_filter=None):

        if question_id_filter is None and case_id_filter is not None:
            raise ValueError("Cannot filter by case_id without filtering by question_id")


        rubric = {
            "accuracy": "Correct",
            "completeness": "High",
            "fluency": "Excellent"
        }

        with open('rag_evaluation_dataset.json', 'r') as f:
            dataset = json.load(f)

        samples = []
        subject = AnswerAnalyst()

        for question in dataset:
            question_id = question['id']

            if(question_id_filter is not None and question_id != question_id_filter):
                continue

            rag_question = question['rag_question']
            context_reference = question['context_reference']

            for case in question['cases']:
                case_id = case['id']

                if(case_id_filter is not None and case_id != case_id_filter):
                    continue

                self._logger.info(f"Processing question {question_id} - case {case_id}...")

                user_answer = case['user_answer']
                rag_feedback_reference = case['rag_feedback_reference']

                rag_output = subject.evaluate_answer_full_ouput(rag_question, user_answer)
                rag_feedback_actual = rag_output['answer']
                context_actual = rag_output["docs"]
                context_actual_contents = [doc.page_content for doc in context_actual]

                #pprint(rag_output)
                self._logger.info(f"Question: {rag_question}")
                self._logger.info(f"Reference feedback: {rag_feedback_reference}")
                self._logger.info(f"Actual feedback: {rag_feedback_actual}")

                sample = SingleTurnSample(
                    user_input=user_answer,

                    response=rag_feedback_actual,
                    reference=rag_feedback_reference,

                    retrieved_contexts=context_actual_contents,
                    reference_contexts=context_reference,
                    
                    rubric=rubric
                )
                samples.append(sample)

        self._logger.info(f"Processing questions done.")

        eval_dataset = EvaluationDataset(samples=samples)

        results = evaluate(dataset=eval_dataset)#, metrics=metrics)
        print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question-id', type=str, help='Specific question ID to evaluate')
    parser.add_argument('--case-id', type=str, help='Specific case ID to evaluate')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    question_id = args.question_id
    case_id = args.case_id

    launcher = RagasLauncher()
    launcher.launch(question_id, case_id)
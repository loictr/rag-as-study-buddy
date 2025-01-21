from abc import abstractmethod
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt
from ragas.metrics.base import MetricType
from dataclasses import dataclass, field

from pprint import pprint
from typing import Any, Dict, List
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Metric, MetricWithLLM, FactualCorrectness, SingleTurnMetric, AspectCritic
import json
import logging
import typing as t
import os

# Local imports
from src.analyst.answer_analyst_stepped import AnswerAnalystStepped, Notation
import argparse
import re
from threading import Lock


parsed_responses: Dict[str, Dict[str, Any]] = {}


class CommonMetricInput(BaseModel):
    rag_feedback_actual: str = Field(description="actual feedback from AI")

class CommonMetricOutput(BaseModel):
    score: float = Field(description="score of similarity of the given criteria in the feedback")


class CriteriaMetricBase(MetricWithLLM, SingleTurnMetric):
    def __init__(self, name: str):
        super().__init__()
        self._field_name = name
        self._logger = logging.getLogger(__name__)

    async def _ascore(self, row):
        pass

    def _get_value(self, response: str) -> Any:
        # Use a thread-safe dictionary to store parsed responses
        if not hasattr(self, '_lock'):
            self._lock = Lock()
        
        with self._lock:
            if response not in parsed_responses:
                parsed_response = json.loads(response)  # super happy to parse something i serialized
                parsed_responses[response] = parsed_response
            else:
                parsed_response = parsed_responses[response]

        return parsed_response[self._field_name]

    @abstractmethod
    def _compare_values(self, actual_value: Any, reference_value: Any) -> float | None:
        """Compare actual and reference values and return score between 0 and 1"""
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        actual_value = self._get_value(sample.response)
        reference_value = self._get_value(sample.reference)

        score = self._compare_values(actual_value, reference_value)
        self._logger.info("comparing field %s ref=%s actual=%s -> %s", self._field_name, reference_value, actual_value, score)
        return score

class FloatCriteriaMetricBase(CriteriaMetricBase):
    def _compare_values(self, actual_value: float, reference_value: float) -> float | None:
        if actual_value is None or actual_value < 0 or actual_value > 1:
            return 0.0
        return 1.0 - abs(reference_value - actual_value)
        
        


class AccuracyMetric(FloatCriteriaMetricBase):
    def __init__(self):
        FloatCriteriaMetricBase.__init__(self, "accuracy")
        self.name = "Accuracy"

class CompletenessMetric(FloatCriteriaMetricBase):
    def __init__(self):
        FloatCriteriaMetricBase.__init__(self, "completeness")
        self.name = "Completeness"

class ClarityMetric(FloatCriteriaMetricBase):
    def __init__(self):
        FloatCriteriaMetricBase.__init__(self, "clarity")
        self.name = "Clarity"

class RelevanceMetric(FloatCriteriaMetricBase):
    def __init__(self):
        FloatCriteriaMetricBase.__init__(self, "relevance")
        self.name = "Relevance"

class SectionsMetric(CriteriaMetricBase):
    def __init__(self):
        CriteriaMetricBase.__init__(self, "sections")
        self.name = "Sections"

    def _compare_values(self, actual_value: List[str], reference_value: List[str]) -> float | None:
        
        if actual_value is None:
            actual_value = []
        if reference_value is None:
            reference_value = []


        if len(actual_value) == 0 and len(reference_value) == 0:
            return 1.0
        if len(actual_value) != len(reference_value):
            return 0.0

        # Convert accented characters to non-accented characters
        actual_value = [self.clean_section_name(item) for item in actual_value]
        reference_value = [self.clean_section_name(item) for item in reference_value]

        actual_value = set(actual_value)
        reference_value = set(reference_value)

        if len(actual_value.intersection(reference_value)) == len(reference_value):
            return 1.0

        return 0.0


    def clean_section_name(self, item:str) -> str:
        cleaned = re.sub(r'[àáâãäçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ]', 
            lambda x: 'aaaaaceeeeiiiinooooouuuuyyAAAAACEEEEIIIINOOOOOUUUUY'['àáâãäçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ'.index(x.group())], 
            item)
        cleaned = re.sub(r'[^a-zA-Z0-9 ]', ' ', cleaned)
        cleaned = ' '.join(cleaned.split())
        return cleaned




class RagasLauncher:

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def launch(self, batch_id_filter = None, question_id_filter = None, case_id_filter = None):

        if question_id_filter is not None and batch_id_filter is None:
            raise ValueError("Cannot filter by question_id without filtering by batch_id")
        if question_id_filter is None and case_id_filter is not None:
            raise ValueError("Cannot filter by case_id without filtering by question_id")


        current_dir = Path(__file__).parent
        with open(current_dir / 'rag_evaluation_dataset.json', 'r') as f:
            dataset = json.load(f)

        samples = []
        subject = AnswerAnalystStepped()

        metrics = [
            SectionsMetric(),
            AccuracyMetric(),
            RelevanceMetric(),
            CompletenessMetric(),
            ClarityMetric(),
#             AspectCritic(
#                 name="Format",
#                 definition="The format of the feedback should strictly be like the following \
# \
# The sections list is optional.\
# No other comment, or text should be included before or after.\
# \
# Did the model respect all this constraints?\
# \
# \
# ----\
# Format:\
#     \
# accuracy: <score>, \
# relevance: <score>, \
# completeness: <score>, \
# clarity: <score>, \
# sections: \
#     - <section - subsection>\
#     - <section - subsection>\
#     - ... \
# ",
#            ),
        ]


        reports = []

        for batch in dataset['batches']:
            batch_id = batch['id']
            if(batch_id_filter is not None and batch_id != batch_id_filter):
                continue

            for question in batch['questions']:
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
                    accuracy_reference = case['accuracy_reference']
                    completeness_reference = case['completeness_reference']
                    relevance_reference = case['relevance_reference']
                    clarity_reference = case['clarity_reference']
                    sections_reference = case['sections_reference']


                    rag_output = subject.notation_full_output(rag_question, user_answer)

                    rag_feedback_actual = rag_output['notation']
                    context_actual = rag_output["docs"]
                    context_actual_contents = [doc.page_content for doc in context_actual]

                    #pprint(rag_output)
                    self._logger.info(f"System: {rag_question}")
                    self._logger.info(f"User: {user_answer}")
                    self._logger.info(f"Actual feedback: {rag_feedback_actual}")

                    if rag_feedback_actual is not str:
                        rag_feedback_actual_str = json.dumps(rag_feedback_actual, indent=4)
                    else:
                        rag_feedback_actual_str = rag_feedback_actual

                    rag_feedback_reference = {
                        'accuracy': case['accuracy_reference'],
                        'completeness': case['completeness_reference'],
                        'relevance': case['relevance_reference'],
                        'clarity': case['clarity_reference'],
                        'sections': case['sections_reference'],
                    }
                    rag_feedback_reference_str = json.dumps(rag_feedback_reference, indent=4)

                    sample = SingleTurnSample(
                        user_input=user_answer,

                        response=rag_feedback_actual_str,
                        reference=rag_feedback_reference_str,

                        retrieved_contexts=context_actual_contents,
                        reference_contexts=context_reference,
                    )

                    eval_dataset = EvaluationDataset(samples=[sample])

                    results = evaluate(dataset=eval_dataset, metrics=metrics, show_progress=False)
                    
                    report = {
                        "batch_id": batch_id,
                        "question_id": question_id,
                        "case_id": case_id,
                        "inputs.rag_question": rag_question,
                        "inputs.user_answer": user_answer,
                        #"reference.rag_feedback_reference": rag_feedback_reference,
                        "reference.accuracy": accuracy_reference,
                        "reference.completeness": completeness_reference,
                        "reference.relevance": relevance_reference,
                        "reference.clarity": clarity_reference,
                        "reference.sections": sections_reference,

                        "outputs.full": rag_feedback_actual,
                    }

                    if relevance_reference > 0: # ignore if irrelevant
                        report["accuracy_score_comparison"] = results['Accuracy'][0]
                        report["completeness_score_comparison"] = results['Completeness'][0]
                    report["relevance_score_comparison"] = results['Relevance'][0]
                    report["sections_score_comparison"] = results['Sections'][0]

                    reports.append(report)
                    
        with open(current_dir / 'rag_evaluation_ragas_results.json', 'w', encoding='utf-8') as f:
            json.dump(reports, f, ensure_ascii=False, indent=4, default=str)


        self._logger.info(f"Processing questions done.")


def run_evaluate():
    launcher = RagasLauncher()
    launcher.launch()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-id', type=str, help='Specific batch ID to evaluate')
    parser.add_argument('--question-id', type=str, help='Specific question ID to evaluate')
    parser.add_argument('--case-id', type=str, help='Specific case ID to evaluate')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().setLevel(logging.WARNING)  # Set root logger to WARNING
    logging.getLogger(__name__).setLevel(logging.INFO)  # Only show INFO for this module

    batch_id = args.batch_id
    question_id = args.question_id
    case_id = args.case_id

    launcher = RagasLauncher()
    launcher.launch(batch_id, question_id, case_id)
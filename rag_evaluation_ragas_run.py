from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt
from ragas.metrics.base import MetricType
from dataclasses import dataclass, field

from pprint import pprint
from typing import Any, Dict, List
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Metric, MetricWithLLM, FactualCorrectness, SingleTurnMetric
import json
import logging
import typing as t
import os

# Local imports
from common import EMBEDDING_MODEL, LLM_MODEL 
from answer_analyst import AnswerAnalyst
import argparse



class SectionsInput(BaseModel):
    #rag_question: str = Field(description="the user request")
    #user_answer: str = Field(description="response from user")
    rag_feedback_reference: str = Field(description="expected feedback from AI")
    rag_feedback_actual: str = Field(description="actual feedback from AI")


class SectionsOutput(BaseModel):
    score: float = Field(description="score of similarity of suggested sections")


class SectionsPrompt(PydanticPrompt[SectionsInput, SectionsOutput]):
    instruction = """\
        The user was asked a question and provided an answer. Then the system provided an evaluation of the answer as a feedback. Its feedback may include sections to review in the documentation. \
        Given the the expected feedback, determine if the system should send sections to review. If so, given the actual feedback, determine id it actually included sections references to its feedback, and determine if the sections are correct.\
        If no sections were expected, then the system should not include any sections in its feedback.\

        Do not judge anything but the sections to review.\

        score is a float between 0 and 1.\
        if the expecteded feedback includes sections to review and the actual feedback includes the sames sections, then score=1.\
        if the expecteded feedback includes sections to review and the actual feedback includes other sections, then score=0.\
        if the expecteded feedback does not include any section to review and the actual feedback does not include any sections, then score=1.\
        if the system included the sections to review in the actual although no section was expected, then score=0.\
        if the system did not include the sections to review in the actual although sections was expected, or if the referenced sections are incorrect in the actual feedback, then score=0.\
        
        Output score
        """
    input_model = SectionsInput
    output_model = SectionsOutput


class SectionsMetric(MetricWithLLM, SingleTurnMetric):
    def __init__(self):
        super().__init__()
        self.name = "Sections"
        # self._required_columns: t.Dict[MetricType, t.Set[str]] = field(
        #     default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
        # )
        self.prompt: PydanticPrompt = SectionsPrompt()

    async def _ascore(self, row):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = SectionsInput(
            user_answer=sample.user_input,
            rag_feedback_reference=sample.reference,
            rag_feedback_actual=sample.response,
        )
        prompt_response = await self.prompt.generate(
            data=prompt_input, llm=self.llm
        )
        return prompt_response.score




class AccuracyInput(BaseModel):
    #rag_question: str = Field(description="the user request")
    #user_answer: str = Field(description="response from user")
    rag_feedback_reference: str = Field(description="expected feedback from AI")
    rag_feedback_actual: str = Field(description="actual feedback from AI")


class AccuracyOutput(BaseModel):
    score: float = Field(description="score of similarity of judged accuracy")


class AccuracyPrompt(PydanticPrompt[AccuracyInput, AccuracyOutput]):
    instruction = """\

        The user was asked a question and provided an answer. Then the system provided an evaluation of the answer as a feedback. \
        Given the the expected feedback and the actual feedback, determine if the system correctly evaluated the accuracy of the user's answer and give a score.\


        Do not judge the completeness of the answer, only the accuracy. If the answer is accurate, then the accuracy is correct.\

        score is a float between 0 and 1.\
        
        case 1:
        The expected feedback stated that the answer is excellent,
        And the actual feedback stated that the answer is excellent,
        Then the score=1.\

        case 2:
        The expected feedback stated that the answer is excellent,
        And the actual feedback stated that the answer is good,
        Then the score=0.9.\

        case 3:
        The expected feedback stated that the answer is correct,
        And the actual feedback stated that the answer is bad,
        Then the score=0 .\

        case 4:
        The expected feedback stated that the answer is not correct,
        And the actual feedback stated that the answer is correct,
        Then the score=0 .\

        case 5:
        The expected feedback stated that the answer is not correct,
        And the actual feedback stated that the answer is not correct,
        Then the score=1 .\

        case 6:
        The expected feedback stated that the answer is correct but incomplete,
        And the actual feedback stated that the answer is correct,
        Then the score=1 .\

        case 7:
        The expected feedback stated that the answer is almost correct,
        And the actual feedback stated that the answer is almost correct,
        Then the score=1 .\

        case 8:
        The expected feedback stated that the answer is almost correct,
        And the actual feedback stated that the answer is correct,
        Then the score=0.7 .\


        Output score\
        """

    input_model = AccuracyInput
    output_model = AccuracyOutput


class AccuracyMetric(MetricWithLLM, SingleTurnMetric):
    def __init__(self):
        super().__init__()
        self.name = "Accuracy"
        # self._required_columns: t.Dict[MetricType, t.Set[str]] = field(
        #     default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
        # )
        self.prompt: PydanticPrompt = AccuracyPrompt()

    async def _ascore(self, row):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = AccuracyInput(
            user_answer=sample.user_input,
            rag_feedback_reference=sample.reference,
            rag_feedback_actual=sample.response,
        )
        prompt_response = await self.prompt.generate(
            data=prompt_input, llm=self.llm
        )
        return prompt_response.score






class NaturalInput(BaseModel):
    #rag_question: str = Field(description="the user request")
    #user_answer: str = Field(description="response from user")
    rag_feedback_reference: str = Field(description="expected feedback from AI")
    rag_feedback_actual: str = Field(description="actual feedback from AI")


class NaturalOutput(BaseModel):
    score: float = Field(description="score of similarity of judged accuracy")


class NaturalPrompt(PydanticPrompt[NaturalInput, NaturalOutput]):
    instruction = """\
        The user was asked a question and provided an answer. Then the system provided an evaluation of the answer as a feedback. \
        Given the actual feedback, determine if the system provided a natural conversational and friendly response. The user should not have the sentiment he is talking to a machine. \
        Do not judge anything but that.\

        A feeback like the following example is not natural, because it shows the mechanical process for building the feedback:
        "
        1. Accuracy statement: That is correct!
        2. Completeness feedback (not needed in this case)
        3. Correct answer (not needed in this case)
        4. Section reference: Section: Windriders, Sub-section: Diet and Habitat
        5. Brief encouragement: Great job identifying their preferred habitats! Keep studying the fascinating Zekryn species.
        "
        Instead, the feedback should be more natural, like the following example:
        "That is correct! for more details you can review: Windriders - Diet and Habitat. Great job identifying their preferred habitats! Keep studying the fascinating Zekryn species."

        Output a score as a float between 0 and 1 that reflects your judgement.\
        """

    input_model = NaturalInput
    output_model = NaturalOutput


class NaturalMetric(MetricWithLLM, SingleTurnMetric):
    def __init__(self):
        super().__init__()
        self.name = "Natural"
        # self._required_columns: t.Dict[MetricType, t.Set[str]] = field(
        #     default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
        # )
        self.prompt: PydanticPrompt = NaturalPrompt()

    async def _ascore(self, row):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = NaturalInput(
            user_answer=sample.user_input,
            rag_feedback_reference=sample.reference,
            rag_feedback_actual=sample.response,
        )
        prompt_response = await self.prompt.generate(
            data=prompt_input, llm=self.llm
        )
        return prompt_response.score





class CompletenessInput(BaseModel):
    #rag_question: str = Field(description="the user request")
    #user_answer: str = Field(description="response from user")
    rag_feedback_reference: str = Field(description="expected feedback from AI")
    rag_feedback_actual: str = Field(description="actual feedback from AI")


class CompletenessOutput(BaseModel):
    score: float = Field(description="score of similarity of judged complete")


class CompletenessPrompt(PydanticPrompt[CompletenessInput, CompletenessOutput]):
    instruction = """\
        The user was asked a question and provided an answer. Then the system provided an evaluation of the answer as a feedback. \
        Given the the expected feedback and the actual feedback, determine if the system correctly evaluated the completeness of the user's answer and give a score between 0 and 1.\

        score is a float between 0 and 1.

        case 1:
        The expected feedback does not contain any remark about the completeness of the answer.
        And the actual feedback does not contain any remark about the completeness of the answer.
        Then the score=1.\

        case 2:
        The expected feedback includes some remark about the completeness of the answer.
        And the actual feedback includes remarks about the completeness of the answer.
        And the actual feedback about the completeness matches the expected feedback.
        Then the score=1.\

        case 2.1:
        The expected feedback includes some remark about the completeness of the answer.
        And the actual feedback includes remarks about the completeness of the answer.
        But the actual feedback about the completeness does not matches the expected feedback.
        Then the score=0.2.\

        case 3:
        The expected feedback includes some remark about the completeness of the answer.
        And the actual feedback does not include any remark about the completeness of the answer.
        Then the score=0.\

        case 4:
        The expected feedback does not include any remark about the completeness of the answer.
        And the actual feedback includes some remark about the completeness of the answer.
        Then the score=0.\


        Do not judge the accuracy of the answer, only the completeness.\

        Output score \
        """


    input_model = CompletenessInput
    output_model = CompletenessOutput


class CompletenessMetric(MetricWithLLM, SingleTurnMetric):
    def __init__(self):
        super().__init__()
        self.name = "Completeness"
        # self._required_columns: t.Dict[MetricType, t.Set[str]] = field(
        #     default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
        # )
        self.prompt: PydanticPrompt = CompletenessPrompt()

    async def _ascore(self, row):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = CompletenessInput(
            user_answer=sample.user_input,
            rag_feedback_reference=sample.reference,
            rag_feedback_actual=sample.response,
        )
        prompt_response = await self.prompt.generate(
            data=prompt_input, llm=self.llm
        )
        return prompt_response.score





class RagasLauncher:

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def launch(self, batch_id_filter = None, question_id_filter = None, case_id_filter = None):

        if question_id_filter is not None and batch_id_filter is None:
            raise ValueError("Cannot filter by question_id without filtering by batch_id")
        if question_id_filter is None and case_id_filter is not None:
            raise ValueError("Cannot filter by case_id without filtering by question_id")



        with open('rag_evaluation_dataset.json', 'r') as f:
            dataset = json.load(f)

        samples = []
        subject = AnswerAnalyst()

        metrics = [
            AccuracyMetric(),
            SectionsMetric(),
            CompletenessMetric(),
            NaturalMetric(),
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
                    rag_feedback_reference = case['rag_feedback_reference']

                    rag_output = subject.evaluate_answer_full_ouput(rag_question, user_answer)
                    rag_feedback_actual = rag_output['answer']
                    context_actual = rag_output["docs"]
                    context_actual_contents = [doc.page_content for doc in context_actual]

                    #pprint(rag_output)
                    self._logger.info(f"System: {rag_question}")
                    self._logger.info(f"User: {user_answer}")
                    self._logger.info(f"Reference feedback: {rag_feedback_reference}")
                    self._logger.info(f"Actual feedback: {rag_feedback_actual}")

                    sample = SingleTurnSample(
                        user_input=user_answer,

                        response=rag_feedback_actual,
                        reference=rag_feedback_reference,

                        retrieved_contexts=context_actual_contents,
                        reference_contexts=context_reference,
                    )

                    eval_dataset = EvaluationDataset(samples=[sample])

                    results = evaluate(dataset=eval_dataset, metrics=metrics)
                    
                    report = {
                        "inputs.rag_question": rag_question,
                        "inputs.user_answer": user_answer,
                        "reference.rag_feedback_reference": rag_feedback_reference,
                        "outputs.rag_feedback_actual": rag_feedback_actual,
                        "feedback.accuracy": results['Accuracy'][0],
                        "feedback.completeness": results['Completeness'][0],
                        "feedback.sections": results['Sections'][0],
                        "feedback.natural": results['Natural'][0],
                    }
                    reports.append(report)
                    
        with open('rag_evaluation_ragas_results.json', 'w', encoding='utf-8') as f:
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
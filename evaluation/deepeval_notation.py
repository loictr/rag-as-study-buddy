import argparse
import json
import logging
from pathlib import Path


class DeepEvalLauncher:
    def launch(self, batch_id_filter = None, question_id_filter = None, case_id_filter = None):
        
        
        current_dir = Path(__file__).parent
        with open(current_dir / 'rag_evaluation_dataset.json', 'r') as f:
            dataset = json.load(f)



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
                    rag_feedback_reference = f"accuracy: {case['accuracy_reference']},\ncompleteness: {case['completeness_reference']},\nrelevance: {case['relevance_reference']},\nclarity: {case['clarity_reference']},\nsections:\n" + "".join([ f"\n  - {section}" for section in case['sections_reference']])
                    accuracy_reference = case['accuracy_reference']
                    completeness_reference = case['completeness_reference']
                    relevance_reference = case['relevance_reference']
                    clarity_reference = case['clarity_reference']
                    sections_reference = case['sections_reference']


                    rag_output = subject.notation_full_output(rag_question, user_answer)

                    pprint(rag_output)

                    rag_feedback_actual = rag_output['notation']
                    context_actual = rag_output["docs"]
                    context_actual_contents = [doc.page_content for doc in context_actual]

                    #pprint(rag_output)
                    self._logger.info(f"System: {rag_question}")
                    self._logger.info(f"User: {user_answer}")
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
                        "reference.accuracy": accuracy_reference,
                        "reference.completeness": completeness_reference,
                        "reference.relevance": relevance_reference,
                        "reference.clarity": clarity_reference,
                        "reference.sections": sections_reference,

                        "outputs.full": rag_feedback_actual,
                        
                        # "feedback.format": results['Accuracy'][0],
                        "accuracy_score_comparison": results['Accuracy'][0],
                        # "completeness_score_comparison": results['Completeness'][0],
                        "sections_score_comparison": results['Sections'][0],
                    }
                    reports.append(report)
                    
        with open(current_dir / 'rag_evaluation_ragas_results.json', 'w', encoding='utf-8') as f:
            json.dump(reports, f, ensure_ascii=False, indent=4, default=str)


        self._logger.info(f"Processing questions done.")



def run_evaluate():
    launcher = DeepEvalLauncher()
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

    launcher = DeepEvalLauncher()
    launcher.launch(batch_id, question_id, case_id)
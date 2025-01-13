import json
from langsmith import EvaluationResult, evaluate, traceable, wrappers, Client
from langsmith.schemas import Run, Example, Dataset
from openai import OpenAI
# Assumes you've installed pydantic
from pydantic import BaseModel

from answer_analyst import AnswerAnalyst

# Optionally wrap the OpenAI client to trace all model calls.
oai_client = wrappers.wrap_openai(OpenAI())


def create_dataset(ls_client: Client, batch_id_filter=None, question_id_filter=None, case_id_filter=None) -> Dataset:
    
    dataset_name = "rag-as-a-study-buddy"
    if ls_client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset {dataset_name} already exists. Returning existing dataset.")
        return dataset_name

    with open('rag_evaluation_dataset.json', 'r') as f:
        dataset = json.load(f)

    inputs = []
    outputs = []
    metadatas = []
    for batch in dataset['batches']:
        batch_id = batch['id']
        # if(batch_id_filter is not None and batch_id != batch_id_filter):
        #     continue

        for question in batch['questions']:
            question_id = question['id']

            # if(question_id_filter is not None and question_id != question_id_filter):
            #     continue

            rag_question = question['rag_question']
            #context_reference = question['context_reference']

            for case in question['cases']:
                case_id = case['id']

                # if(case_id_filter is not None and case_id != case_id_filter):
                #     continue

                user_answer = case['user_answer']
                rag_feedback_reference = case['rag_feedback_reference']

                input = {
                    "rag_question": rag_question,
                    "user_answer": user_answer,
                }
                inputs.append(input)

                output = {
                    "rag_feedback_reference": rag_feedback_reference,
                    #"context_reference": context_reference
                }
                outputs.append(output)

                metadata = {
                    "batch_id": batch_id,
                    "question_id": question_id,
                    "case_id": case_id
                }
                metadatas.append(metadata)

    dataset_langsmith = ls_client.create_dataset(dataset_name)
    ls_client.create_examples(dataset_id=dataset_langsmith.id, inputs=inputs, outputs=outputs, metadata=metadatas)
    return dataset_name


def valid_accuracy(run: Run, example: Example) -> EvaluationResult:
  run.wait()  # Wait for the run to complete
  rag_question = example.inputs["rag_question"]
  user_answer = example.inputs["user_answer"]
  rag_feedback_reference = example.outputs["rag_feedback_reference"]
  rag_feedback_actual = run.outputs["rag_feedback_actual"]

  print(f"Evaluating accuracy for batch_id={example.metadata['batch_id']} question_id={example.metadata['question_id']} case_id={example.metadata['case_id']}")

  instructions = """\

The user was asked a question and provided an answer. Then the system provided an evaluation of the answer as a feedback. \
Given the the expected feedback and the actual feedback, determine if the system correctly evaluated the accuracy of the user's answer and give a score.\


Do not judge the completeness of the answer, only the accuracy. If the answer is accurate, then the accuracy is correct.\

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


return the score in accuracy_evaluation_is_correct.\
"""

  class Response(BaseModel):
    accuracy_evaluation_is_correct: bool

  msg = f"\n\nExpected feedback:\n{rag_feedback_reference}\n\nActual feedback:\n{rag_feedback_actual}"
  response = oai_client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "system", "content": instructions,}, {"role": "user", "content": msg}],
    response_format=Response
  )

  return EvaluationResult(key="accuracy", score=int(response.choices[0].message.parsed.accuracy_evaluation_is_correct))


def valid_completeness(run: Run, example: Example) -> EvaluationResult:
  run.wait()  # Wait for the run to complete
  rag_question = example.inputs["rag_question"]
  user_answer = example.inputs["user_answer"]
  rag_feedback_reference = example.outputs["rag_feedback_reference"]
  rag_feedback_actual = run.outputs["rag_feedback_actual"]

  print(f"Evaluating completeness for batch_id={example.metadata['batch_id']} question_id={example.metadata['question_id']} case_id={example.metadata['case_id']}") 

  instructions = """\

The user was asked a question and provided an answer. Then the system provided an evaluation of the answer as a feedback. \
Given the the expected feedback and the actual feedback, determine if the system correctly evaluated the completeness of the user's answer and give a score between 0 and 1.\

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

give the score in completeness_evaluation_is_correct.\
"""

  class Response(BaseModel):
    completeness_evaluation_is_correct: bool

  msg = f"\n\nQuestion:\n{rag_question}\n\nExpected feedback:\n{rag_feedback_reference}\n\nActual feedback:\n{rag_feedback_actual}"
  response = oai_client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "system", "content": instructions,}, {"role": "user", "content": msg}],
    response_format=Response
  )

  return EvaluationResult(key="completeness", score=int(response.choices[0].message.parsed.completeness_evaluation_is_correct))


def valid_sections(run: Run, example: Example) -> EvaluationResult:
  run.wait()  # Wait for the run to complete
  rag_question = example.inputs["rag_question"]
  user_answer = example.inputs["user_answer"]
  rag_feedback_reference = example.outputs["rag_feedback_reference"]
  rag_feedback_actual = run.outputs["rag_feedback_actual"]

  print(f"Evaluating sections to review for batch_id={example.metadata['batch_id']} question_id={example.metadata['question_id']} case_id={example.metadata['case_id']}") 
  
  instructions = """\

The user was asked a question and provided an answer. Then the system provided an evaluation of the answer as a feedback. Its feedback may include sections to review in the documentation. \
Given the the expected feedback, determine if the system should send sections to review. If so, given the actual feedback, determine id it actually included sections references to its feedback, and determine if the sections are correct.\
If no sections were expected, then the system should not include any sections in its feedback.\

Do not judge anything but the sections to review.\

if the system correctly included the expecteded sections to review in the actual feedback when needed, then sections_feedback_is_correct=1.\
if the system included the sections to review in the actual although no section was expected, then sections_feedback_is_correct=0.\
if the system did not include the sections to review in the actual although sections was expected, or if the referenced sections are incorrect in the actual feedback, then sections_feedback_is_correct=0.\

"""

  class Response(BaseModel):
    sections_feedback_is_correct: bool

  msg = f"\n\nExpected feedback:\n{rag_feedback_reference}\n\nActual feedback:\n{rag_feedback_actual}"
  response = oai_client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "system", "content": instructions,}, {"role": "user", "content": msg}],
    response_format=Response
  )

  return EvaluationResult(key="sections", score=int(response.choices[0].message.parsed.sections_feedback_is_correct))


def valid_natural(run: Run, example: Example) -> EvaluationResult:
  run.wait()  # Wait for the run to complete
  rag_question = example.inputs["rag_question"]
  user_answer = example.inputs["user_answer"]
  rag_feedback_reference = example.outputs["rag_feedback_reference"]
  rag_feedback_actual = run.outputs["rag_feedback_actual"]

  print(f"Evaluating sections to review for batch_id={example.metadata['batch_id']} question_id={example.metadata['question_id']} case_id={example.metadata['case_id']}") 
  
  instructions = """\

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

give a score between 0 and 1 in sections_feedback_is_natural.\
"""

  class Response(BaseModel):
    sections_feedback_is_natural: bool

  msg = f"\n\nActual feedback:\n{rag_feedback_actual}"
  response = oai_client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "system", "content": instructions,}, {"role": "user", "content": msg}],
    response_format=Response
  )

  return EvaluationResult(key="natural", score=int(response.choices[0].message.parsed.sections_feedback_is_natural))
  

def run_evaluate():
  ls_client = Client()
  dataset_name = create_dataset(ls_client)

  analyst = AnswerAnalyst()

  #@traceable
  def target(inputs: dict) -> dict:
    print("Running target...")
    res = analyst.evaluate_answer_full_ouput(inputs["rag_question"], inputs["user_answer"])

    if res['answer'] is None:
      raise ValueError("Target failed: Answer is None")
      

    print("Target finished running.")
    return { 'rag_feedback_actual': res['answer'] }

  print("Evaluating the model...")
  results = evaluate(
    target,
    data=dataset_name,
    evaluators=[
      valid_accuracy, 
      valid_completeness, 
      valid_sections, 
      valid_natural,
      ],
    experiment_prefix="rag-as-a-study-buddy",
  )
  results.wait()

  df = results.to_pandas()


  # ls_client.read_run(run_id=results.id)
  # df = ls_client.get_test_results(project_name="rag-as-a-study-buddy")
  #df.to_csv("rag_evaluation_langsmith_results.csv", index=False)
  #df = df.iloc[-1]

  with open('rag_evaluation_langsmith_results.json', 'w', encoding='utf-8') as f:
    json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=4, default=str)



if __name__ == "__main__":
    run_evaluate()
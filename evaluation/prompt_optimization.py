import argparse
import json
import logging
from pathlib import Path
from pprint import pprint
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

from langchain_ollama import OllamaLLM
#from rag_evaluation_langsmith_run import run_evaluate
from .rag_evaluation_ragas_stepped_notation_run import run_evaluate
import re
import shutil

current_dir = Path(__file__).parent
results_file = current_dir / "rag_evaluation_ragas_results.json"

analyst_dir = current_dir.parent / "src/analyst"

LLM_MODEL = "phi4"



def run_evaluation():
    """Run the RAG evaluation script and return results"""
    max_retries = 3
    for retry in range(max_retries):
        try:
            run_evaluate()
            break
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Evaluation failed with error: {e}, retrying in 10 seconds... (Attempt {retry + 1}/{max_retries})")
                time.sleep(10)
            else:
                raise Exception("Evaluation failed after maximum retries")
    with open(results_file, "r") as f:
        return json.load(f)
    
def get_global_score(result):
    """Get the global score from the evaluation result"""

    metric_keys = ["accuracy_score_comparison", "completeness_score_comparison", "sections_score_comparison", "relevance_score_comparison"]
    scores = [result.get(key) for key in metric_keys]
    scores = [score for score in scores if score is not None]
    return sum(scores) / len(scores) if scores else 0

def get_results_doc():
    """Read the evaluation results documentation from the file"""
    with open(current_dir / "rag_evaluation_langsmith_results_doc.md", "r") as f:
        return f.read()

def get_current_prompt():
    """Read the current prompt from the file"""
    with open(analyst_dir / "answer_analyst_stepped_prompt_notation.txt", "r") as f:
        return f.read()

def save_new_prompt(prompt):
    """Save the optimized prompt to the file"""
    with open(analyst_dir / "answer_analyst_stepped_prompt_notation.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

def optimize_prompt(results, results_doc, current_prompt: str, variables: List[str]):
    """Use OpenAI to optimize the prompt based on evaluation results"""
    #llm = ChatOpenAI(temperature=1, model_name="o1-mini")
    llm = OllamaLLM(model="phi4")
    
    optimization_template = """
I designed a RAG system using {llm_model}:

The system choose a question from a set of predefined questions, which have been generated from a knowledge base. The user answers the question The system gives a feedback to the user, judging the accuracy and completeness of the answer. If useful, it also includes references to the sections of the knowledge base

    Evaluation results documentation:
    ```
    {results_doc}
    ```

    Evaluation results:
    ```
    {results}
    ```

    Current prompt:
    ```
    {current_prompt}
    ``
    
    You are an experienced AI developer and you want to improve the feedback of the system.
    Rewrite the prompt to maximize these scores for all cases : 
        - accuracy_score_comparison 
        - relevance_score_comparison
        - completeness_score_comparison 
        - sections_score_comparison 
    Each comparison should be equal to 1 for each case.
    For each case where any comparison is less than 1, analyse the case and figure out a way to improve it.
    The most important is accuracy_score_comparison, it must be 1.0 for each case.

    Rewrite the prompt to improve the feedback for cases that did not get a perfect comparison in the evaluation, but preserve the feedback for cases that already got a perfect comparison.
    
    Do not modify the variables: {variables}
    Do not add anything after "Your Evaluation:".

    
    Important: Only return the new prompt text, no explanations or introduction. Return the full prompt to be sent to the llm.

    """
    
    variables = ", ".join(variables)

    prompt = PromptTemplate(
        input_variables=["current_prompt", "results", "results_doc", "llm_model", "variables"],
        template=optimization_template
    )
    
    data = {
        "current_prompt": current_prompt,
        "results": json.dumps(results, indent=2),
        "results_doc": results_doc,
        "llm_model": LLM_MODEL,
        "variables": variables
    }

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(data)
    
    return response.strip().strip('```').strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, help='Max iterations count to run', default=5)
    parser.add_argument('--target', type=float, help='Target score, optimization is stopped when reached', default=1.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    #logging.getLogger().setLevel(logging.WARNING)  # Set root logger to WARNING

    max_iterations = args.iterations
    target_score = args.target

    results_doc = get_results_doc()
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Run evaluation
        results = run_evaluation()

        # Save the old prompt with timestamp
        shutil.copy(analyst_dir / "answer_analyst_stepped_prompt_notation.txt", analyst_dir / f"answer_analyst_stepped_prompt_notation.{timestamp}.txt")
        shutil.copy(results_file, analyst_dir / f"answer_analyst_stepped_prompt_notation.{timestamp}.results.json")


        #pprint(results)
        average_score = sum(get_global_score(result) for result in results) / len(results)
        accuracy_scores = [result.get("accuracy_score_comparison") for result in results]
        accuracy_scores = [score for score in accuracy_scores if score is not None]
        accuracy_scores_acceptable = [score for score in accuracy_scores if score >= 0.9]
        print(f"Current average score: {average_score:.2f}")
        print(f"Acceptable accuracy scores: {len(accuracy_scores_acceptable)}/{len(accuracy_scores)}")

        
        if average_score >= target_score:
            print("Target score achieved!")
            break
            
        # Get current prompt and optimize it
        current_prompt = get_current_prompt()

        variables = [match.group(1) for match in re.finditer(r'{(?!{)([^}]*)}(?!})', current_prompt)] # get all '{my_variable}' excluding '{{my_variable}}'
        new_prompt = optimize_prompt(results, results_doc, current_prompt, variables)

        if(new_prompt == current_prompt):
            print("Prompt was not changed. Stopping iterations.")
            break
        
        
        # Save the new prompt
        save_new_prompt(new_prompt)
        print("Prompt updated. Waiting before next evaluation...")
        time.sleep(10)  # Wait between iterations to avoid rate limits

if __name__ == "__main__":
    main()
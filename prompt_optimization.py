import json
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

from langchain_ollama import OllamaLLM
from common import LLM_MODEL
#from rag_evaluation_langsmith_run import run_evaluate
from rag_evaluation_ragas_run import run_evaluate


results_file = "rag_evaluation_ragas_results.json"

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
    scores = [
        result.get("feedback.accuracy", 0.0),
        result.get("feedback.completeness", 0.0),
        result.get("feedback.sections", 0.0),
        result.get("feedback.natural", 0.0)
    ]
    return sum(scores) / len(scores)

def get_results_doc():
    """Read the evaluation results documentation from the file"""
    with open("rag_evaluation_langsmith_results_doc.md", "r") as f:
        return f.read()

def get_current_prompt():
    """Read the current prompt from the file"""
    with open("answer_analyst_prompt.txt", "r") as f:
        return f.read()

def save_new_prompt(prompt):
    """Save the optimized prompt to the file"""
    with open("answer_analyst_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

def optimize_prompt(results, results_doc, current_prompt):
    """Use OpenAI to optimize the prompt based on evaluation results"""
    llm = ChatOpenAI(temperature=1, model_name="o1-mini")
    
    optimization_template = """
I designed a RAG system using {llm_model}:

The system choose a question from a set of predefined questions, which have been generated from a knowledge base. The user answers the question The system gives a feedback to the user, judging the accuracy and completeness of the answer. If useful, it also includes references to the sections of the knowledge base

The feedback sould be friendly and natural. The end user should not feel like he is talking to a machine. Feedback should not be repetitive. Feedback must not start by hello or something like that, it is part of an ongoing conversation.

    
    Evaluation results documentation:
    ```
    {results_doc}
    ```

    Evaluation results:`
    ```
    {results}
    ```

    Current prompt:
    ```
    {current_prompt}
    ``
    
    You are an experienced AI developer and you want to improve the feedback of the system.
    Rewrite the prompt to maximize the score for all cases : feedback.accuracy feedback.completeness feedback.sections feedback.natural
    Each score should be equal to 1 for each case. The most important scores are accuracy and natural
    
    Important: Only return the new prompt text, no explanations or introduction. Return the full prompt to be sent to the llm.

    """
    
    prompt = PromptTemplate(
        input_variables=["current_prompt", "results", "results_doc", "llm_model"],
        template=optimization_template
    )
    
    data = {
        "current_prompt": current_prompt,
        "results": json.dumps(results, indent=2),
        "results_doc": results_doc,
        "llm_model": LLM_MODEL,
    }

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(data)
    
    return response.strip().strip('```').strip()

def main():
    max_iterations = 2
    target_score = 0.9  # Adjust this threshold as needed
    
    results_doc = get_results_doc()
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        # Run evaluation
        results = run_evaluation()
        #pprint(results)
        average_score = sum(get_global_score(result) for result in results) / len(results)
        print(f"Current average score: {average_score:.2f}")
        
        if average_score >= target_score:
            print("Target score achieved!")
            break
            
        # Get current prompt and optimize it
        current_prompt = get_current_prompt()
        new_prompt = optimize_prompt(results, results_doc, current_prompt)

        if(new_prompt == current_prompt):
            print("Prompt was not changed. Stopping iterations.")
            break
        
        # Save the old prompt with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"answer_analyst_prompt.{timestamp}.txt"
        with open(backup_filename, "w") as f:
            f.write(current_prompt)
        
        # Save the new prompt
        save_new_prompt(new_prompt)
        print("Prompt updated. Waiting before next evaluation...")
        time.sleep(10)  # Wait between iterations to avoid rate limits

if __name__ == "__main__":
    main()
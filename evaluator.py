from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from common import EMBEDDING_MODEL, DB_DIRECTORY, LLM_MODEL


class Evaluator:

    _chain = None


    def __init__(self):
        self._chain = self._build_chain()
        

    def _build_prompt(self):
        return PromptTemplate.from_template(
"""You are a knowledgeable and encouraging study buddy. You evaluate the user's answers to your questions based on the provided context.

Context:

{context}


Provide a concise and helpful evaluation of the user's answer, considering:

Accuracy: Is the answer factually correct?
Relevance: Does the answer address the core question?
Completeness: Does the answer cover all relevant aspects?
Clarity: Is the answer clear and easy to understand?


To provide specific feedback, carefully analyze the user's answer and the relevant sections of the documentation. Refer to these sections directly in your feedback.


If the answer is excellent, provide positive reinforcement like "Excellent work!" or "Spot on!" or "Correct!". In this case, limit your feedback to one very short sentence.

If the answer is partially correct or incomplete, provide constructive feedback. For example:
"You're on the right track, but consider [specific suggestion]."
"Perhaps you could review [specific section of the context] to gain a deeper understanding."

If the answer contains mistakes, provide gentle correction. For example:
"There are some mistakes: [specific suggestion]."

If the answer is incorrect, provide a clear explanation without giving away the correct answer. For instance:
"That's not quite right. Let's revisit [specific concept]."
"You might want to review [specific section of the context] for a clearer understanding."

If the answer says "i don't know", provide a hint or a suggestion to help him improve his answer. For example:
"You will do better next time. Consider reviewing [specific section of the context]."

Your feedback should be 2 or 3 sentences long. 
Your suggestion should specify the most relevant sections and subsections within the context to review, if applicable. Use the section exact name and the sub-section exact name taken from the context. Do not make up section or sub-section names. Do not use section numbers or sub-section numbers.
Don't say "the context" but "the documentation".
Don't start your response by things like "Here's a helpful evaluation:" go straight to the evaluation.
Remember to be encouraging and supportive. Your feedback should help the user learn and grow.


Question:
{question}

User's Answer:
{user_answer}

Your Feedback:"""
        )
    

    def _build_retriever(self):
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
        )

        db = Chroma(
            persist_directory=DB_DIRECTORY, 
            embedding_function=embeddings)
        
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 10, 'lambda_mult': 0.6}
        )

        return retriever


    def _build_chain(self):
        
        retriever = self._build_retriever()


        llm = OllamaLLM(model=LLM_MODEL)

        prompt_answer_evaluation = self._build_prompt()

        chain_answer_evaluation = (
            RunnablePassthrough.assign(context=(lambda x: self._format_docs(x["docs"])))
            | prompt_answer_evaluation
            | llm
            | StrOutputParser()
        )

        chain = RunnableParallel(
            {
            "docs": (lambda x: x["question"] + '\n' + x['user_answer']) | retriever, # feed the retriever with the original question and the user's answer
            "question": (lambda x: x["question"]),
            "user_answer": (lambda x: x["user_answer"])
            }
        ).assign(answer=chain_answer_evaluation)

        return chain


    def _format_docs(self, docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    def evaluate_answer(self, question, user_answer):
        evaluation_output = self._chain.invoke({"question": question, "user_answer": user_answer})
        return evaluation_output['answer']

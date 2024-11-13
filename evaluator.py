from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


class Evaluator:

    _chain = None


    def __init__(self):
        self._chain = self._build_chain()
        

    def _build_prompt(self):
        return PromptTemplate.from_template(
"""You are a study buddy. You ask the user questions that will help him to learn the concepts within the given knowledge context. 
You asked the following question to the user. Your evaluation should be only based on the context and the question.
You evaluate the response accuracy, relevance, and completeness.
Do not use any other information than the context.

Be very concise and to the point. Provide a useful feedback.
Do not say things like "based on the context" or "I think" or "According to the text". Do not mention a "context".
Never give the correct response to the user, instead suggest parts of the context to review.

If the answer to the question is accurate, relevant and complete, your feedback is just "Correct!" without any comment. Do not include anything else.
If the answer to the question is not accurate, relevant and complete, sum up your evaluation in two sentences. The first sentence is a funny short, overall evaluation of the answer. The second sentence is optional and is a suggestion for content to review. Do not include any clue to the correct answer.
Never mention the score in the feedback.

You talk to the user and give him feedback on his answer. You don't talk about him as "the user" but as "you".


Context:
---------
{context}
---------
Questions: 
{question}:
Answer:
{user_answer}
Feedback:"""
        )
    

    def _build_retriever(self):
        embeddings = OllamaEmbeddings(
            model="llama3.2",
        )

        db = Chroma(
            persist_directory='./db/chroma', 
            embedding_function=embeddings)
        
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 10, 'lambda_mult': 0.6}
        )

        return retriever


    def _build_chain(self):
        
        retriever = self._build_retriever()


        llm = OllamaLLM(model="llama3.2")

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

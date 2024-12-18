from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from common import EMBEDDING_MODEL, DB_DIRECTORY, LLM_MODEL


class AnswerAnalyst:

    _chain = None
    _prompt_template: str = None


    def __init__(self):
        with open("evaluator_prompt.txt", "r") as f:
            self._prompt_template = f.read()
        self._chain = self._build_chain()
        

    def _build_prompt(self):
        return PromptTemplate.from_template(self._prompt_template)
    

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

    def evaluate_answer_full_ouput(self, question, user_answer):
        evaluation_output = self._chain.invoke({"question": question, "user_answer": user_answer})
        return evaluation_output

    def evaluate_answer(self, question, user_answer) -> str:
        evaluation_output = self.evaluate_answer_full_ouput(self, question, user_answer)
        return evaluation_output['answer']


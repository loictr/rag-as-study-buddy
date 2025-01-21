import os
from dotenv import load_dotenv
load_dotenv()

from pprint import pprint
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from pathlib import Path

root_dir = Path(__file__).parent.parent
DOCUMENTS_DIRECTORY = root_dir / os.getenv("DOCUMENTS_DIRECTORY")
DB_DIRECTORY = root_dir / os.getenv("DB_DIRECTORY")
QUESTIONS_PATH = root_dir / os.getenv("QUESTIONS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

class Notation(BaseModel):
    accuracy: float = Field(description="Accuracy of the answer, between 0 and 1", examples=[0.0, 1.0, 0.7])
    completeness: float = Field(description="Completeness of the answer, between 0 and 1", examples=[0.0, 1.0, 0.7])
    clarity: float = Field(description="Clarity of the answer, between 0 and 1", examples=[0.0, 1.0, 0.7])
    relevance: float = Field(description="Relevance of the answer, between 0 and 1", examples=[0.0, 1.0, 0.7])
    sections: List[str] = Field(description="Sections to review, each item represents one couple section and subsection to review", examples=["section - subsection", "section - other subsection"])
    

class AnswerAnalystStepped:

    _prompt_template_notation: str = None
    _prompt_template_textualization: str = None

    _chain_notation = None
    _chain_full = None

    _llm_notation = OllamaLLM(model=LLM_MODEL, format='json')
    _llm_textualization = OllamaLLM(model=LLM_MODEL)


    def __init__(self):
        current_dir = Path(__file__).parent
        self._notation_parser = JsonOutputParser(pydantic_object=Notation)
        with open(current_dir / "answer_analyst_stepped_prompt_notation.txt", "r") as f:
            self._prompt_template_notation = f.read()
        with open(current_dir / "answer_analyst_stepped_prompt_textualization.txt", "r") as f:
            self._prompt_template_textualization = f.read()
        self._chain_notation = self._build_chain_notation()
        self._chain_full = self._build_chain_full()
        

    def _build_prompt_notation(self):
        return PromptTemplate.from_template(self._prompt_template_notation,  partial_variables={"format_instructions": self._notation_parser.get_format_instructions()})
    
    def _build_prompt_textualization(self):
        return PromptTemplate.from_template(self._prompt_template_textualization)
    

    def _build_retriever(self):
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
        )

        db = Chroma(
            persist_directory=str(DB_DIRECTORY), 
            embedding_function=embeddings)
        
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 10, 'lambda_mult': 0.6}
        )

        return retriever


    def _build_chain_notation(self):
        
        retriever = self._build_retriever()

        prompt_notation = self._build_prompt_notation()

        chain_notation = (
            RunnablePassthrough.assign(
                context=(lambda x: self._format_docs(x["docs"]))
                )
            | prompt_notation
            | self._llm_notation
            | self._notation_parser
        )


        chain = RunnableParallel(
            {
            "docs": (lambda x: x["question"] + '\n' + x['user_answer']) | retriever, # feed the retriever with the original question and the user's answer
            "question": (lambda x: x["question"]),
            "user_answer": (lambda x: x["user_answer"]),
            }
        ).assign(notation=chain_notation)

        return chain


    def _format_docs(self, docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )
    


    def _build_chain_full(self):
        prompt_textualization = self._build_prompt_textualization()

        chain_evaluation = (
            prompt_textualization
            | self._llm_textualization
            | StrOutputParser()
        )

        chain = RunnableParallel(
            {
            "notation": self._chain_notation,
            "question": (lambda x: x["question"]),
            "user_answer": (lambda x: x["user_answer"])
            }
        ).assign(answer=chain_evaluation)

        return chain




    def notation_full_output(self, question, user_answer):
        output = self._chain_notation.invoke({
            "question": question, 
            "user_answer": user_answer})
        return output

    def evaluate_answer_full_ouput(self, question, user_answer):
        evaluation_output = self._chain_full.invoke({
            "question": question, 
            "user_answer": user_answer})
        return evaluation_output

    def evaluate_answer(self, question, user_answer) -> str:
        evaluation_output = self.evaluate_answer_full_ouput(question, user_answer)

        pprint(evaluation_output)
        return evaluation_output['answer']

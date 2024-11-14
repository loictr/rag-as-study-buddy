# RAG usage as a study buddy

Demo of using a RAG as a study buddy.
The goal is:
1. Ingest original knowledge
2. Generate questions for the user
3. Evaluate user's answers to each question and provide feedback

## Requirements

This demo needs a running Ollama instance with a pulled llama3.2 model.

You may need a `.env` file like:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
LANGCHAIN_PROJECT="rag-as-study-buddy"
```

## Knowledge generation and disclaimer

This demo uses an imaginary animal species called Zephryn as knowledge database to emulate a private or evolving knowledge base, not covered by base models.

I generated some knowledge on the imaginary species and its subspecies, including the naming, using Gemini. It is stored as markdown files. 

### Disclaimer

It should not rely on any existing real or fictional content. If it does by accident, do not check the accuracy of this files upon the original content, that was really random. I later realized Zephryn could be used as a first name. See no relation.

## Database and question generation

The study buddy demo consider the database already exists.

See `notebook Database and Questions.ipynb` for explanations about database and questions generation. Implementation for new knowledge content may be added later.

The questions are already generated in the `questions` directory but can be re-generated.

## Study buddy demo

Launch `run.py` for a simple demo running with [Gradio](https://www.gradio.app/)

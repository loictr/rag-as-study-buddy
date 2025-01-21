import gradio as gr
from gradio import ChatMessage
from  analyst.answer_analyst_stepped import AnswerAnalystStepped
from questions_repository_for_demo import QuestionsRepository

analyst = AnswerAnalystStepped()
repository = QuestionsRepository()

def on_load(messages, output):
    question = repository.get_question()
    messages.append({"role": "assistant", "content": question})
    return messages



# TODO : get the answer from langchain
def submit_answer(answer, history):
    latest_assistant_message = next((item for item in reversed(history) if item["role"] == "assistant"), None)
    result = analyst.evaluate_answer(latest_assistant_message["content"], answer)
    new_question = repository.get_question()

    # add the user answer
    history.append(
        ChatMessage(role="user",
                    content=answer)
        )
    # add the evaluation result
    history.append(
        ChatMessage(role="assistant",
                    content=result)
        )
    # new question
    history.append(
        ChatMessage(role="assistant",
                    content="Here is a new question:")
        )
    history.append(
        ChatMessage(role="assistant",
                    content=new_question)
        )

    return "", history
        



with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": "Hi."}],
        type="messages",
        height=600)

    answer = gr.Textbox()
    answer.submit(submit_answer, [answer, chatbot], [answer, chatbot])
    
    demo.load(on_load, [chatbot], [chatbot])
    
demo.launch()

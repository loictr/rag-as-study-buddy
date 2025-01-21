results are exported as json containing an array. Each item respresents a test result with

inputs.rag_question=the question the system asked to the user
inputs.user_answer=the answer of the user
inputs.rag_feedback_reference=an example of expected feedback for this case, used as a reference for judging the quality of the feedback
outputs.rag_feedback_actual=the actual feedback from the system when tested
accuracy_score_comparison=the score representing whether the actual feedback and reference feedback gave the same remarks about the accuray of the answer
completeness_score_comparison=the score representing whether the actual feedback and reference feedback gave the same remarks about the completeness of the answer
sections_score_comparison=the score representing whether the actual feedback and reference feedback gave the same remarks about the sections to review

ignore fields execution_time, example_id, id

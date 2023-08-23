import streamlit as st
import openai
openai.api_key = st.session_state["api_key"]

progress = None
try:
    progress = openai.FineTuningJob.retrieve(st.session_state['fine_tune_metadata']['id'])
except:
    pass
st.subheader('Chat with your fine-tuned model')
prompt = st.chat_input('Are you a fine-tuned model?')
if not progress:
    st.warning('Your fine-tuned model is not finished.')
else:

    def chatgpt(prompt):
        completion = openai.ChatCompletion.create(
        model=progress['fine-tuned_model'],
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message
    if prompt:
        st.text(chatgpt(prompt))

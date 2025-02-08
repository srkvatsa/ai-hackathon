import streamlit as st
import frontend.inference as inference

st.set_page_config(page_title="Ask Tao", page_icon='favicon.png')
st.title("TinyTutor 🧑‍🎓")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("What is the significance of Julius Caesar's crossing the Rubicon?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Call RAG backend (llama.cpp)
        try:
            full_response = inference.query_llama(prompt)  # Fetch response from LLaMA
        except Exception as e:
            full_response = f"Error: {e}"

        message_placeholder.markdown(full_response)
    
    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})

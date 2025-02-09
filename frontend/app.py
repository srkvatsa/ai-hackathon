import streamlit as st
import inference

st.set_page_config(page_title="Ask TinyTutor", page_icon='favicon.png')
st.title("TinyTutor 🧑‍🎓")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "textbook_results_content" not in st.session_state:
    st.session_state.textbook_results_content = ""

# CSS for independent column scrollability using container keys
st.markdown(
    """
    <style>
    /* Scrollable container for Chat History (left column) */
    .st-container-chat-history-container {
        overflow-y: scroll;
        height: 70vh; /* Adjust height as needed for chat history */
        border-right: 1px solid #eee; /* Optional separator between columns */
        padding-right: 10px;
        margin-bottom: 10px;
    }
    /* Scrollable container for Textbook Results (right column) */
    .st-container-textbook-results-container {
        overflow-y: scroll;
        max-height: 70vh; /* Adjust height as needed for textbook results */
        padding-left: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="stVerticalBlock"] {
        overflow-y: hidden;
        height: 70vh;
    }
    body {
        overflow-y: hidden; /* Hide main page vertical scrollbar */
        max-height: 70vh; /* Limit body height to viewport */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create two columns
left_column, right_column = st.columns(2)

# Left column with scrollable chat history container
with left_column:
    st.subheader("Chat History")
    chat_history_container = st.container(key="chat-history-container") # Container with key        # Display previous messages in the scrollable container
        

# Right column with scrollable textbook results container
with right_column:
    st.subheader("Textbook Results")
    textbook_results_container = st.container(key="textbook-results-container") # Container with key

    with textbook_results_container: # Textbook results content goes here
        if st.session_state.textbook_results_content:
            st.markdown(st.session_state.textbook_results_content)
        else:
            st.markdown("Textbook results will appear here after you ask a question.")

# Chat input below columns, full width
if prompt := st.chat_input("Ask TinyTutor:", key="chat_input_bottom"):
    # Display user message in left column immediately in the chat history container
    # with left_column:
    #     with chat_history_container: # Append to chat history container
    #         with st.chat_message("user"):
    #             st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # with st.chat_message("assistant"): # Main assistant message context (not displayed directly)
    #     message_placeholder = st.empty()
    #     full_response = ""

        # Call RAG backend (llama.cpp)
    try:
        full_response, context = inference.query(prompt)  # Fetch response from LLaMA
    except Exception as e:
        full_response, context = (f"Error: {e}", "")

    # message_placeholder.markdown(f"Chat Summary:\n {full_response}") # For main assistant message (not directly used in columns)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Update Textbook Results in right column container
    st.session_state.textbook_results_content = "" # Reset before updating
    if context:
            st.session_state.textbook_results_content = f"Textbook Results:\n{context}" # Add a title for clarity

    with right_column: # Update the textbook results area in right column container
        with textbook_results_container: # Update content within textbook container
            textbook_results_container.markdown(st.session_state.textbook_results_content) # Use the container to update

    # Display assistant summary in left column chat history container
    with left_column:
        with chat_history_container: # Append assistant message to chat history container
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

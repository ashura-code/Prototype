import streamlit as st
from streamlit_chat import message  # Install via: pip install streamlit-chat
import pandas as pd
import plotly.express as px
from Visualizations.AutoVisualizer import to_dataframe, auto_visualize  
from sql_LLM import run_sql_llm
from utilities.is_relevant import is_relevant_log_query,is_relevant_chart_query
import re


st.title("Logbot - Your Log Assistant")

# Keep track of chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text input area styled like a chatbot prompt
user_input = st.chat_input("Ask me anything about your logs")

# Handle input
if user_input:
    with st.spinner("Thinking real hard..."):
        try:
            if is_relevant_log_query(user_input):
                result = run_sql_llm(user_input)
                print(result['query'])  # For debugging
                df = to_dataframe(result['result'], result['columns'])

                # Save to chat history
                st.session_state.chat_history.append({
                    "role": "user", "text": user_input
                })
                st.session_state.chat_history.append({
                    "role": "assistant", "text": result['answer'], "df": df
                })
            else:
                st.session_state.chat_history.append({
                    "role": "user", "text": user_input
                })
                st.session_state.chat_history.append({
                    "role": "assistant", "text": "I can't help you with that, sorry", "df": None
                })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "user", "text": user_input
            })
            st.session_state.chat_history.append({
                "role": "assistant", "text": f"Uhh an error occurred: {e}", "df": None
            })

# ✅ Display chat history once, at the bottom
for i, entry in enumerate(st.session_state.chat_history):
    if entry["role"] == "user":
        message(entry["text"], is_user=True, key=f"user_{i}")
    else:
        message(entry["text"], key=f"assistant_{i}")
        if entry.get("df") is not None:
            st.subheader("📊 Result Table")
            st.dataframe(entry["df"], use_container_width=True)
            

            if(is_relevant_chart_query(user_input)):
                st.subheader("📈 Auto Visualization")
                figs = auto_visualize(entry["df"])
                for j, fig in enumerate(figs):
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}_{j}")







# import streamlit as st
# from Visualizations.AutoVisualizer import to_dataframe, auto_visualize  
# from sql_LLM import run_sql_llm
# import numpy as np


# st.title("📊 QueryBot")

# question = st.text_input("uhh,What do you want now?")

# if st.button("Submit") and question:
#     with st.spinner("just a sec"):
#         try:
#             result = run_sql_llm(question)
#             st.subheader("📝 Assistant Answer")
#             df = to_dataframe(result['result'], result['columns'])
            
#             st.dataframe(df, use_container_width=True)
#             st.write(result['answer'])
            

#             st.subheader("📈 Auto Visualization")
#             figs = auto_visualize(df)
#             for fig in figs:
#                 st.plotly_chart(fig, use_container_width=True)
        
#         except Exception as e:
#             st.error(f"Uhh an error occured here;s the error: {e}")





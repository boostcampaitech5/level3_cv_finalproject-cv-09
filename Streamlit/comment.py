import streamlit as st

def show_comment_page():
    st.header("Comment")
    user_comment = st.text_area("건의사항 있으시면 작성해주세요")

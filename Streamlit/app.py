import streamlit as st
from home import show_home_page
from tool import show_tool_page
from demo import show_demo_page
from comment import show_comment_page

# 사이드바에 페이지 링크 추가
page = st.sidebar.selectbox("Go to", ["Home", "Demo", "Tool", "Comment"])

# 선택된 페이지에 따라 해당 페이지의 내용을 표시
if page == "Home":
    show_home_page()
elif page == "Demo":
    show_demo_page()
elif page == "Tool":
    show_tool_page()
elif page == "Comment":
    show_comment_page()

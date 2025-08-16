import streamlit as st

# Simple test to see if Streamlit is working
st.title("🧠 Bean's Bolus Brain - Test")
st.write("If you can see this, Streamlit is working!")

# Test data loading
if 'test' not in st.session_state:
    st.session_state.test = "Session state works!"

st.write(st.session_state.test)

# Test sidebar
with st.sidebar:
    st.write("Sidebar works!")
    
if st.button("Test Button"):
    st.success("Button works!")
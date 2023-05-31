 # import libraries
import streamlit as st
import time

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")

# Title
st.title("Pomodoro Timer")

# Timer
button_clicked = st.button("Start Timer")

t1 = 1500
t2 = 300

if button_clicked:
    with st.empty():
        while t1:
            mins, secs = divmod(t1, 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            st.title(f"⏳ {timer}")
            time.sleep(0.01)
            t1 -= 1
            st.success('🍅 Time to take a break!')
    
    with st.empty():
        while t2:
            mins, secs = divmod(t2, 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            st.title(f"⏳ {timer}")
            time.sleep(0.01)
            t2 -= 1
            st.success('🍅 Time to work!')
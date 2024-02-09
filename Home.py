import streamlit as st
from modules.korean_word_maker import KoreanWordMaker

# 머신러닝 모델 캐싱
@st.cache_data
def loader():
    return KoreanWordMaker()

def generate(kw, max_length, input_text):
    return kw.make_text(max_length, input_text);

def main():
    st.title("아무말 생성기")
    kw = loader()

    input_text = st.text_input(label="한국어로 입력해주세요(외국어 모름)", value="")
    if st.button("아무말이나 만들어줘", type="primary"):
        st.write(generate(kw, 200, input_text))
        # st.write(input_text)

if __name__ == "__main__":
    main()
import streamlit as st
from modules.korean_word_maker import KoreanWordMaker

# 캐싱
@st.cache_data
def model_loader():
    return KoreanWordMaker()


def generate(model, max_length, input_text):
    return model.make_text(max_length, input_text);


def main():
    model = model_loader()
    st.title("아무말 생성기")

    input_text = st.text_input(label="한국어로 입력해주세요(외국어 모름)", value="")
    if st.button("아무말이나 만들어줘", type="primary"):
        st.write(generate(model, 200, input_text))
        # st.write(input_text)

if __name__ == "__main__":
    main()

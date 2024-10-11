import streamlit as st
from chroma import initialize_chromadb, answer
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import torch

def get_ai_message(user_message, webtoon_list):
    load_dotenv()
    llm = ChatUpstage()
    message = f"""
        사용자의 질문을 보고 사전을 참고하여
        어떤 하나의 웹툰 제목을 요구하는지 판단해주세요.
        <다른 말은 하지말고 "제목"만 대답해줘.>

        ---
        예시1)
        질문: 화산귀환의 줄거리와 유사한 웹툰을 찾아줘
        답변: 화산귀환

        예시2)
        질문: 제왕과 줄거리와 유사한 웹툰을 찾아줘
        답변: 제왕

        예시3)
        질문: 개같이탈출과 유사한 웹툰을 찾아줘
        답변: 개같이 탈출
        ---

        사전: {webtoon_list}

        질문: {user_message}
        답변: 
        """

    res = llm.invoke(message)
    return res.content


PROJECT_TITLE = "챗봇, 근데 이제 추천시스템을 곁들인.."
PAGE_FAVICON = "✌️"


st.set_page_config(page_title=PROJECT_TITLE, page_icon=PAGE_FAVICON,layout="wide")

st.title(f"{PAGE_FAVICON} {PROJECT_TITLE}")
st.caption("당신에게 알맞는 웹툰을 추천해드립니다.")

if "chromadb_collection" not in st.session_state:
    with st.spinner("ChromaDB를 초기화하는 중입니다..."):
        image_collection, synopsis_collection, webtoon_list = initialize_chromadb()
        st.session_state.chromadb_collection = image_collection
        st.session_state.synopsis_collection = synopsis_collection
        
        st.session_state.model = torch.load("./model_synopsis.pth", map_location=torch.device('cpu'))
        st.session_state.webtoon_list = webtoon_list
        st.success("ChromaDB가 성공적으로 초기화되었습니다!")

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input(
    placeholder="궁금한 내용을 질문하고 웹툰을 추천받아보세요"
):
    with st.chat_message("user"):
        st.write(user_question)
        
    with st.spinner("AI가 답변을 생성하는 중입니다."):
        with st.chat_message("ai"):
            target_webtoon = ""
            if user_question in "과 비슷한 그림체의 웹툰을 추천해줘" or user_question == "과 비슷한 줄거리의 웹툰을 추천해줘":
                    target_webtoon = user_question.split("과 비슷한 줄거리의 웹툰을 추천해줘")[0]
            else:
                target_webtoon = get_ai_message(user_question, st.session_state.webtoon_list)
                target_webtoon = target_webtoon.replace(".","")
            if "그림체" in user_question:
                similar_items = answer(st.session_state.chromadb_collection, target_webtoon)
                if similar_items == None:
                    st.success("해당 웹툰은 준비 중에 있습니다.")
                else:
                    st.write(f"재밌게 본 웹툰 : {target_webtoon}")
                    q_col = st.columns(3)
                    with q_col[0]:
                        st.image(similar_items["metadatas"][0][0]["image"])
                    rec_cols = st.columns(3)
                    for idx, item in enumerate(similar_items["metadatas"][0][1:]):
                        with rec_cols[idx]:
                            t = (
                                item["title"][:10] + ".."
                                if len(item["title"]) > 10
                                else item["title"]
                            )
                            st.write(f"ㅇ {t}")
                            st.image(item["image"])
                st.session_state.message_list.append({"role": "user", "content": user_question})
            
            if "줄거리" in user_question:

                similar_items = answer(st.session_state.synopsis_collection, target_webtoon, task="synopsis")
                if similar_items == None:
                    st.success("해당 웹툰은 준비 중에 있습니다.")
                else:
                    st.write(f"재밌게 본 웹툰 : {target_webtoon}")
                    q_col = st.columns(3)
                    with q_col[0]:
                        st.write(similar_items["metadatas"][0][0]["synopsis"])
                    
                    for idx, item in enumerate(similar_items["metadatas"][0][1:]):
                        t = (
                            item["title"][:10] + ".."
                            if len(item["title"]) > 10
                            else item["title"]
                        )
                        st.write(f"ㅇ {t}")
                        st.write(item["synopsis"])
                st.session_state.message_list.append({"role": "user", "content": user_question})

            if "찾아줘" in user_question:
                similar_items = answer(st.session_state.synopsis_collection, "튜토리얼 탑의 고인물", task="synopsis")
                if similar_items == None:
                    st.success("해당 웹툰은 준비 중에 있습니다.")
                else:
                    rec_cols = st.columns(3)
                    for idx, item in enumerate(similar_items["metadatas"][0][1:]):
                        with rec_cols[idx]:
                            t = (
                                item["title"][:10] + ".."
                                if len(item["title"]) > 10
                                else item["title"]
                            )
                            st.write(f"ㅇ {t}")
                            st.write(item["synopsis"])
                st.session_state.message_list.append({"role": "user", "content": user_question})

            
            # if "찾아줘" in user_question:
            #     target_sysnopsis = user_question[:-1].split(" ")
            #     print(st.session_state.model.encode([target_sysnopsis]))
                
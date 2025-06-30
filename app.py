# streamlit_rag_supplements.py
# ─────────────────────────────
# Ⅰ. Import & 환경 설정
# ─────────────────────────────
import os
import streamlit as st
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="영양제 Check 챗봇", page_icon="💊")

# ─────────────────────────────
# Ⅱ. PDF 로드 & 벡터스토어
# ─────────────────────────────
@st.cache_resource
def load_and_split_pdfs(folder_path: str):
    docs = []
    for f in os.listdir(folder_path):
        if f.endswith(".pdf"):
            docs.extend(PyPDFLoader(os.path.join(folder_path, f)).load_and_split())
    return docs


@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(_docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )
    faiss_index = FAISS.from_documents(split_docs, embeddings)

    persist_dir = "./faiss_index"
    # 저장 -> faiss_index/ 자동 생성됨
    # : FAISS가 생성·사용하는 벡터 인덱스 파일 보관 폴더
    faiss_index.save_local(persist_dir)
    return faiss_index


@st.cache_resource
def get_vectorstore(_docs):
    persist_dir = "./faiss_index"
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )

    if os.path.exists(os.path.join(persist_dir, "index.faiss")):   # FAISS 저장 파일
        return FAISS.load_local(persist_dir, embeddings)
    return create_vector_store(_docs)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ──────────────────────────────────────────────
# Ⅲ. LangChain 컴포넌트
# ──────────────────────────────────────────────
@st.cache_resource
def initialize_components(selected_model):
    """영양제 문서 기반 RAG 체인 초기화"""
    pages = load_and_split_pdfs("./data/supplement_knowledge")
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 1) 질문 재구성 프롬프트(동일)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question relating to "
        "dietary supplements, diseases, or schedules, reformulate a standalone "
        "question. Do NOT answer the question."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 2) Q&A 프롬프트(변경)
    qa_system_prompt = """당신은 개인 맞춤 영양제 코치입니다. 아래 정보를 기반으로
[1] 적합한 **영양제 추천 Top 3** (제품명·주요성분·추천이유)
[2] **하루 섭취 스케줄** (아침/점심/저녁/취침 전 등, 용량·주기)
[3] **주의·금기 사항** (질환, 약물, 술·흡연·카페인 상호작용)
을 표와 리스트로 한국어·존댓말로 제시하세요.

반드시 제공된 문헌에서 근거를 찾아 반영하고, 모르면 모른다고 답하세요.

{context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model, openai_api_key=OPENAI_API_KEY)
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def build_user_query(disease: str, taking: str, prefers: str) -> str:
    """3개 입력을 하나의 스탠드얼론 질문으로 묶어 줌"""
    return (
        f"질환/증상: {disease} | 현재 복용: {taking} | 원하는 특징: {prefers}.\n"
        "위 정보를 바탕으로 적합한 영양제 추천, 섭취 스케줄, 주의사항을 알려줘."
    )

# ──────────────────────────────────────────────
# Ⅳ. Streamlit UI
# ──────────────────────────────────────────────
st.header("💊 영양제 Check RAG 챗봇")

model_option = st.selectbox("GPT 모델 선택", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))
rag_chain = initialize_components(model_option)

chat_history = StreamlitChatMessageHistory(key="chat_messages")
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 입력 폼
with st.form("user_profile_form"):
    col1, col2 = st.columns(2)
    with col1:
        disease = st.text_area("① 현재 질환·증상", placeholder="예) 만성 위염, 고지혈증")
        taking = st.text_area("② 현재 복용 중인 영양제/약", placeholder="예) 밀크시슬, 종합비타민")
    with col2:
        prefers = st.text_area(
            "③ 원하는 영양제 특징", placeholder="예) 피로 회복, 간 보호, 간단한 하루 한 알"
        )
    submitted = st.form_submit_button("추천받기")

# 기존 대화 출력
for m in chat_history.messages:
    st.chat_message(m.type).write(m.content)

# 제출 이벤트
if submitted:
    if not all([disease, taking, prefers]):
        st.warning("세 항목을 모두 입력해주세요.")
        st.stop()

    user_query = build_user_query(disease, taking, prefers)
    st.chat_message("human").write(user_query)

    with st.chat_message("ai"):
        with st.spinner("분석 중..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": user_query}, config)
            answer = response["answer"]
            st.write(answer)

            with st.expander("📚 참고 문헌 보기"):
                for doc in response["context"]:
                    st.markdown(
                        f"**{os.path.basename(doc.metadata['source'])}**",
                        help=doc.page_content,
                    )

# 면책 고지
st.caption("⚠️ 본 서비스는 일반적인 건강 정보 제공을 목적으로 하며, "
           "개인 처방이 아닙니다. 복용 전 반드시 전문가와 상담하세요.")

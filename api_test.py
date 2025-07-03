# api.py
# 1. 라이브러리 임포트
import os
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import time
import re

# 2. 공공데이터 API에서 건강기능식품 수집 (XML 파싱)
def fetch_health_product_data(max_pages=5):
    url = "http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01"
    service_key = st.secrets["PUBLIC_API_KEY"]

    all_items = []
    for page in range(1, max_pages + 1):
        params = {
            'serviceKey': service_key,
            'pageNo': page,
            'numOfRows': 100,
            'type': 'xml'
        }
        response = requests.get(url, params=params, verify=False)
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        if not items:
            break
        for item in items:
            row = {child.tag: child.text for child in item}
            all_items.append(row)
        print(f"{page}페이지 완료, 누적: {len(all_items)}개")
        time.sleep(0.2)
    return all_items

items = fetch_health_product_data(max_pages=5)  # 500건

# 3. 각 item → LangChain Document로 변환
def item_to_document(item):
    text = f"""제품명: {item.get('PRDUCT', '')}
업체명: {item.get('ENTRPS', '')}
기능성: {item.get('MAIN_FNCTN', '')}
주의사항: {item.get('INTAKE_HINT1', '')}
섭취방법: {item.get('SRV_USE', '')}
표준성분: {item.get('BASE_STANDARD', '')}
"""
    return Document(page_content=text, metadata={"product": item.get("PRDUCT", "")})

documents = [item_to_document(item) for item in items]
print(f"변환된 문서 수: {len(documents)}")

# 4. 벡터 임베딩 및 FAISS 저장소 생성 (메모리 내 테스트)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)
print("벡터 저장소 생성 완료!")

# 5. 추가 데이터 수집 (6~10페이지, 총 500건)
additional_items = fetch_health_product_data(max_pages=5)  # 단순히 다시 호출하면 1~5페이지가 또 나옵니다.

# 페이지 번호 겹치지 않게 start_page 지정:
def fetch_additional_data(start_page=6, max_pages=5):
    url = "http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01"
    service_key = st.secrets["PUBLIC_API_KEY"]

    all_items = []
    for page in range(start_page, start_page + max_pages):
        params = {
            'serviceKey': service_key,
            'pageNo': page,
            'numOfRows': 100,
            'type': 'xml'
        }
        response = requests.get(url, params=params, verify=False)
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        if not items:
            break
        for item in items:
            row = {child.tag: child.text for child in item}
            all_items.append(row)
        print(f"추가 {page}페이지 완료, 누적: {len(all_items)}개")
        time.sleep(0.2)
    return all_items

# 호출
additional_items = fetch_additional_data(start_page=6, max_pages=5)  # 6~10페이지 (500건)

# 6. 문서 변환
additional_documents = [item_to_document(item) for item in additional_items]

# 8. 벡터 DB에 추가
db.add_documents(additional_documents)
print(f"벡터 저장소에 추가 완료! 전체 문서 수: {len(db.docstore._dict)}")


def search_products(ingredient_query, avoid=None, top_k: int = 5):
    """
    ingredient_query : 추천 성분(리스트 또는 쉼표 문자열)
    avoid           : 제외할 성분 리스트(옵션)
    top_k           : 최종 반환 개수
    """

    # ── [A] 입력 전처리 & 로그 ──────────────────────────────
    if isinstance(ingredient_query, list):
        ingredient_query = ", ".join(ingredient_query)
    # avoid가 문자열이면 쉼표 기준 분리
    if isinstance(avoid, str):
        avoid = [a.strip() for a in avoid.split(",") if a.strip()]

    print(f"\n[DEBUG] ingredient_query = {ingredient_query}")
    print(f"[DEBUG] avoid            = {avoid}")

    # ── [B] 후보군 검색 ───────────────────────────────────
    candidate_docs = db.similarity_search(ingredient_query, k=max(1, top_k * 4))
    print(f"[DEBUG] candidate_docs   = {len(candidate_docs)}개")

    # ── [C] ‘avoid’ 성분 필터 ─────────────────────────────
    if avoid:
        avoid_lower = [a.lower() for a in avoid]
        filtered_docs = []

        for doc in candidate_docs:
            text_lower = doc.page_content.lower()
            # 현재 문서에서 발견된 피해야 할 성분 목록
            matched = [a for a in avoid_lower if a in text_lower]

            if matched:
                # 어떤 성분 때문에 제외됐는지 표시
                product_name = doc.metadata.get("product", "이름없음")
                print(f"  ↪︎ [FILTER] {product_name}  (제외 이유: {', '.join(matched)})")
                continue

            filtered_docs.append(doc)
            if len(filtered_docs) >= top_k:
                break

        print(f"[DEBUG] filtered_docs    = {len(filtered_docs)}개 최종 반환\n")
        return filtered_docs

    # ── [D] 필터링 불필요 시 ───────────────────────────────
    print(f"[DEBUG] filtered_docs    = {top_k}개 최종 반환 (필터 없음)\n")
    return candidate_docs[:top_k]

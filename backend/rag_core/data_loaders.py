# ===== rag_core/data_loaders.py - 데이터 로딩 함수들 =====
from typing import List
from bs4 import BeautifulSoup  
import requests
import pandas as pd
from langchain_core.documents import Document
import re
from .config import DATA_DIR

def _load_web_docs() -> List[Document]:
    """한성대 도서관 웹 규정 문서를 h4+ul 단위로 로드"""
    url = "https://hsel.hansung.ac.kr/intro_data.mir"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    rule_div = soup.find("div", id="intro_rule")
    if not rule_div:
        return []

    docs: List[Document] = []

    # h4 (섹션 제목) + ul (내용 리스트) 추출
    for h4 in rule_div.find_all("h4", class_="sub_title"):
        section_title = h4.get_text(strip=True)
        ul = h4.find_next_sibling("ul")
        if ul:
            items = [li.get_text(strip=True) for li in ul.find_all("li")]
            section_text = section_title + " " + " ".join(items)

            docs.append(Document(
                page_content=section_text,
                metadata={"source": "도서관_규정", "section": section_title}
            ))

    return docs

def _load_lending_guide() -> List[Document]:
    """반납/연체 가이드 문서 로드 - 탭 구조 대응"""
    url = "https://hsel.hansung.ac.kr/lend_guide.mir"
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        docs: List[Document] = []
        
        # 탭별로 파싱 (lend02, lend03, lend04, lend05, lend06, lend08)
        tab_mapping = {
            "lend02": "반납_연체",
            "lend03": "연장_예약_분실", 
            "lend04": "이용시_주의사항",
            "lend05": "졸업생",
            "lend06": "일반/특별회원",
            "lend08": "장애학생지원"
        }
        
        for tab_id, tab_name in tab_mapping.items():
            tab_div = soup.find("div", id=tab_id)
            if not tab_div:
                continue
                
            # 각 탭 내의 h4/h5 + ul 구조 파싱
            for heading_tag in ["h4", "h5"]:
                headings = tab_div.find_all(heading_tag, class_="sub_title")
                for heading in headings:
                    section_title = heading.get_text(strip=True)
                    
                    # 다음 형제 요소에서 ul 찾기
                    ul = heading.find_next_sibling("ul", class_="info_list")
                    if ul:
                        items = [li.get_text(strip=True) for li in ul.find_all("li")]
                        section_text = section_title + " " + " ".join(items)
                        
                        docs.append(Document(
                            page_content=section_text,
                            metadata={
                                "source": "대출반납_가이드",
                                "section": section_title,
                                "tab": tab_name,
                                "type": "lending_guide"
                            }
                        ))
        
        print(f"반납/연체 가이드: {len(docs)}개 문서 로드됨")
        return docs
        
    except requests.RequestException as e:
        print(f"반납 가이드 페이지 로딩 실패: {e}")
        return []
    except Exception as e:
        print(f"반납 가이드 파싱 실패: {e}")
        return []

def _load_recommend_docs() -> List[Document]:
    """추천 결과 CSV(recommend_all.csv)를 로드하여 Document 리스트 반환"""
    csv = DATA_DIR / "recommend_all.csv"
    df = pd.read_csv(csv, encoding="utf-8")

    docs: List[Document] = []
    for _, row in df.iterrows():
        text = (
            f"[서명]{row['서명']} | "
            f"[대출횟수]{row['대출횟수']} | "
            f"[대출학생수]{row['대출학생수']} | "
            f"[학년]{row['학년']} | "
            f"[학기]{row['학기']} | "
            f"[분야]{row['분야']}"
        )
        docs.append(Document(page_content=text, metadata={
            "학년": row["학년"],
            "학기": row["학기"],
            "분야": row["분야"]
        }))
    return docs

def _extract_title(text: str) -> str:
    if not text:
        return ""
    # 케이스 A: [서명]제목 | [대출…] 패턴
    m = re.search(r"\[서명\]\s*([^|\n]+)", text)
    if m:
        return m.group(1).strip()
    # 케이스 B: "제목 | 대출…" 형태
    if " | " in text:
        return text.split(" | ", 1)[0].strip()
    # 케이스 C: 그냥 한 줄 제목만 있는 경우
    return text.strip()
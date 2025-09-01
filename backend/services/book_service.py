import os, json
from datetime import datetime

def load_books(path: str):
    """책 목록 JSON 파일 로드"""
    # path에 해당하는 JSON 파일이 존재하면 열어서 파싱하고, 없으면 빈 리스트 반환
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else []


def save_books(books, path: str):
    """책 목록 JSON 파일 저장"""
    # books 리스트를 지정된 경로(path)에 JSON 형식으로 저장
    with open(path, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=4)


def generate_expected_barcodes(books, path: str):
    """책들의 location → barcode 순서 매핑 생성 및 저장"""
    location_dict = {}
    # 각 책을 location(책장 위치)별로 그룹화해서 (barcode, position) 튜플 저장
    for book in books:
        loc, bc, pos = book["location"], book["barcode"], book["position"]
        location_dict.setdefault(loc, []).append((bc, pos))

    # 위치별로 position 기준 정렬 → 순서대로 barcode 리스트 생성
    expected = {
        loc: [bc for bc, _ in sorted(barcodes, key=lambda x: x[1])]
        for loc, barcodes in location_dict.items()
    }

    # 결과를 JSON 파일로 저장
    with open(path, "w", encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False, indent=4)

    return expected


def update_book_status_logic(books, expected_barcodes, scanned: dict):
    """스캔 결과를 바탕으로 책 상태 업데이트"""
    # scanned 데이터에서 location 키 추출 (예: {"1F-A-1": ["123", "456"]})
    location = list(scanned.keys())[0]
    scanned_barcodes = scanned[location]

    # 해당 위치가 예상 바코드 목록에 없으면 오류
    if location not in expected_barcodes:
        raise ValueError("잘못된 위치")

    expected_list = expected_barcodes[location]
    expected_set, scanned_set = set(expected_list), set(scanned_barcodes)

    # (1) 빠진 책 (expected에는 있으나 scanned에는 없음)
    missing = expected_set - scanned_set
    # (2) 잘못된 위치 책 (scanned에는 있으나 expected에는 없음)
    wrong_location = scanned_set - expected_set
    # 잘못된 위치 책은 이후 검사에서 제외
    scanned_barcodes = [b for b in scanned_barcodes if b not in wrong_location]

    # (3) 순서 검사 - 버퍼 나누기
    buffers, current = [], [scanned_barcodes[0]]
    for prev, curr in zip(scanned_barcodes, scanned_barcodes[1:]):
        # curr이 prev보다 예상 순서에서 뒤에 있으면 같은 버퍼에 추가
        if expected_list.index(curr) > expected_list.index(prev):
            current.append(curr)
        else:
            # 순서가 끊기면 새 버퍼 시작
            buffers.append(current)
            current = [curr]
    buffers.append(current)

    # 가장 긴 버퍼를 "정상적으로 꽂힌 책"으로 간주
    largest = max(buffers, key=len)
    available, misplaced = [], []
    for buf in buffers:
        if buf == largest:
            available.extend(buf)
        else:
            misplaced.extend(buf)

    # (4) 책 상태 업데이트
    for book in books:
        if book["barcode"] in available:
            book.update({"available": True, "misplaced": False, "wrong-location": False})
        elif book["barcode"] in misplaced:
            book.update({"available": False, "misplaced": True, "wrong-location": False})
        elif book["barcode"] in wrong_location:
            book.update({"available": False, "misplaced": False, "wrong-location": True})
        elif book["barcode"] in missing:
            book.update({"available": False, "misplaced": False, "wrong-location": False})

    # 갱신된 books 리스트와 요약 결과 반환
    return books, {
        "available": available,                 # 정상 위치 책
        "misplaced": misplaced,                 # 순서 잘못된 책
        "wrong-location": list(wrong_location), # 잘못 꽂힌 책
        "not-available": list(missing),         # 없는 책
    }
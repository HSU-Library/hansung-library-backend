# Flask 서버 진입점
# - React 클라이언트와 통신하는 HTTP API 제공
# - /health : 서버 상태 체크
# - /api/chat : 질문을 받아서 RAG 응답 반환
# - /api/books : 책 목록 조회
# - /api/search : 책 검색
# - /api/update_book_status : 스캔 결과 반영 및 상태 갱신

import os
import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import json, os, subprocess, paramiko
from flask_socketio import SocketIO, emit

# 내부 서비스 import
from services.rag_service import ask as rag_ask  # RAG 호출 함수
from services.book_service import (
    load_books,
    save_books,
    generate_expected_barcodes,
    update_book_status_logic
)

# .env 파일 로드 (환경변수 세팅)
load_dotenv()

# Flask App 초기화
app = Flask(__name__)

# CORS 설정: 프론트엔드 개발 서버 도메인 허용
allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
origins = [o.strip() for o in allowed.split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": origins}}, supports_credentials=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-api")

# ====== 전역 데이터 ======
BOOKS_FILE = os.path.join("data", "books.json")
EXPECTED_BARCODES_FILE = os.path.join("data", "expected_barcodes.json")

##############################################################################    수정사항1  ###
###############################################################################################
RASPBERRY_IP = "192.168.137.185"
RASPBERRY_USER = "pi"
RASPBERRY_PASS = "raspberry"

C_CODE = "/home/pi/barcode_scanner.c"
COMPILED = "/home/pi/pulse/v1/pulse"

JETSON_IP = '192.168.137.233'  # 예시 IP 주소입니다. 실제 젯슨 IP로 변경하세요.
JETSON_PORT = 22                 # SSH 기본 포트
JETSON_USER = 'hansung'    # 젯슨 사용자 이름
JETSON_PASSWORD = 'hansung'  # 젯슨 비밀번호 (sudo에도 사용)

ALLOWED_MOTOR_COMMANDS = ['1f', '2f', '3f', 'top', 'bottom']

SHELF_TO_MOTOR_COMMAND = {
    "3F-1-A-1-c": "2f",
    "3F-1-A-1-d": "1f",
    "3F-1-A-2-a": "2f",
    "3F-1-A-2-b": "2f",
    "3F-1-A-3-a": "3f",
    # 필요에 따라 다른 책장 위치와 모터 명령을 계속 추가...
    # 예: "WAREHOUSE-A": "bottom"
}
##############################################################################    수정사항1  끝 #

##############################################################################    수정사항2  ###
###############################################################################################
def send_motor_command(command):
    """
    라즈베리파이에 SSH로 접속하여 모터 제어 명령을 전송합니다.
    기존 서버의 접속 정보를 사용하도록 수정되었습니다.
    Args:
        command (str): '1f', '2f', 'top', 'bottom' 등의 명령어
    """
    # 라즈베리파이의 motor_service.py에서 지정한 파이프 경로
    pipe_path = "/tmp/motor_command_pipe"

    ssh_client = None  # finally 블록에서 사용하기 위해 밖에 선언
    try:
        # 1. SSH 클라이언트 생성 및 설정
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 2. 기존 서버의 변수를 사용하여 라즈베리파이에 연결
        print(f"Connecting to {RASPBERRY_IP} with user '{RASPBERRY_USER}'...")
        ssh_client.connect(
            hostname=RASPBERRY_IP,
            port=22,  # SSH 기본 포트
            username=RASPBERRY_USER,
            password=RASPBERRY_PASS
        )
        print("SSH connection successful.")

        # 3. 'echo' 명령어를 사용해 파이프 파일에 명령어를 씀
        ssh_command = f'echo "{command}" > {pipe_path}'
        
        stdin, stdout, stderr = ssh_client.exec_command(ssh_command)
        
        # 4. 에러 출력 확인 (디버깅에 유용)
        error_output = stderr.read().decode('utf-8').strip()
        if error_output:
            print(f"SSH command returned an error: {error_output}")
            return False
        
        print(f"Successfully sent command '{command}' to motor service.")
        return True

    except Exception as e:
        print(f"An error occurred while sending SSH command: {e}")
        return False
    finally:
        # 5. 연결 종료
        if ssh_client:
            ssh_client.close()
            print("SSH connection closed.")


def run_motion_on_jetson(block=True):
    """
    젯슨에 SSH로 접속해 /home/hansung/run_motion.sh 를 실행.
    block=True  → 시퀀스 완료까지 대기 (stdout/err 받아옴)
    block=False → 젯슨에서 백그라운드 실행(nohup) 후 즉시 응답
    """
    ssh_client = None
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=JETSON_IP,
            port=JETSON_PORT,
            username=JETSON_USER,
            password=JETSON_PASSWORD,
            timeout=10
        )

        if block:
            # 완료까지 기다리기 (디버그용·관리자용)
            full = "bash -lc '/home/hansung/run_motion.sh'"
            #full = "sudo shutdown now'"
            #full = "touch /home/hansung/test_file.txt"
            stdin, stdout, stderr = ssh_client.exec_command(full, timeout=600)
            exit_status = stdout.channel.recv_exit_status()
            out = stdout.read().decode('utf-8', 'replace')
            err = stderr.read().decode('utf-8', 'replace')
            return exit_status == 0, out, err
        else:
            # 비동기 실행(즉시 응답): HTTP 타임아웃/프론트 대기 방지
            # 원래 의도했던 스크립트 실행
            command_to_run = "/home/hansung/run_motion.sh"
            # 만약 shutdown을 실행하고 싶다면
            #command_to_run = "sudo shutdown now"
            
            # nohup과 &를 사용하여 백그라운드 실행 보장
            #full = "touch /home/hansung/test_file.txt"
            full = f"bash -lc '{command_to_run} > /dev/null 2>&1 & disown'"

            ssh_client.exec_command(full)
            return True, "started (background)", ""

    except Exception as e:
        return False, "", str(e)
    finally:
        if ssh_client:
            ssh_client.close()

@app.route('/move/sequence_bg', methods=['POST'])
def move_sequence_bg():
    ok, out, err = run_motion_on_jetson(block=False)
    if ok:
        return jsonify({"success": True, "message": out})  # "started (background)"
    return jsonify({"success": False, "stderr": err}), 500

##############################################################################    수정사항2  끝 #
####

##############################################################################    수정사항3  ###
###############################################################################################
# 1. C 코드 컴파일 및 실행 (터미널에서)
@app.route('/scan', methods=['POST'])
def scan_book():
    global robot_status
    robot_status = "scanning"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(RASPBERRY_IP, username=RASPBERRY_USER, password=RASPBERRY_PASS)

        # 터미널에서 scanner 실행 (종료는 따로)
        command = f"nohup {COMPILED} > /dev/null 2>&1 &"

        ssh.exec_command(command)

        return jsonify({"success": True, "message": "C 프로그램 백그라운드 실행 시작"})
    except Exception as e:
        return jsonify({"success": False, "message": "실행 오류", "error": str(e)})


# 2. scanner 종료 (kill)
@app.route('/scan_exit', methods=['POST'])
def stop_scanner():
    global robot_status
    robot_status = "complete"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(RASPBERRY_IP, username=RASPBERRY_USER, password=RASPBERRY_PASS)
        stop_command = f"pkill -9 -f {COMPILED}"
        ssh.exec_command(stop_command)

        return jsonify({"success": True, "message": "pulse 프로세스 종료 완료"})
    except Exception as e:
        return jsonify({"success": False, "message": "종료 오류", "error": str(e)})

# --- 모터 제어 API 라우트 ---
@app.route('/control_motor', methods=['POST'])
def control_motor():
    data = request.get_json()
    command = data.get('command')

    # 1. 유효한 명령인지 확인
    if not command or command not in ALLOWED_MOTOR_COMMANDS:
        return jsonify({"success": False, "error": f"Invalid or missing command: {command}"}), 400

    # 2. 이전에 만든 SSH 명령 전송 함수 호출
    success = send_motor_command(command)
    
    # 3. 결과 응답
    if success:
        return jsonify({
            "success": True, 
            "message": f"Command '{command}' was sent successfully."
        })
    else:
        return jsonify({
            "success": False, 
            "error": "Failed to send command to Raspberry Pi."
        }), 500
    
#--- 칸바코드에 따라 리니어모터 이동
def handle_motor_movement_for_shelf(location):
    """
    Checks the mapping table for a given shelf location and sends a command to the motor if a match is found.
    """
    # 1. Look up the motor command corresponding to the current location in the mapping table.
    motor_command = SHELF_TO_MOTOR_COMMAND.get(location)
    
    # 2. If a corresponding command exists, move the motor.
    if motor_command:
        print(f"Shelf '{location}' detected. Sending motor command: '{motor_command}'")
        # Call the previously created send_motor_command function.
        send_motor_command(motor_command)
    else:
        print(f"No motor command mapped for shelf '{location}'.")
##############################################################################    수정사항3  끝 #
###############################################################################################

##############################################################################    수정사항4  ###
 ###############################################################################################
    handle_motor_movement_for_shelf(location)
##############################################################################    수정사항4  끝 #
###############################################################################################

# 서버 시작 시 책/expected_barcodes 로드
books = load_books(BOOKS_FILE)
expected_barcodes = generate_expected_barcodes(books, EXPECTED_BARCODES_FILE)

# ====== Health Check ======
@app.get("/health")
def health():
    return {"status": "ok"}, 200

# ====== RAG 챗봇 API ======
@app.post("/api/chat")
def chat():
    """
    챗봇 API 엔드포인트
    요청(JSON): {"message": "...", "history": [...]}
    응답(JSON): {"answer": "...", "sources": [...], "usage": {...}, "requestId": "..."}
    """
    rid = str(uuid.uuid4())  # 요청 식별자
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or data.get("query") or "").strip()
    history = data.get("history") or []

    # 필수 파라미터 확인
    if not question:
        return jsonify({"error": "message (or query) is required", "requestId": rid}), 400

    try:
        # RAG 호출
        res = rag_ask(question, history=history)
        res["requestId"] = rid
        return jsonify(res), 200
    except Exception as e:
        # 에러 발생 시 로깅 후 500 반환
        log.exception("chat error")
        return jsonify({"error": str(e), "requestId": rid}), 500

# ====== Book Management API ======
@app.get("/api/books")
def get_books():
    """책 목록 조회"""
    return jsonify(books), 200

@app.get("/api/search")
def search_books():
    """책 검색 (제목/저자 포함 여부)"""
    query = request.args.get("query", "").lower()
    filtered = [b for b in books if query in b["title"].lower() or query in b["author"].lower()]
    return jsonify(filtered), 200

@app.post("/api/update_book_status")
def update_book_status():
    """
    책 상태 업데이트 API
    요청(JSON): {"location": [...barcode list...]}
    응답(JSON): {available, misplaced, wrong-location, not-available}
    """
    global books, expected_barcodes

    scanned = request.get_json()
    if not scanned:
        return jsonify({"error": "잘못된 데이터"}), 400

    try:
        books, result = update_book_status_logic(books, expected_barcodes, scanned)
        save_books(books, BOOKS_FILE)  # 저장
        return jsonify(result), 200
    except Exception as e:
        log.exception("update_book_status error")
        return jsonify({"error": str(e)}), 500

# ====== 실행 ======
if __name__ == "__main__":
    # 개발 서버 실행 (http://localhost:8000)
    app.run(host="0.0.0.0", port=8000, debug=True)
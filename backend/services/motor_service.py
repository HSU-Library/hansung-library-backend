import paramiko

# ====== 환경 변수 & 설정 ======
RASPBERRY_IP = "192.168.137.185"
RASPBERRY_USER = "pi"
RASPBERRY_PASS = "raspberry"

SHELF_TO_MOTOR_COMMAND = {
    "3F-1-A-1-c": "2f",
    "3F-1-A-1-d": "1f",
    "3F-1-A-2-a": "2f",
    "3F-1-A-2-b": "2f",
    "3F-1-A-3-a": "3f",
    # 필요 시 확장
}

# ====== 모터 제어 함수 ======
def send_motor_command(command: str):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh_client.connect(RASPBERRY_IP, username=RASPBERRY_USER, password=RASPBERRY_PASS)
        pipe_path = "/tmp/motor_command_pipe"
        ssh_client.exec_command(f'echo "{command}" > {pipe_path}')
        print(f"✅ Motor command sent: {command}")
        return True
    except Exception as e:
        print(f"❌ Motor command failed: {e}")
        return False
    finally:
        ssh_client.close()

def handle_motor_movement_for_shelf(location: str):
    command = SHELF_TO_MOTOR_COMMAND.get(location)
    if command:
        print(f"Shelf '{location}' detected → Motor command: '{command}'")
        send_motor_command(command)
    else:
        print(f"No motor command mapped for shelf '{location}'")
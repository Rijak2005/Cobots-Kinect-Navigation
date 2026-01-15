import socket
import struct
import time
import threading

ROBOT_IP = "192.168.2.1"  # use your robot ip
ROBOT_PORT = 43893
LOCAL_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', LOCAL_PORT))


def build_command(code, value=0, type=0):
    if value < 0:
        value = value + (1 << 32)
    head = struct.pack('<III', code, value, type)
    return head


# -heartbeat thread -
def heartbeat_loop():
    heartbeat_cmd = build_command(0x21040001, 0, 0)
    while True:
        try:
            sock.sendto(heartbeat_cmd, (ROBOT_IP, ROBOT_PORT))
            time.sleep(0.5)
        except:
            break


t = threading.Thread(target=heartbeat_loop)
t.daemon = True
t.start()
print(f"Success: {ROBOT_IP} Heartbeat started...")


# --- Action Functions ---

def robot_stand_up():
    print(">>> Standing up...")
    cmd = build_command(0x21010202, 0, 0)
    for _ in range(3):
        sock.sendto(cmd, (ROBOT_IP, ROBOT_PORT))
        time.sleep(0.1)
    time.sleep(5)


def robot_sit_down():
    print(">>> Sitting down...")
    cmd = build_command(0x21010202, 0, 0)
    for _ in range(3):
        sock.sendto(cmd, (ROBOT_IP, ROBOT_PORT))
        time.sleep(0.1)
    time.sleep(3)


def enter_move_mode():
    print(">>> Executing: Switch to Move Mode")
    # Code: 0x21010D06
    cmd = build_command(0x21010D06, 0, 0)
    for _ in range(5):  # Send multiple times to ensure successful switch
        sock.sendto(cmd, (ROBOT_IP, ROBOT_PORT))
        time.sleep(0.05)
    time.sleep(1)


def move_x_axis(duration, speed_value):
    if speed_value > 0:
        direction = "Forward"
    else:
        direction = "Backward"

    print(f">>> Executing: Move {direction} (Duration: {duration} seconds, Speed: {speed_value})")

    # Code: 0x21010130 (X-axis Velocity: Forward/Backward)
    start_time = time.time()
    cmd = build_command(0x21010130, speed_value, 0)

    while (time.time() - start_time) < duration:
        sock.sendto(cmd, (ROBOT_IP, ROBOT_PORT))
        time.sleep(0.02)  # 50Hz


def stop_moving():
    print(">>> Executing: Stop Moving (Buffering)")
    cmd = build_command(0x21010130, 0, 0)
    for _ in range(20):  # Send stop signal for about 0.4 seconds
        sock.sendto(cmd, (ROBOT_IP, ROBOT_PORT))
        time.sleep(0.02)
    time.sleep(1)  # Wait to come to a complete stop


# --- Main Program Flow ---
try:
    input("Please ensure there is space in front and behind the robot dog, press Enter to start...")

    # 1. Stand up
    robot_stand_up()

    # 2. Switch mode
    enter_move_mode()

    # 3. Move forward (3 seconds)
    move_x_axis(duration=3.0, speed_value=40000)

    # 4. Stop in the middle (Important! It's best to stop before changing direction)
    stop_moving()

    # 5. Move backward (3 seconds) - Use negative value
    move_x_axis(duration=3.0, speed_value=-40000)

    # 6. Stop
    stop_moving()

    # 7. Sit down
    robot_sit_down()

    print("Task completed")

except KeyboardInterrupt:
    print("Forced interruption")
finally:
    sock.close()

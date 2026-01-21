import threading
import time
import msvcrt

from jointcontroltest import Lite3UDP, RobotConfig

# Adjustable increments
COARSE_STEP = 2000
FINE_STEP = 500
HOLD_DURATION_S = 0.2
HOLD_HZ = 40.0


def clamp_axis(v: int) -> int:
    return max(Lite3UDP.AXIS_MIN, min(Lite3UDP.AXIS_MAX, v))


def adjust(target: int, delta: int) -> int:
    return clamp_axis(target + delta)


def main() -> None:
    cfg = RobotConfig()
    bot = Lite3UDP(cfg)

    target = 0
    stop_event = threading.Event()

    def holder() -> None:
        # Keep sending the target so the robot does not auto-reset height.
        while not stop_event.is_set():
            bot.send_body_height_axis(target, duration_s=HOLD_DURATION_S, hz=HOLD_HZ)

    print(
        "Height tuner controls:\n"
        "  w / Up Arrow   : raise by coarse step\n"
        "  s / Down Arrow : lower by coarse step\n"
        "  e              : raise by fine step\n"
        "  d              : lower by fine step\n"
        "  0              : return to normal (0)\n"
        "  p              : print current target\n"
        "  q              : quit (ramps back to 0)\n"
        f"Coarse step: {COARSE_STEP}, fine step: {FINE_STEP}\n"
    )

    try:
        input(
            "Safety:\n"
            "- Clear area around the robot\n"
            "- Be ready to hit STOP\n"
            "Press Enter to start..."
        )

        bot.start_heartbeat(hz=2.0)
        bot.set_manual_mode()

        print("Standing up...")
        bot.stand_toggle()
        time.sleep(5.0)

        print("Entering Pose Mode...")
        bot.enter_pose_mode()
        time.sleep(0.5)

        hold_thread = threading.Thread(target=holder, daemon=True)
        hold_thread.start()

        print("Use keys to adjust height. Current target: 0")
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch()

                # Arrow keys start with 0xe0
                if ch in (b"\x00", b"\xe0"):
                    ch = msvcrt.getch()
                    if ch == b"H":  # Up arrow
                        target = adjust(target, COARSE_STEP)
                        print(f"Raise -> target {target}")
                    elif ch == b"P":  # Down arrow
                        target = adjust(target, -COARSE_STEP)
                        print(f"Lower -> target {target}")
                    continue

                if ch in (b"w", b"W"):
                    target = adjust(target, COARSE_STEP)
                    print(f"Raise -> target {target}")
                elif ch in (b"s", b"S"):
                    target = adjust(target, -COARSE_STEP)
                    print(f"Lower -> target {target}")
                elif ch in (b"e", b"E"):
                    target = adjust(target, FINE_STEP)
                    print(f"Fine raise -> target {target}")
                elif ch in (b"d", b"D"):
                    target = adjust(target, -FINE_STEP)
                    print(f"Fine lower -> target {target}")
                elif ch == b"0":
                    target = 0
                    print("Reset -> target 0")
                elif ch in (b"p", b"P"):
                    print(f"Current target: {target}")
                elif ch in (b"q", b"Q"):
                    print("Quitting...")
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted -> STOP")
        bot.emergency_stop()
    except Exception as e:
        print(f"\nError: {e}\nSTOP for safety.")
        bot.emergency_stop()
        raise
    finally:
        stop_event.set()
        # Smoothly go back to neutral before exit.
        try:
            bot.ramp_body_height(start=target, end=0, ramp_s=1.5, hz=50.0)
        except Exception:
            pass
        bot.enter_move_mode()
        bot.close()


if __name__ == "__main__":
    main()

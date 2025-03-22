import cv2
from djitellopy import Tello
import zxc_tello as zxc
import time

tello = Tello()
tello.connect()

MAX_SPEED = 40
MIN_DISTANCE = 25


def main(drone: Tello) -> int:
    zxc.get_telemetry(drone)
    zxc.check_motors(drone)

    return 0


if __name__ == "__main__":
    main(tello)

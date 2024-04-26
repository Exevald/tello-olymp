import cv2
from djitellopy import Tello
import zxc_tello as zxc
import time

tello = Tello()
tello.connect()

MAX_SPEED = 40
MIN_DISTANCE = 25


def main(drone: Tello) -> int:
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.streamon()
    zxc.get_telemetry(drone)
    drone.set_speed(zxc.MAX_STRAIGHT_SPEED)
    drone.takeoff()

    # Взлёт
    drone.move_up(30)
    time.sleep(0.5)

    # Полёт к 5 маркеру
    drone.go_xyz_speed(85, 95, 0, zxc.MAX_DIAGONAL_SPEED)
    zxc.fly_to_aruco_center(drone, 5)
    time.sleep(0.5)

    # Полёт к 24 маркеру
    drone.move_forward(130)
    zxc.fly_to_aruco_center(drone, 24)
    time.sleep(0.5)

    # Полёт к 20 маркеру
    drone.move_left(255)
    zxc.fly_to_aruco_center(drone, 20)
    time.sleep(3)

    # Полёт к 22 маркеру
    drone.move_forward(130)
    zxc.fly_to_aruco_center(drone, 22)
    time.sleep(0.5)

    # Полёт к 25 маркеру
    drone.move_right(130)
    zxc.fly_to_aruco_center(drone, 25)
    time.sleep(0.5)

    # Полёт к 23 маркеру
    drone.move_back(255)
    zxc.fly_to_aruco_center(drone, 23)
    time.sleep(3)

    # Полёт к 17 маркеру
    drone.move_left(130)
    zxc.fly_to_aruco_center(drone, 17)
    time.sleep(0.5)

    # Полёт к финишу
    drone.go_xyz_speed(-85, -348, 0, zxc.MAX_DIAGONAL_SPEED)
    time.sleep(0.5)

    # Посадка
    drone.streamoff()
    cv2.destroyAllWindows()
    drone.land()
    drone.end()

    return 0


if __name__ == "__main__":
    main(tello)

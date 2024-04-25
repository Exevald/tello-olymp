import time
from djitellopy import Tello
from zxc_tello import aruco_utils
import cv2

tello = Tello()
tello.connect()

MAX_DIAGONAL_SPEED = 40
MAX_STRAIGHT_SPEED = 60
MIN_DISTANCE = 25
FRAME_WIDTH = 240
FRAME_HEIGHT = 320


# Get aruco telemetry data
def get_telemetry(drone: Tello):
    battery = drone.get_battery()
    height = drone.get_height()
    print('battery: ', battery)
    print('height: ', height)


# Check drone motors
def check_motors(drone: Tello):
    drone.takeoff()
    drone.land()


# Go to aruco marker with aruco_id
# TODO: нормально обработать аварийную посадку при ненахождении маркера
def fly_to_aruco_center(drone: Tello, aruco_id: int):
    stop_tello_motion = False
    while True:
        if stop_tello_motion is True:
            tello.send_rc_control(0, 0, 0, 0)
            break
        frame_img = drone.get_frame_read()
        if frame_img is None:
            continue
        img = frame_img.frame
        img = img[0:FRAME_WIDTH, 0:FRAME_HEIGHT]
        img, marker_details = aruco_utils.detect_markers_in_image(img, draw_center=True, draw_reference_corner=True)
        index = [i for i, d in enumerate(marker_details) if d[1] == aruco_id]
        if len(marker_details) > 0 and index:
            center_x, center_y = marker_details[index[0]][0]
            img, x_distance, y_distance, distance = aruco_utils.detect_distance_from_image_center(img, center_x,
                                                                                                  center_y)
            l_r_speed = int((MAX_DIAGONAL_SPEED * y_distance) / (FRAME_WIDTH // 2) * (-1))
            f_b_speed = int((MAX_DIAGONAL_SPEED * x_distance / (FRAME_HEIGHT // 2)) * (-1))
            try:
                if abs(distance) <= MIN_DISTANCE:
                    tello.send_rc_control(0, 0, 0, 0)
                    stop_tello_motion = True
                else:
                    tello.send_rc_control(l_r_speed, f_b_speed, 0, 0)
            except Exception as exc:
                print("send_rc_control exception: ", exc)

        cv2.imshow('drone', img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            drone.streamoff()
            break

    drone.send_rc_control(0, 0, 0, 0)


# Get video stream with aruco markers detect
def get_aruco_stream(drone: Tello):
    stop_tello_motion = False
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.streamon()
    while True:
        if stop_tello_motion is True:
            tello.send_rc_control(0, 0, 0, 0)
            break
        frame_img = drone.get_frame_read()
        if frame_img is None:
            continue
        frame_width = 240
        frame_height = 320
        img = frame_img.frame
        img = img[0:frame_width, 0:frame_height]
        img, marker_details = aruco_utils.detect_markers_in_image(img, draw_center=True, draw_reference_corner=True)

        cv2.imshow('drone', img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            drone.streamoff()
            break


def main(drone: Tello) -> int:
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.streamon()
    get_telemetry(drone)
    drone.set_speed(MAX_STRAIGHT_SPEED)
    drone.takeoff()
    # Взлёт
    drone.move_up(30)
    time.sleep(0.5)

    # Полёт к 8 маркеру
    drone.go_xyz_speed(100, 100, 0, MAX_DIAGONAL_SPEED)
    fly_to_aruco_center(drone, 8)
    time.sleep(0.5)

    # Полёт к 2 маркеру
    drone.move_forward(145)
    fly_to_aruco_center(drone, 2)
    time.sleep(4)

    # Полёт к 0 маркеру
    drone.move_left(145)
    fly_to_aruco_center(drone, 0)
    time.sleep(0.5)

    # Полёт к 6 маркеру
    drone.move_back(145)
    fly_to_aruco_center(drone, 6)
    time.sleep(0.5)

    # Полёт к 7 маркеру
    drone.move_right(75)
    fly_to_aruco_center(drone, 7)
    time.sleep(0.5)

    # Полёт к 4 маркеру
    drone.move_forward(75)
    fly_to_aruco_center(drone, 4)
    time.sleep(0.5)

    # Полёт к 3 маркеру
    drone.move_left(75)
    fly_to_aruco_center(drone, 3)
    time.sleep(4)

    # Полёт к 6 маркеру
    drone.move_back(75)
    fly_to_aruco_center(drone, 6)
    time.sleep(0.5)

    # Полёт к 8 маркеру
    drone.move_right(145)
    fly_to_aruco_center(drone, 8)
    time.sleep(0.5)

    # Финиш
    drone.go_xyz_speed(-93, -93, 0, MAX_DIAGONAL_SPEED)
    time.sleep(0.5)

    # Посадка
    drone.streamoff()
    cv2.destroyAllWindows()
    drone.land()
    drone.end()

    return 0


if __name__ == "__main__":
    main(tello)

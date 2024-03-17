from djitellopy import Tello
import cv2
import aruco_utils

tello = Tello()
tello.connect()

MAX_SPEED = 40
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
            print(index)
            center_x, center_y = marker_details[index[0]][0]
            img, x_distance, y_distance, distance = aruco_utils.detect_distance_from_image_center(img, center_x,
                                                                                                  center_y)
            l_r_speed = int((MAX_SPEED * y_distance) / (FRAME_WIDTH // 2) * (-1))
            f_b_speed = int((MAX_SPEED * x_distance / (FRAME_HEIGHT // 2)) * (-1))
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

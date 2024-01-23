from djitellopy import Tello
import cv2

tello = Tello()
tello.connect()


def get_telemetry(drone: Tello):
    battery = drone.get_battery()
    height = drone.get_height()
    print('battery: ', battery)
    print('height: ', height)


def check_motors(drone: Tello):
    drone.takeoff()
    drone.land()


def get_frame(drone: Tello):
    drone.streamon()
    frame_read = drone.get_frame_read()

    drone.takeoff()
    while True:
        img = frame_read.frame
        cv2.imshow('drone', img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            drone.streamoff()
            break


get_telemetry(tello)
get_frame(tello)

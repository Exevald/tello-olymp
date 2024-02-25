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


def run_bottom_video(drone: Tello):
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.streamon()
    frame_read = drone.get_frame_read()

    drone.takeoff()
    while True:
        frame = frame_read.frame
        crop_img = frame[0:240, 0:320]
        cv2.imshow('drone', crop_img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    drone.streamoff()
    cv2.destroyAllWindows()
    drone.end()


def main(drone: Tello):
    get_telemetry(drone)
    run_bottom_video(drone)


if __name__ == "__main__":
    main(tello)

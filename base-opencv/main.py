from djitellopy import Tello
import cv2
import aruco_utils
tello = Tello()
tello.connect()
MAX_SPEED = 30
MIN_DISTANCE = 20

def get_telemetry(drone: Tello):
    battery = drone.get_battery()
    height = drone.get_height()
    print('battery: ', battery)
    print('height: ', height)


def check_motors(drone: Tello):
    drone.takeoff()
    drone.land()

def get_frame_down(drone: Tello):
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.streamon()

    #drone.takeoff()
    while True:
        frame_img = drone.get_frame_read()
        img = frame_img.frame
        img = img[0:240, 0:320]        
        cv2.imshow('drone', img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            drone.streamoff()
            break
    # drone.streamoff()
    cv2.destroyAllWindows()
    drone.end()

def find_aruco_on_bottom_cam(drone: Tello):
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.streamon()

    #drone.takeoff()
    while True:
        frame_img = drone.get_frame_read()
        img = frame_img.frame
        img = img[0:240, 0:320]
        img, marker_details = aruco_utils.detect_markers_in_image(img, draw_center=True, draw_reference_corner=True)
        if len(marker_details) > 0:
            center_x, center_y = marker_details[0][0]
            img, x_distance, y_distance, distance = aruco_utils.detect_distance_from_image_center(img, center_x,
                                                                                    center_y)
        
        cv2.imshow('drone', img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            drone.streamoff()
            break
    # drone.streamoff()
    cv2.destroyAllWindows()
    drone.end()

def fly_to_aruco_center(drone: Tello):
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.streamon()
    STOP_TELLO_MOTION = False
    drone.takeoff()
    while True:
        if STOP_TELLO_MOTION is True:
            # then send the zero velocity via rc_control to stop
            # any motion of the Tello.
            tello.send_rc_control(0, 0, 0, 0)
            drone.land()
            drone.end()
            break
            # reset the STOP_TELLO_MOTION flag to false as we have handled the
            # request
            STOP_TELLO_MOTION = False
        frame_img = drone.get_frame_read()
        #(H, W) = frame_img.shape[:2]
        W = 240
        H = 320
        img = frame_img.frame
        img = img[0:240, 0:320]
        img, marker_details = aruco_utils.detect_markers_in_image(img, draw_center=True, draw_reference_corner=True)
        if len(marker_details) > 0:
            center_x, center_y = marker_details[0][0]
            img, x_distance, y_distance, distance = aruco_utils.detect_distance_from_image_center(img, center_x,
                                                                                    center_y)
            l_r_speed = int((MAX_SPEED * x_distance) / (W // 2))
            # *-1 because the documentation says
            # that negative numbers go up but I am
            # seeing negative numbers go down
            f_b_speed = int((MAX_SPEED * y_distance / (H // 2)))
            try:
                if abs(distance) <= MIN_DISTANCE:
                    f_b_speed = 0
                    l_r_speed = 0
                    # True - we want to keep looking for markers
                    # Instruct Tello to hover
                    tello.send_rc_control(0, 0, 0, 0)
                    STOP_TELLO_MOTION = True

                else:
                    # we are not close enough to the ArUco marker, so keep flying
                    tello.send_rc_control(l_r_speed, f_b_speed, 0, 0)

            except Exception as exc:
                print("send_rc_control exception")
        
        cv2.imshow('drone', img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            drone.streamoff()
            break
    # drone.streamoff()
    cv2.destroyAllWindows()
    drone.end()    

def get_frame(drone: Tello):
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.streamon()
    # frame_read = drone.get_frame_read()

    drone.takeoff()
    while True:
        frame = drone.get_frame_read().frame
        crop_img = frame[0:240, 0:320]
        cv2.imshow('drone', crop_img)
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
    # run_bottom_video(drone)
    #get_frame(drone)
    #get_frame_down(drone)
    fly_to_aruco_center(drone)


if __name__ == "__main__":
    main(tello)

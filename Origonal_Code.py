import os
import time
import random
import cv2
import numpy as np
import rclpy

from std_msgs.msg import String
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from red_detection import RedStopController
from april_tag_detection_original  import AprilTagInterpreter


class MoorebotLaneFollower(Node):
    def __init__(self, bot_name, save_debug=False):
        super().__init__(f'{bot_name}_lane_follower')

        self.bot_name = bot_name
        self.save_debug = save_debug

        self.bridge = CvBridge()
        self.latest_image = None

        self.red_stop = RedStopController()
        self.apriltag = AprilTagInterpreter()

        self.subscription = self.create_subscription(
            Image,
            f'/{self.bot_name}/camera/image',
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            f'/{self.bot_name}/cmd_vel',
            10
        )

        self.state_pub = self.create_publisher(
            String,
            f'/{self.bot_name}/state',
            10
        )

        self.timer = self.create_timer(0.05, self.process_image)

        self.linear_speed = 0.11
        self.right_forward_distance = 0.144
        self.left_forward_distance = 0.54       # was 56 longer straight before left turn
        self.straight_forward_distance = 0.195
        self.post_left_forward_distance = 0.14
        # extra straight-only settle after straight decision
        self.post_straight_forward_distance = 0.18

        # turn durations
        self.left_turn_time = 1.75
        self.right_turn_time = 2.2
        self.straight_turn_time = 2.35

        # sharper correction ONLY when a line is seen
        self.angular_gain = 0.0085
        self.min_angular = 0.02
        self.max_angular = 0.60
        self.max_angular_step = 0.07
        self.last_angular_z = 0.0

        if self.bot_name == 'duckiedonald':
            self.linear_speed = 0.09
            self.right_forward_distance = 0.13
            self.left_forward_distance = 0.36
            self.straight_forward_distance = 0.180
            self.post_left_forward_distance = 0.10
            # extra straight-only settle after straight decision
            self.post_straight_forward_distance = 0.18

            # turn durations
            self.left_turn_time = 1.60
            self.right_turn_time = 2.1
            self.straight_turn_time = 2.3
            self.angular_gain = 0.0068
            self.max_angular_step = 0.04
            self.max_angular = 0.45

        self.lane_center_history = []
        self.history_len = 6

        self.intersection_state = "FOLLOW_LANE"
        self.selected_direction = None
        self.state_start_time = None

        self.last_tag_directions = None
        self.last_tag_time = None

    # lost-lane behavior:
    # go straight first for longer, then begin gentle search turn
        self.lost_lane_state = "NONE"
        self.lost_lane_start_time = None
        self.lost_lane_straight_time = 0.45
        self.lost_lane_turn_speed = 0.09
        self.lost_lane_forward_speed = 0.055
    def image_callback(self, msg):
        self.latest_image = msg

    def clamp_step(self, target, current, max_step):
        if target > current + max_step:
            return current + max_step
        elif target < current - max_step:
            return current - max_step
        return target

    def reset_lost_lane(self):
        self.lost_lane_state = "NONE"
        self.lost_lane_start_time = None

    def maybe_save_debug(self, debug, yellow_mask, white_mask):
        if not self.save_debug:
            return

        now = time.time()
        if now - self.last_debug_save_time < self.debug_save_interval:
            return

        self.last_debug_save_time = now
        stamp = int(now * 1000)

        try:
            cv2.imwrite(f"{self.debug_dir}/{stamp}_debug.jpg", debug)
            cv2.imwrite(f"{self.debug_dir}/{stamp}_yellow.jpg", yellow_mask)
            cv2.imwrite(f"{self.debug_dir}/{stamp}_white.jpg", white_mask)
        except Exception as e:
            self.get_logger().error(f"{self.bot_name} debug save error: {e}")

    def process_image(self):
        if self.latest_image is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_image, 'rgb8')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (320, 240))

            h, w, _ = frame.shape
            roi = frame[int(h * 0.70):h, :]
            roi = cv2.GaussianBlur(roi, (5, 5), 0)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([10, 40, 40])
            upper_yellow = np.array([48, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            lower_white = np.array([0, 0, 185])
            upper_white = np.array([180, 40, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            lower_red1 = np.array([0, 150, 150])
            upper_red1 = np.array([10, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([170, 150, 150])
            upper_red2 = np.array([180, 255, 255])
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            kernel = np.ones((5, 5), np.uint8)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

            current_time = time.time()

            allowed_directions = self.apriltag.update(frame)
            if allowed_directions is not None:
                self.last_tag_directions = allowed_directions
                self.last_tag_time = current_time

            red_detected = self.red_stop.update(red_mask)

            yellow_cx = self.get_centroid(yellow_mask, min_area=0)
            white_cx = self.get_centroid(white_mask, min_area=40)  # changed from 80 to 40

            if white_cx is not None:
                white_cx += 42

            twist = Twist()
            img_center = w // 2

            if yellow_cx is not None and white_cx is not None:
                lane_center = (yellow_cx + white_cx) // 2
            elif yellow_cx is not None:
                lane_center = yellow_cx + 130
            elif white_cx is not None:
                lane_center = white_cx - 125
            else:
                lane_center = None

            lane_center_smoothed = None

            if self.intersection_state == "FOLLOW_LANE":

                if lane_center is not None:
                    self.reset_lost_lane()

                    self.lane_center_history.append(lane_center)
                    if len(self.lane_center_history) > self.history_len:
                        self.lane_center_history.pop(0)

                    lane_center_smoothed = int(np.mean(self.lane_center_history))
                    error = img_center - lane_center_smoothed

                    if abs(error) < 8:
                        error = 0

                    target_angular = self.angular_gain * error

                    if 0 < abs(target_angular) < self.min_angular:
                        target_angular = np.sign(target_angular) * self.min_angular

                    target_angular = np.clip(target_angular, -self.max_angular, self.max_angular)

                    smooth_angular = self.clamp_step(
                        target_angular,
                        self.last_angular_z,
                        self.max_angular_step
                    )
                    self.last_angular_z = smooth_angular

                    twist.linear.x = self.linear_speed
                    twist.angular.z = float(smooth_angular)

                else:
                    # no line seen: go straight longer first, then begin gentle search
                    if self.lost_lane_state == "NONE":
                        self.lost_lane_state = "STRAIGHT"
                        self.lost_lane_start_time = current_time

                    elapsed = current_time - self.lost_lane_start_time

                    if self.lost_lane_state == "STRAIGHT":
                        twist.linear.x = self.linear_speed
                        twist.angular.z = 0.0
                        self.last_angular_z = 0.0

                        if elapsed >= self.lost_lane_straight_time:
                            self.lost_lane_state = "SEARCH"
                            self.lost_lane_start_time = current_time

                    elif self.lost_lane_state == "SEARCH":
                        target_angular = self.lost_lane_turn_speed

                        smooth_angular = self.clamp_step(
                            target_angular,
                            self.last_angular_z,
                            self.max_angular_step
                        )
                        self.last_angular_z = smooth_angular

                        twist.linear.x = self.lost_lane_forward_speed
                        twist.angular.z = float(smooth_angular)

                if red_detected:
                    if self.last_tag_directions is not None and \
                       current_time - self.last_tag_time < 2.0:
                        self.selected_direction = random.choice(self.last_tag_directions)
                    else:
                        self.selected_direction = random.choice(
                            ["LEFT", "RIGHT", "STRAIGHT"]
                        )

                    print(f"{self.bot_name} INTERSECTION DECISION: {self.selected_direction}")

                    self.intersection_state = "STOP"
                    self.state_start_time = current_time
                    self.last_angular_z = 0.0
                    self.reset_lost_lane()

            elif self.intersection_state == "STOP":
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.last_angular_z = 0.0
                self.reset_lost_lane()

                if current_time - self.state_start_time > 1.5:
                    self.intersection_state = "FORWARD"
                    self.state_start_time = current_time

            elif self.intersection_state == "FORWARD":
                twist.angular.z = 0.0
                self.last_angular_z = 0.0
                self.reset_lost_lane()

                if self.selected_direction == "RIGHT":
                    twist.linear.x = 0.12
                    required_distance = self.right_forward_distance
                elif self.selected_direction == "LEFT":
                    twist.linear.x = 0.12
                    required_distance = self.left_forward_distance
                else:
                    twist.linear.x = self.linear_speed
                    required_distance = self.straight_forward_distance

                required_time = required_distance / max(twist.linear.x, 0.001)

                if current_time - self.state_start_time > required_time:
                    self.intersection_state = "TURN"
                    self.state_start_time = current_time

            elif self.intersection_state == "TURN":
                self.reset_lost_lane()
                self.last_angular_z = 0.0

                if self.selected_direction == "LEFT":
                    twist.linear.x = 0.1
                    twist.angular.z = 0.72
                    turn_time = self.left_turn_time
                elif self.selected_direction == "RIGHT":
                    twist.linear.x = 0.1
                    twist.angular.z = -0.9
                    turn_time = self.right_turn_time
                else:
                    twist.linear.x = self.linear_speed
                    twist.angular.z = 0.0
                    turn_time = self.straight_turn_time

                if current_time - self.state_start_time > turn_time:
                    if self.selected_direction == "LEFT":
                        self.intersection_state = "POST_LEFT_FORWARD"
                        self.state_start_time = current_time
                    elif self.selected_direction == "STRAIGHT":
                        self.intersection_state = "POST_STRAIGHT_FORWARD"
                        self.state_start_time = current_time
                    else:
                        self.intersection_state = "FOLLOW_LANE"
                        self.selected_direction = None
                        self.lane_center_history = []

            elif self.intersection_state == "POST_LEFT_FORWARD":
                twist.linear.x = self.linear_speed
                twist.angular.z = 0.0
                self.last_angular_z = 0.0
                self.reset_lost_lane()

                required_time = self.post_left_forward_distance / max(twist.linear.x, 0.001)

                if current_time - self.state_start_time > required_time:
                    self.intersection_state = "FOLLOW_LANE"
                    self.selected_direction = None
                    self.lane_center_history = []

            elif self.intersection_state == "POST_STRAIGHT_FORWARD":
                twist.linear.x = self.linear_speed
                twist.angular.z = 0.0
                self.last_angular_z = 0.0
                self.reset_lost_lane()

                required_time = self.post_straight_forward_distance / max(twist.linear.x, 0.001)

                if current_time - self.state_start_time > required_time:
                    self.intersection_state = "FOLLOW_LANE"
                    self.selected_direction = None
                    self.lane_center_history = []

            self.cmd_pub.publish(twist)

            state_msg = String()
            state_msg.data = self.intersection_state
            self.state_pub.publish(state_msg)

            # save debug images instead of opening GUI windows
            if self.save_debug:
                debug = roi.copy()

                cv2.line(debug, (img_center, 0), (img_center, debug.shape[0]), (0, 0, 255), 2)

                if yellow_cx is not None:
                    cv2.circle(debug, (yellow_cx, 30), 6, (0, 255, 255), -1)
                    cv2.putText(debug, "yellow", (yellow_cx + 8, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                if white_cx is not None:
                    cv2.circle(debug, (white_cx, 55), 6, (255, 255, 255), -1)
                    cv2.putText(debug, "white", (white_cx + 8, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                if lane_center is not None:
                    cv2.circle(debug, (lane_center, 80), 6, (0, 255, 0), -1)
                    cv2.putText(debug, "lane center", (lane_center + 8, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                if lane_center_smoothed is not None:
                    cv2.circle(debug, (lane_center_smoothed, debug.shape[0] - 18), 8, (255, 0, 0), -1)
                    cv2.putText(debug, "target", (lane_center_smoothed + 8, debug.shape[0] - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

                cv2.putText(debug, f"State: {self.intersection_state}", (10, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                self.maybe_save_debug(debug, yellow_mask, white_mask)

        except Exception as e:
            self.get_logger().error(f"{self.bot_name} processing error: {e}")

    def get_centroid(self, mask, min_area=50):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        if min_area <= 0:
            valid_contours = contours
        else:
            valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not valid_contours:
            return None

        largest = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        return int(M["m10"] / M["m00"])


def main(args=None):
    rclpy.init(args=args)

    # save_debug=True will write images to /tmp/duckiescrooge_debug
    # without using any GUI windows
    bot2 = MoorebotLaneFollower('duckiescrooge', save_debug=False)
    executor = MultiThreadedExecutor()
    #executor.add_node(bot2)
    # save_debug=True will write images to /tmp/duckiescrooge_debug
    # without using any GUI windows
    bot1 = MoorebotLaneFollower('duckiedonald', save_debug=False)

    executor.add_node(bot1)
    executor.add_node(bot2)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown(timeout_sec=0.1)
        except Exception:
            pass

        bot2.destroy_node()
        bot1.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

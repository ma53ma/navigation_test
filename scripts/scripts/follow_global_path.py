#!/usr/bin/env python
import rospy
from nav_msgs.srv import GetPlan
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped, Twist, Vector3Stamped, PoseArray

import tf_conversions

import tf2_ros
import tf2_geometry_msgs

class Agent:
    def __init__(self, num_obsts, world, start_xs, start_ys):
        # /move_base for TEB
        # /move_base_virtual for DGap
        rospy.wait_for_service('/move_base_virtual/make_plan')
        self.get_plan = rospy.ServiceProxy('/move_base_virtual/make_plan', GetPlan)
        self.world = world
        self.plan_idx = 0
        self.x = 0.0
        self.y = 0.0
        self.plan = None
        self.agent_odoms = None
        self.odom_subs = {}
        self.cmd_vel_pubs = {}
        self.need_plan = {}
        self.plans = {}
        self.plan_indices = {}
        self.plan_publishers = {}
        self.plans_to_publish = {}
        self.tolerances = {}
        self.error_min1s = {}
        self.error_min2s = {}
        self.prev_ts = {}

        stripped_xs = start_xs[1:-1]
        self.split_xs = stripped_xs.split(',')
        stripped_ys = start_ys[1:-1]
        self.split_ys = stripped_ys.split(',')

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # top left to bottom right ((x, y) to (x,y)
        # these are in world / known_map
        self.campus_goal_regions = [[6, 8, 7, 16], [5, 15, 7, 18],[1, 28, 5, 30],[1, 16, 4, 22],
                              [4, 20, 11, 24],[13, 22, 16, 28],[16, 25, 29, 26],
                              [12, 18, 15, 22],[20, 18, 24, 22],[16, 12, 18, 17],
                              [19, 14, 23, 17],[25, 12, 29, 18],[10, 8, 17, 12],
                              [1, 6, 11, 8], [9, 4, 14, 6], [16, 1, 22, 8]]

        self.empty_goal_regions = [[18, 10, 19, 9]]

        self.empty_world_transform = [13.630, 13.499]
        self.campus_world_transform = [14.990204, 13.294787]

        world = "campus"
        if world == "empty":
            self.world_transform = self.empty_world_transform
            self.goal_regions = self.empty_goal_regions
        elif world == "campus":
            self.world_transform = self.campus_world_transform
            self.goal_regions = self.campus_goal_regions

        for i in range(0, num_obsts):
            robot_namespace = "robot" + str(i)
            self.need_plan[robot_namespace] = True
            self.plan_indices[robot_namespace] = 0
            self.plan_publishers[robot_namespace] = rospy.Publisher(robot_namespace + "/global_path", PoseArray, queue_size=5)
            self.tolerances[robot_namespace] = 0.5
            start = self.get_start(i)
            self.get_global_plan(start, robot_namespace)

            self.odom_subs[robot_namespace] = rospy.Subscriber(robot_namespace + "/odom", Odometry, self.odom_CB, queue_size=5)
            self.cmd_vel_pubs[robot_namespace] = rospy.Publisher(robot_namespace + "/cmd_vel", Twist, queue_size=5)
            self.error_min1s[robot_namespace] = np.array([0.0, 0.0])
            self.error_min2s[robot_namespace] = np.array([0.0, 0.0])
            # self.prev_ts[robot_namespace] = rospy.Time.now().to_sec()

    def get_start(self, i):
        new_start = [int(self.split_xs[i]), int(self.split_ys[i])]

        start = PoseStamped()
        start.header.frame_id = "known_map"
        start.header.stamp = rospy.Time.now()
        start.pose.position.x = new_start[0] - self.world_transform[0]
        start.pose.position.y = new_start[1] - self.world_transform[1]
        start.pose.position.y = new_start[1] - self.world_transform[1]
        start.pose.position.z = 0.0
        start.pose.orientation.w = 1.0

        return start

    def odom_CB(self, msg):
        robot_namespace = msg.child_frame_id

        map_static_to_known_map_trans = self.tfBuffer.lookup_transform("known_map", "map_static", rospy.Time(), rospy.Duration(3.0))

        odom_in_known_map = tf2_geometry_msgs.do_transform_pose(msg.pose, map_static_to_known_map_trans)

        desired_pose = self.plans[robot_namespace].plan.poses[self.plan_indices[robot_namespace]]
        x_diff = odom_in_known_map.pose.position.x - desired_pose.pose.position.x
        y_diff = odom_in_known_map.pose.position.y - desired_pose.pose.position.y

        # print('delta_x: ', delta_x)

        # calculate cmd_vel
        known_map_to_robot_trans = self.tfBuffer.lookup_transform(robot_namespace, "known_map", rospy.Time())
        diff_vect = Vector3Stamped()
        diff_vect.header.frame_id = "known_map"
        diff_vect.header.stamp = rospy.Time.now()
        diff_vect.vector.x = -x_diff
        diff_vect.vector.y = -y_diff
        #print('difference vector in known map: ', diff_vect.vector.x, diff_vect.vector.y)
        diff_in_robot_0 = tf2_geometry_msgs.do_transform_vector3(diff_vect, known_map_to_robot_trans)
        #print('difference vector in robot0: ', diff_in_robot_0.vector.x, diff_in_robot_0.vector.y)
        twist = Twist()
        error_t = np.array([diff_in_robot_0.vector.x, diff_in_robot_0.vector.y])
        # t = rospy.Time.now().to_sec()
        # d_error_d_t = (error_t - self.prev_errors[robot_namespace]) / (t - self.prev_ts[robot_namespace])
        avg_error = (error_t + self.error_min1s[robot_namespace] + self.error_min2s[robot_namespace]) / 3.0
        # print('dt: ', (t - self.prev_ts[robot_namespace]), 'error_t: ', error_t, ', prev_error: ', self.prev_errors[robot_namespace], ', d_error_d_t: ', d_error_d_t)
        cmd_vel = self.get_cmd_vel(avg_error)

        self.error_min2s[robot_namespace] = self.error_min1s[robot_namespace]
        self.error_min1s[robot_namespace] = error_t
        # self.prev_ts[robot_namespace] = t
        # print('x_vel: ', x_vel, ', y_vel: ', y_vel)
        twist.linear.x = cmd_vel[0]
        twist.linear.y = cmd_vel[1]
        # print('twist: ', twist)
        self.cmd_vel_pubs[robot_namespace].publish(twist)

        delta_x = np.sqrt(np.square(x_diff) + np.square(y_diff))
        if delta_x < 0.4:
            self.plan_indices[robot_namespace] += 1

        if len(self.plans[robot_namespace].plan.poses) <= self.plan_indices[robot_namespace]:
            self.plan_indices[robot_namespace] = 0
            self.plans[robot_namespace].plan.poses = np.flip(self.plans[robot_namespace].plan.poses, axis=0)
            # print('flipping plan')

        self.plan_publishers[robot_namespace].publish(self.plans_to_publish[robot_namespace])

    def get_global_plan(self, start, robot_namespace):
        # print('generating plan for ' + robot_namespace)
        goal = PoseStamped()
        goal.header.frame_id = "known_map"
        goal.header.stamp = rospy.Time.now()

        rand_region = self.goal_regions[np.random.randint(0, len(self.goal_regions))]
        x_pos_in_init_frame = np.random.randint(rand_region[0], rand_region[2])
        y_pos_in_init_frame = np.random.randint(rand_region[1], rand_region[3])
        goal.pose.position.x = x_pos_in_init_frame - self.world_transform[0]
        goal.pose.position.y = y_pos_in_init_frame - self.world_transform[1]
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        req = GetPlan()
        req.start = start
        req.goal = goal
        req.tolerance = self.tolerances[robot_namespace]
        # print('start of : ', req.start.pose.position.x, req.start.pose.position.y)
        # print('goal of : ', req.goal.pose.position.x, req.goal.pose.position.y)
        self.plans[robot_namespace] = self.get_plan(req.start, req.goal, req.tolerance)
        # print('plan has length of: ', len(self.plans[robot_namespace].plan.poses))
        pub_pose_array = PoseArray()
        pub_pose_array.header.frame_id = "known_map"
        pub_pose_array.header.stamp = rospy.Time.now()
        for i in range(0, len(self.plans[robot_namespace].plan.poses)):
            new_pose = Pose()
            new_pose.position.x = self.plans[robot_namespace].plan.poses[i].pose.position.x
            new_pose.position.y = self.plans[robot_namespace].plan.poses[i].pose.position.y
            new_pose.position.z = 0.5
            pub_pose_array.poses.append(new_pose)

        self.plans_to_publish[robot_namespace] = pub_pose_array

        # print('publishing this pose array: ', pub_pose_array)

        # print('get plan return plan: ', plan)
        # print("trying self.plan.respone: ", self.plan.response)
        # print("trying self.plan.plan: ", self.plan.plan)

    def get_cmd_vel(self, error_t):
        K_p = 0.5
        cmd_vel = K_p*error_t
        # delta_x_norm = np.sqrt(np.square(diff_in_robot_0.vector.x) + np.square(diff_in_robot_0.vector.y))
        # print('delta x norm: ', delta_x_norm)
        thresh = 0.50
        if cmd_vel[0] > thresh or cmd_vel[1] > thresh:
            return thresh * cmd_vel / (np.maximum(np.abs(cmd_vel[0]), np.abs(cmd_vel[1])))
        else:
            return cmd_vel

if __name__ == '__main__':
    try:
        rospy.init_node("follow_global_path", anonymous=True)
        num_obsts = rospy.get_param("~num_obsts")
        world = rospy.get_param("~world")
        start_xs = rospy.get_param("~start_xs")
        start_ys = rospy.get_param("~start_ys")
        #print("robot namespace: ", robot_namespace)
        Agent(num_obsts, world, start_xs, start_ys)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

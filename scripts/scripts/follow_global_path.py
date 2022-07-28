#!/usr/bin/env python2
import rospy
from nav_msgs.srv import GetPlan
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped, Twist, Vector3Stamped, PoseArray

import tf_conversions

import tf2_ros
import tf2_geometry_msgs

class Agent:
    def __init__(self, num_obsts, world, controller, seed, start_xs, start_ys):
        # /move_base for TEB
        # /move_base_virtual for DGap
        self.num_obsts = num_obsts
        self.world = world
        self.controller = controller
        self.seed = int(seed) + 100  # need to change seed or else start and goal will be the same
        # print('self.seed: ', self.seed)
        np.random.seed(self.seed)
        self.start_xs = start_xs
        self.start_ys = start_ys
        # print('self.controller: ', self.controller)
        if self.controller == "dynamic_gap":
            self.plan_topic = '/move_base_virtual/make_plan'
        elif self.controller == "teb":
            self.plan_topic = '/move_base/make_plan'
        else:
            print("CONTROLLER IS NOT ON LIST, NOT RUNNING PATHS")

        rospy.wait_for_service(self.plan_topic)
        self.get_plan = rospy.ServiceProxy(self.plan_topic, GetPlan)
        self.rate = rospy.Rate(10.0)
        self.world = world
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
        self.error_min3s = {}
        self.errors = {}
        self.ts = {}
        self.t_min1s = {}

        stripped_xs = self.start_xs[1:-1]
        self.split_xs = stripped_xs.split(',')
        stripped_ys = self.start_ys[1:-1]
        self.split_ys = stripped_ys.split(',')

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # bottom left to top right ((x, y) to (x,y)
        # these are in world / known_map
        self.campus_goal_regions = [[6, 8, 7, 16], [5, 15, 7, 18],[1, 28, 5, 30],[1, 16, 4, 22],
                                [4, 20, 11, 24],[13, 22, 16, 28],[16, 25, 29, 26],
                                [12, 18, 15, 22],[20, 18, 24, 22],[16, 12, 18, 17],
                                [19, 14, 23, 17],[25, 12, 29, 18],[10, 8, 17, 12],
                                [1, 6, 11, 8], [9, 4, 14, 6], [16, 1, 22, 8]]

        self.empty_goal_regions = [[18, 9, 19, 10]]

        self.empty_world_transform = [13.630, 13.499]
        self.campus_world_transform = [14.990204, 13.294787]

        # print('self.world: ', self.world)
        if self.world == "empty":
            self.world_transform = self.empty_world_transform
            self.goal_regions = self.empty_goal_regions
        elif self.world == "campus":
            self.world_transform = self.campus_world_transform
            self.goal_regions = self.campus_goal_regions

        for i in range(0, num_obsts):
            robot_namespace = "robot" + str(i)
            self.need_plan[robot_namespace] = True
            self.plan_indices[robot_namespace] = 0
            self.plan_publishers[robot_namespace] = rospy.Publisher(robot_namespace + "/global_path", PoseArray, queue_size=2)
            self.tolerances[robot_namespace] = 0.5
            start = self.get_start(i)
            self.get_global_plan(start, robot_namespace, i)

            self.odom_subs[robot_namespace] = rospy.Subscriber(robot_namespace + "/odom", Odometry, self.odom_CB, queue_size=2)
            self.cmd_vel_pubs[robot_namespace] = rospy.Publisher(robot_namespace + "/cmd_vel", Twist, queue_size=5)
            self.errors[robot_namespace] = np.array([0.0, 0.0])
            self.error_min1s[robot_namespace] = np.array([0.0, 0.0])
            self.error_min2s[robot_namespace] = np.array([0.0, 0.0])
            self.error_min3s[robot_namespace] = np.array([0.0, 0.0])
            self.ts[robot_namespace] = 1.0
            self.t_min1s[robot_namespace] = 0.0

    def get_start(self, i):
        new_start = [float(self.split_xs[i]), float(self.split_ys[i])]

        # print('pre_transform start: ', new_start)
        start = PoseStamped()
        start.header.frame_id = "known_map"
        start.header.stamp = rospy.Time.now()
        start.pose.position.x = new_start[0] - self.world_transform[0]
        start.pose.position.y = new_start[1] - self.world_transform[1]
        # print('post_transform start: ', start.pose.position.x, ', ', start.pose.position.y)
        start.pose.position.z = 0.0
        start.pose.orientation.w = 1.0

        return start

    def odom_CB(self, msg):
        ## Odom comes in as map_static, comes in as "PoseWithCovariance"
        robot_namespace = msg.child_frame_id
        
        # Desired poses are all in map_static
        desired_pose = self.plans[robot_namespace].poses[self.plan_indices[robot_namespace]]

        print('desired_pose for ' + robot_namespace + ': ' + str(desired_pose.pose.position.x) + ', ' + str(desired_pose.pose.position.y))
        print('current_pose ' + robot_namespace + ': ' + str(msg.pose.pose.position.x) + ', ' + str(msg.pose.pose.position.y))
        x_diff = msg.pose.pose.position.x - desired_pose.pose.position.x
        y_diff = msg.pose.pose.position.y - desired_pose.pose.position.y

        # calculate cmd_vel
        try:
            map_static_to_robot_trans = self.tfBuffer.lookup_transform(robot_namespace, "map_static", rospy.Time(), rospy.Duration(3.0))
        except tf2_ros.ExtrapolationException:
            self.rate.sleep()
            return

        diff_vect = Vector3Stamped()
        diff_vect.header.frame_id = "map_static"
        diff_vect.header.stamp = rospy.Time.now()
        diff_vect.vector.x = -x_diff
        diff_vect.vector.y = -y_diff
        #print('difference vector in known map: ', diff_vect.vector.x, diff_vect.vector.y)
        diff_in_robot_frame = tf2_geometry_msgs.do_transform_vector3(diff_vect, map_static_to_robot_trans)
        #print('difference vector in robot0: ', diff_in_robot_0.vector.x, diff_in_robot_0.vector.y)
        self.errors[robot_namespace] = np.array([diff_in_robot_frame.vector.x, diff_in_robot_frame.vector.y])

        avg_error = (self.errors[robot_namespace] +
                     self.error_min1s[robot_namespace] + 
                     self.error_min2s[robot_namespace] +
                     self.error_min3s[robot_namespace]) / 4.0

        cmd_vel = self.get_cmd_vel(avg_error)
        twist = Twist()
        twist.linear.x = cmd_vel[0]
        twist.linear.y = cmd_vel[1]        
        
        self.cmd_vel_pubs[robot_namespace].publish(twist)

        
        self.error_min3s[robot_namespace] = self.error_min2s[robot_namespace]
        self.error_min2s[robot_namespace] = self.error_min1s[robot_namespace]
        self.error_min1s[robot_namespace] = self.errors[robot_namespace]

        print('command vel for ' + robot_namespace + ': ' + str(twist.linear.x) + ', y_vel: ' + str(twist.linear.y))
        # print('twist: ', twist)

        delta_x = np.sqrt(np.square(x_diff) + np.square(y_diff))
        if delta_x < 0.4:
            self.plan_indices[robot_namespace] += 1

        if len(self.plans[robot_namespace].poses) <= self.plan_indices[robot_namespace]:
            self.plan_indices[robot_namespace] = 0
            self.plans[robot_namespace].poses = np.flip(self.plans[robot_namespace].poses, axis=0)
            print(robot_namespace + ' flipping plan')

        self.plan_publishers[robot_namespace].publish(self.plans_to_publish[robot_namespace])
        

    def get_global_plan(self, start, robot_namespace, i):

        goal = PoseStamped()
        goal.header.frame_id = "known_map"
        goal.header.stamp = rospy.Time.now()

        known_map_to_map_static = self.tfBuffer.lookup_transform("map_static", "known_map", rospy.Time(), rospy.Duration(3.0))

        '''
        if i == 0:
            x_pos_in_init_frame = 17.63
            y_pos_in_init_frame = 12
        else:
            x_pos_in_init_frame = 9.63
            y_pos_in_init_frame = 12
        '''

        rand_int = np.random.randint(0, len(self.goal_regions))
        # print('rand_int: ', rand_int)
        rand_region = self.goal_regions[rand_int]
        # print('rand_region: ', rand_region)
        x_pos_in_init_frame = np.random.randint(rand_region[0], rand_region[2])
        y_pos_in_init_frame = np.random.randint(rand_region[1], rand_region[3])

        # print('x_pos: ', x_pos_in_init_frame)
        # print('y_pos: ', y_pos_in_init_frame)
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
        plan_in_known_map = self.get_plan(req.start, req.goal, req.tolerance)
        # print('plan has length of: ', len(plan_in_known_map.plan.poses))
        pub_pose_array = PoseArray()
        pub_pose_array.header.frame_id = "known_map"
        pub_pose_array.header.stamp = rospy.Time.now()
        for i in range(0, len(plan_in_known_map.plan.poses)):
            new_pose = PoseStamped()
            new_pose.header.frame_id = "known_map"
            new_pose.header.stamp = rospy.Time.now()
            new_pose.pose.position.x = plan_in_known_map.plan.poses[i].pose.position.x
            new_pose.pose.position.y = plan_in_known_map.plan.poses[i].pose.position.y
            new_pose.pose.position.z = 0.5
            # print('original plan pose (known_map): ', new_pose.pose.position.x, ', ', new_pose.pose.position.y, ', ', new_pose.pose.position.z)
            new_pose_in_map_static = tf2_geometry_msgs.do_transform_pose(new_pose, known_map_to_map_static)
            pub_pose_array.poses.append(new_pose_in_map_static)

        # print('transformed plan pose (map_static): ', pub_pose_array.poses)
        self.plans_to_publish[robot_namespace] = pub_pose_array
        self.plans[robot_namespace] = pub_pose_array
        # print('publishing this pose array: ', pub_pose_array)

        # print('get plan return plan: ', plan)
        # print("trying self.plan.respone: ", self.plan.response)
        # print("trying self.plan.plan: ", self.plan.plan)

    def get_cmd_vel(self, error_t):
        K_p = 0.6         # 0.75 for ~0.3 m/s

        cmd_vel = K_p*error_t

        thresh = 0.50
        if np.abs(cmd_vel[0]) > thresh or np.abs(cmd_vel[1]) > thresh:
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
        controller = rospy.get_param("~controller")
        seed = rospy.get_param("~seed")
        #print("robot namespace: ", robot_namespace)
        Agent(num_obsts, world, controller, seed, start_xs, start_ys)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

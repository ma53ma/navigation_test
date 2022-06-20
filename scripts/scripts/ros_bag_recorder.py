
import os
import rosbag
import rospy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Quaternion
from rosgraph_msgs.msg import Log
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
import threading
import datetime

class RosbagRecorder:
    def __init__(self, taskid, num_obsts):
        self.bag_closed = False
        self.num_obsts = num_obsts
        self.lock = threading.Lock()
        bagpath = "~/simulation_data/bagfile/" + str(datetime.datetime.now()) + "_" + str(taskid) + ".bag"
        self.bag_file_path = os.path.expanduser(bagpath)
        # print("bag file = " + self.bag_file_path + "\n")
        self.bag_file = rosbag.Bag(f=self.bag_file_path, mode='w', compression=rosbag.Compression.LZ4)
        self.scan_data = None
        self.tf_data = None
        self.exe_traj_sub = rospy.Subscriber("pg_traj", PoseArray, self.exe_traj_cb, queue_size = 2)
        self.score_sub = rospy.Subscriber("traj_score", MarkerArray, self.score_cb, queue_size = 2)
        self.traj_sub = rospy.Subscriber("all_traj_vis", MarkerArray, self.traj_cb, queue_size = 2)
        self.scan_sub = rospy.Subscriber("point_scan", LaserScan, self.scan_cb, queue_size = 2)
        self.tf_sub = rospy.Subscriber("tf", TFMessage, self.tf_cb, queue_size = 1000)  # we need robot1_laser_0 frame
        self.map_sub = rospy.Subscriber("map", OccupancyGrid, self.map_cb, queue_size = 5)
        self.dg_model_pos_sub = rospy.Subscriber("dg_model_pos", MarkerArray, self.dg_model_pos_cb, queue_size=5)
        self.dg_model_vel_sub = rospy.Subscriber("dg_model_vel", MarkerArray, self.dg_model_vel_cb, queue_size=5)
        self.po_dir_sub = rospy.Subscriber("po_dir", Marker, self.po_dir_cb, queue_size=5)
        self.pg_arcs_sub = rospy.Subscriber("pg_arcs", MarkerArray, self.pg_arcs_cb, queue_size=5)
        self.pg_sides_sub = rospy.Subscriber("pg_sides", MarkerArray, self.pg_sides_cb, queue_size=5)
        self.reachable_gap_sub = rospy.Subscriber("reachable_gaps", MarkerArray, self.reachable_gap_cb, queue_size=5)

        self.agent_odom_subs = {}
        for i in range(num_obsts):
            self.agent_odom_subs[i] = rospy.Subscriber("robot" + str(i) + "/odom", Odometry, self.agent_odom_cb, queue_size=5)
        self.ego_odom_sub = rospy.Subscriber("robot" + str(self.num_obsts) + "/odom", Odometry, self.ego_odom_cb, queue_size=5)

    def exe_traj_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("pg_traj", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("Exe Traj written")

    def score_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("traj_score", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("Score written")

    def traj_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("all_traj_vis", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("Traj written")

    def scan_cb(self, data):
        if self.bag_closed:
            return
        self.lock.acquire()
        self.scan_data = data
        self.bag_file.write("point_scan", data, data.header.stamp)
        self.lock.release()
        rospy.logdebug("Laserscan written")

    def tf_cb(self, data):
        if self.bag_closed:
            return
        self.lock.acquire()
        self.tf_data = data
        self.bag_file.write("tf", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("tf written")

    def ego_odom_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        robot_namespace = data.child_frame_id
        marker = self.create_marker(data, 0.0, 0.0, 1.0)
        self.bag_file.write(robot_namespace + "/marker", marker, marker.header.stamp)

        self.lock.release()
        rospy.logdebug(robot_namespace + " odom marker written")

    def agent_odom_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        robot_namespace = data.child_frame_id
        marker = self.create_marker(data, 1.0, 0.0, 0.0)
        self.bag_file.write(robot_namespace + "/marker", marker, marker.header.stamp)

        self.lock.release()
        rospy.logdebug(robot_namespace + " odom marker written")

    def create_marker(self, data, r, g, b):
        robot_namespace = data.child_frame_id
        marker = Marker()
        marker.header = data.header
        marker.ns = robot_namespace
        marker.id = 0
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.pose.orientation.w = 1

        marker.pose.position.x = data.pose.pose.position.x
        marker.pose.position.y = data.pose.pose.position.y
        marker.pose.position.z = data.pose.pose.position.z

        marker.lifetime = rospy.Duration(0)
        marker.scale.x = 2 * 0.2
        marker.scale.y = 2 * 0.2
        marker.scale.z = 0.000001
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        return marker

    def map_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        # print('original map header: ', data.header)
        data.header.seq = 0
        data.header.stamp = self.tf_data.transforms[0].header.stamp
        data.header.frame_id = "known_map"
        # print('revised map header: ', data.header)
        self.bag_file.write("map", data, data.header.stamp)
        self.lock.release()
        rospy.logdebug("map written")

    def dg_model_pos_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("dg_model_pos", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("dg_model_pos written")

    def dg_model_vel_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("dg_model_vel", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("dg_model_vel written")

    def po_dir_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("po_dir", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("po_dir written")

    def pg_arcs_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("pg_arcs", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("pg_arcs written")

    def pg_sides_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("pg_sides", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("pg_sides written")

    def reachable_gap_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bag_file.write("reachable_gaps", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("reachable_gaps written")

    def done(self, collided):
        self.bag_closed = True
        self.lock.acquire()

        self.map_sub.unregister()
        self.pg_arcs_sub.unregister()
        self.dg_model_pos_sub.unregister()
        self.dg_model_vel_sub.unregister()
        self.po_dir_sub.unregister()
        self.reachable_gap_sub.unregister()

        self.scan_sub.unregister()
        self.tf_sub.unregister()
        self.score_sub.unregister()
        self.traj_sub.unregister()
        self.ego_odom_sub.unregister()
        self.exe_traj_sub.unregister()

        for i in range(self.num_obsts):
            self.agent_odom_subs[i].unregister()

        self.bag_file.close()
        self.lock.release()
        rospy.logdebug("Result finished")

        if not collided:
            os.remove(self.bag_file_path)

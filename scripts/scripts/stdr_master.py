#!/usr/bin/env python

import subprocess
import multiprocessing as mp
import os
import sys
import rospkg
import roslaunch
import time
# import nav_scripts.movebase_driver as test_driver
import rosgraph
import threading
import Queue
from ctypes import c_bool
import rospkg
rospack = rospkg.RosPack()

import math
from move_base_msgs.msg import *
from sensor_msgs.msg import Range
import signal
import actionlib
from actionlib_msgs.msg import GoalStatus

import socket
import contextlib
import tf
import numpy as np
import nav_msgs.srv

from stdr_testing_scenarios import TestingScenarios

from actionlib_msgs.msg import GoalStatusArray
from nav_msgs.msg import Odometry
import rospy
import csv
import datetime
import tf2_ros
from stdr_testing_scenarios import SectorScenario, CampusScenario, FourthFloorScenario, SparseScenario, EmptyScenario
import rosbag
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan

from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Quaternion
from rosgraph_msgs.msg import Log

class BumperChecker:
    def __init__(self, num_obsts):
        self.sub = rospy.Subscriber("robot" + str(num_obsts) + "/bumpers", Range, self.bumperCB, queue_size=5)
        self.mod_sub = rospy.Subscriber("robot" + str(num_obsts) + "/mod_bumpers", Range, self.mod_bumperCB, queue_size=5)

        self.collided = False

    def bumperCB(self, data):
        if data.range < 0:
            print('~~~~~~~~~~~ NORMAL BUMPER COLLISION ~~~~~~~~~~~~~~')
            self.collided = True

    def mod_bumperCB(self, data):
        if data.range < 0:
            print('~~~~~~~~~~~ MOD BUMPER COLLISION ~~~~~~~~~~~~~~')
            self.collided = True

class OdomAccumulator:
    def __init__(self, num_obsts):
        self.feedback_subscriber = rospy.Subscriber("robot" + str(num_obsts) + "/odom", Odometry, self.odomCB, queue_size=5)
        self.path_length = 0
        self.prev_msg = None

    def odomCB(self, odom):
        if self.prev_msg is not None:
            prev_pos = self.prev_msg.pose.pose.position
            cur_pos = odom.pose.pose.position

            deltaX = cur_pos.x - prev_pos.x
            deltaY = cur_pos.y - prev_pos.y

            displacement = math.sqrt(deltaX*deltaX + deltaY*deltaY)
            self.path_length += displacement
        self.prev_msg = odom

    def getPathLength(self):
        return self.path_length

class ResultRecorder:
    def __init__(self, taskid):
        self.lock = threading.Lock()
        self.tf_sub = rospy.Subscriber("tf", TFMessage, self.tf_cb, queue_size = 1000)
        self.scan_sub = rospy.Subscriber("point_scan", LaserScan, self.scan_cb, queue_size = 5)
        self.score_sub = rospy.Subscriber("traj_score", MarkerArray, self.score_cb, queue_size = 5)
        self.traj_sub = rospy.Subscriber("all_traj_vis", MarkerArray, self.traj_cb, queue_size = 5)
        self.exe_traj_sub = rospy.Subscriber("dg_traj", PoseArray, self.exe_traj_cb, queue_size = 5)

        bagpath = "~/simulation_data/bagfile/" + str(datetime.datetime.now()) + "_" + str(taskid) + ".bag"
        self.bagfilepath = os.path.expanduser(bagpath)
        print("bag file = " + self.bagfilepath + "\n")
        self.bagfile = rosbag.Bag(f=self.bagfilepath, mode='w', compression=rosbag.Compression.LZ4)
        self.scan_data = None
        self.tf_data = None
        self.bag_closed = False

    def exe_traj_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bagfile.write("dg_traj", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("Exe Traj written")

    def score_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bagfile.write("traj_score", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("Score written")

    def traj_cb(self, data):
        if self.tf_data is None or self.bag_closed:
            return
        self.lock.acquire()
        self.bagfile.write("all_traj_vis", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("Traj written")

    def scan_cb(self, data):
        if self.bag_closed:
            return
        self.lock.acquire()
        self.scan_data = data
        self.bagfile.write("point_scan", data, data.header.stamp)
        self.lock.release()
        rospy.logdebug("Laserscan written")

    def tf_cb(self, data):
        if self.bag_closed:
            return
        self.lock.acquire()
        self.tf_data = data
        self.bagfile.write("tf", data, self.tf_data.transforms[0].header.stamp)
        self.lock.release()
        rospy.logdebug("tf written")

    def done(self):
        self.bag_closed = True
        self.lock.acquire()
        self.scan_sub.unregister()
        self.tf_sub.unregister()
        self.score_sub.unregister()
        self.traj_sub.unregister()
        self.bagfile.close()
        self.lock.release()
        rospy.logdebug("Result finished")

def port_in_use(port):
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            print "Port " + str(port) + " is in use"
            return True
        else:
            print "Port " + str(port) + " is not in use"
            return False


class MultiMasterCoordinator:
    def __init__(self, num_masters=4, record=False):
        signal.signal(signal.SIGINT, self.signal_shutdown)
        signal.signal(signal.SIGTERM, self.signal_shutdown)
        self.children_shutdown = mp.Value(c_bool, False)
        self.soft_shutdown = mp.Value(c_bool, False)

        self.should_shutdown = False
        self.num_masters = num_masters

        self.save_results = True
        self.task_queue_capacity = 2000 #2*self.num_masters
        self.task_queue = mp.JoinableQueue(maxsize=self.task_queue_capacity)
        self.result_queue_capacity = 2000 #*self.num_masters
        self.result_queue = mp.JoinableQueue(maxsize=self.result_queue_capacity)
        self.stdr_masters = []
        self.result_list = []
        self.stdr_launch_mutex = mp.Lock()
        self.record = record

        self.fieldnames = ["controller"]
        self.fieldnames.extend(TestingScenarios.getFieldNames())
        self.fieldnames.extend(["taskid","world", "pid", "result","time","path_length","robot"])
        self.fieldnames.extend(["fov", "radial_extend", "projection", "r_min", 'r_norm', 'k_po', 'reduction_threshold', 'reduction_target'])
        self.fieldnames.extend(["sim_time", "obstacle_cost_mode", "sum_scores"])
        self.fieldnames.extend(["bag_file_path",'converter', 'costmap_converter_plugin', 'global_planning_freq', 'feasibility_check_no_poses', 'simple_exploration', 'weight_gap', 'gap_boundary_exponent', 'egocircle_early_pruning', 'gap_boundary_threshold', 'gap_boundary_ratio', 'feasibility_check_no_tebs', 'gap_exploration', 'gap_h_signature', ])
        self.world_queue = []

    def start(self):
        self.startResultsProcessing()
        self.startProcesses()

    def startResultsProcessing(self):
        self.result_thread = threading.Thread(target=self.processResults,args=[self.result_queue])
        self.result_thread.daemon=True
        self.result_thread.start()

    def startProcesses(self):
        self.ros_port = 11411
        self.gazebo_port = self.ros_port + 100
        for ind in xrange(self.num_masters):
            self.addProcess(ind)

    def addProcess(self, ind):
        while port_in_use(self.ros_port):
            self.ros_port += 1

        while port_in_use(self.gazebo_port):
            self.gazebo_port += 1

        stdr_master = STDRMaster(
            self.task_queue,
            self.result_queue,
            self.children_shutdown,
            self.soft_shutdown,
            self.ros_port,
            self.gazebo_port,
            stdr_launch_mutex=self.stdr_launch_mutex,
            record=self.record)
        stdr_master.start()
        self.stdr_masters.append(stdr_master)

        self.ros_port += 1
        self.gazebo_port += 1

        time.sleep(1)


    def processResults(self,queue):

        outputfile_name = "~/simulation_data/results_" + str(datetime.datetime.now())
        #outputfile_name = "/data/fall2018/chapter_experiments/chapter_experiments_" + str(datetime.datetime.now())
        outputfile_name = os.path.expanduser(outputfile_name)

        with open(outputfile_name, 'wb') as csvfile:
            seen = set()
            fieldnames = [x for x in self.fieldnames if not (x in seen or seen.add(x))] #http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html

            datawriter = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='', extrasaction='ignore')
            datawriter.writeheader()

            while not self.should_shutdown: #This means that results stop getting saved to file as soon as I try to kill it
                try:
                    task = queue.get(block=False)

                    result_string = "Result of ["
                    for k,v in task.iteritems():
                        #if "result" not in k:
                            result_string+= str(k) + ":" + str(v) + ","
                    result_string += "]"

                    print result_string

                    if "error" not in task:
                        self.result_list.append(result_string)
                        if self.save_results:
                            datawriter.writerow(task)
                            csvfile.flush()
                    else:
                        del task["error"]
                        self.task_queue.put(task)
                        self.addProcess()

                    #print "Result of " + task["world"] + ":" + task["controller"] + "= " + str(task["result"])
                    queue.task_done()
                except Queue.Empty, e:
                    #print "No results!"
                    time.sleep(1)

    def signal_shutdown(self,signum,frame):
        self.shutdown()

    def shutdown(self):
        with self.children_shutdown.get_lock():
            self.children_shutdown.value = True

        for process in mp.active_children():
            process.join()

        self.should_shutdown = True

    def waitToFinish(self):
        print "Waiting until everything done!"
        self.task_queue.join()
        print "All tasks processed!"
        with self.soft_shutdown.get_lock():
            self.soft_shutdown.value = True

        #The problem is that this won't happen if I end prematurely...
        self.result_queue.join()
        print "All results processed!"

    # This list should be elsewhere, possibly in the configs package
    def addTasks(self):
        worlds = ['campus_laser']  #["dense_laser", "campus_laser", "sector_laser", "office_laser"] # "dense_laser", "campus_laser", "sector_laser", "office_laser"
        fovs = ['360'] #['90', '120', '180', '240', '300', '360']
        seeds = list(range(25))
        controllers = ['dynamic_gap'] # ['dynamic_gap', 'teb']
        pi_selection = ['3.14159']
        taskid = 0

        # Nonholonomic dynamic Gap Experiments
        # for world in ["office_laser"]:
        for world in worlds:
            self.world_queue.append(world)
            for robot in ['holonomic']:
                for controller in controllers:
                    for fov in fovs:
                        for seed in seeds:
                            task = {
                                    'controller' : controller,
                                    'robot' : robot,
                                    'world' : world,
                                    'fov' : fov,
                                    'seed' : seed,
                                    "taskid" : taskid,
                                    'controller_args': {
                                        'far_feasible' : str(world != 'office_laser'),
                                        'holonomic' : str(robot == 'holonomic')
                                        }
                                    }
                            taskid += 1
                            self.task_queue.put(task)


class STDRMaster(mp.Process):
    def __init__(self,
    task_queue,
    result_queue,
    kill_flag,
    soft_kill_flag,
    ros_port,
    stdr_port,
    stdr_launch_mutex,
    record, **kwargs):
        super(STDRMaster, self).__init__()
        self.agent_global_path_manager_parent = None
        self.daemon = False

        self.task_queue = task_queue
        self.result_queue = result_queue
        self.ros_port = ros_port
        self.stdr_port = stdr_port
        self.stdr_launch_mutex = stdr_launch_mutex
        self.core = None
        self.stdr_launch = None
        self.controller_launch = None
        self.spawner_launch = None
        self.stdr_driver = None
        self.current_world = None
        self.kill_flag = kill_flag
        self.soft_kill_flag = soft_kill_flag
        self.is_shutdown = False
        self.had_error = False
        self.record = record

        self.first = True

        self.tfBuffer = None

        self.gui = True
        self.world_queue = []
        self.dynamic_obstacles = True
        self.agent_launch = []
        self.obstacle_spawns = []
        self.obstacle_goals = []
        self.obstacle_backup_goals = []
        self.valid_regions = []
        self.agent_bounds = []
        self.num_obsts = 0
        self.rosout_msg = ""
        self.trans = [0.0, 0.0]
        self.new_goal_list = np.zeros(self.num_obsts)

        print "New master"

        self.ros_master_uri = "http://localhost:" + str(self.ros_port)
        self.stdr_master_uri = "http://localhost:" + str(self.stdr_port)
        os.environ["ROS_MASTER_URI"] = self.ros_master_uri
        os.environ["STDR_MASTER_URI"]= self.stdr_master_uri

        # self.rosout_sub = rospy.Subscriber('/rosout_agg', Log, self.rosoutCB, queue_size=5)

        #if 'SIMULATION_RESULTS_DIR' in os.environ:

        if self.gui==False:
            if 'DISPLAY' in os.environ:
                del os.environ['DISPLAY']   #To ensure that no GUI elements of gazebo activated
        else:
            if 'DISPLAY' not in os.environ:
                os.environ['DISPLAY']=':0'

    def rosoutCB(self, data):
        if len(data.msg) > 24:
            print('data msg: ', data.msg[0:24])
            self.rosout_msg = data.msg[0:24]

    def run(self):
        while not self.is_shutdown and not self.had_error:
            self.process_tasks()
            time.sleep(5)
            if not self.is_shutdown:
                print >> sys.stderr, "(Not) Relaunching on " + str(os.getpid()) + ", ROS_MASTER_URI=" + self.ros_master_uri
        print "Run totally done"

    # called ONCE
    def process_tasks(self):
        print('PROCESS TASKS')
        # if self.tfBuffer is None:
        #     self.tfBuffer = tf2_ros.Buffer()
        #     self.listener = tf2_ros.TransformListener(self.tfBuffer)
        # this is starting roscore
        self.roslaunch_core()
        # rospy.set_param('/use_sim_time', 'True')
        rospy.init_node('test_driver', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        # scenarios = TestingScenarios()

        self.had_error = False
        path = rospack.get_path("dynamic_gap")
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, True)


        while not self.is_shutdown and not self.had_error:
            # TODO: If fail to run task, put task back on task queue
            try:
                #for world in self.world_queue:
                #    self.roslaunch_stdr(world)


                task = self.task_queue.get(block=False)
                # scenario = scenarios.getScenario(task)

                if True:

                    (start, goal) = self.generate_start_goal(task)
                    self.move_robot(start)  # relocating robot to start position
                    self.roslaunch_stdr(task) #pass in world info, start STDR world with dynamic obstacles, want to only run ONCE
                    time.sleep(5)

                    # rospy.sleep(5)
                    # try:
                    # except:
                    #     print "move robot failed"
                    #     self.roslaunch_stdr(task) #pass in world info
                    #     self.move_robot(start)


                    if task["controller"] is None:
                        result = "nothing"
                    elif not self.stdr_launch._shutting_down:
                        # controller_args = task["controller_args"] if "controller_args" in task else {}

                        try:
                            controller_args = task["controller_args"] if "controller_args" in task else {}
                            # scenario.setupScenario()
                            print("Upper level")
                            print(controller_args)
                            print(task["robot"])
                            print(task["controller"])

                            fov = "GM_PARAM_RBT_FOV"
                            seed_fov = str(task['fov'])
                            os.environ[fov] = seed_fov
                            self.roslaunch_controller(task["robot"], task["controller"], controller_args)

                            cli_args = [path + "/launch/agent_global_path_manager.launch",
                                                'num_obsts:=' + str(self.num_obsts),
                                                'world:=' + str(task["world"])]
                            roslaunch_args = cli_args[1:]
                            roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

                            self.agent_global_path_manager_parent = roslaunch.parent.ROSLaunchParent(
                                run_id=uuid, roslaunch_files=roslaunch_file,
                                is_core=False, port=self.ros_port  # , roslaunch_strs=controller_args
                            )
                            self.agent_global_path_manager_parent.start()
                            rospy.sleep(5.0)

                            task.update(controller_args)    #Adding controller arguments to main task dict for easy logging

                            print("Running test...")

                            #master = rosgraph.Master('/mynode')
                            #TODO: make this a more informative type
                            time.sleep(5)
                            # running a single test
                            result = self.run_test(goal_pose=goal, record=self.record, taskid=task["taskid"], num_obsts=self.num_obsts)

                            if self.spawner_launch is not None:
                                self.spawner_launch.shutdown()

                            self.controller_launch.shutdown()

                            # self.deleter_launch.shutdown()
                            # possibly delete ego robot?
                            for i in range(0, len(self.agent_launch)):
                                self.agent_launch[i].shutdown()
                            self.agent_launch = []


                        except rospy.ROSException as e:
                            result = "ROSException: " + str(e)
                            task["error"]= True
                            self.had_error = True

                        if self.spawner_launch is not None:
                            self.spawner_launch.shutdown()

                        self.controller_launch.shutdown()

                        for i in range(0, len(self.agent_launch)):
                            self.agent_launch[i].shutdown()
                        self.agent_launch = []


                    else:
                        result = "gazebo_crash"
                        task["error"] = True
                        self.had_error = True

                else:
                    result = "bad_task"

                if isinstance(result, dict):
                    task.update(result)
                else:
                    task["result"] = result
                task["pid"] = os.getpid()
                self.return_result(task)

                if self.had_error:
                    print(sys.stderr, result)


            except Queue.Empty, e:
                with self.soft_kill_flag.get_lock():
                    if self.soft_kill_flag.value:
                        self.shutdown()
                        print("Soft shutdown requested")
                time.sleep(1)


            with self.kill_flag.get_lock():
                if self.kill_flag.value:
                    self.shutdown()

        print("Done with processing, killing launch files...")
        # It seems like killing the core should kill all of the nodes,
        # but it doesn't
        if self.stdr_launch is not None:
            self.stdr_launch.shutdown()

        if self.spawner_launch is not None:
            self.spawner_launch.shutdown()

        if self.controller_launch is not None:
            self.controller_launch.shutdown()

        for i in range(0, len(self.agent_launch)):
            if self.agent_launch[i] is not None:
                self.agent_launch[i].shutdown()
        self.agent_launch = []

        print("STDRMaster shutdown: killing core...")
        self.core.shutdown()
        #self.core.kill()
        #os.killpg(os.getpgid(self.core.pid), signal.SIGTERM)
        print("All cleaned up")

        # Send goal to Move base and receive result

    def run_test(self, goal_pose, record=False, taskid=0, num_obsts=0):
        print('CALLING RUN_TEST')
        bumper_checker = BumperChecker(num_obsts)
        odom_accumulator = OdomAccumulator(num_obsts)

        if record:
            result_recorder = ResultRecorder(taskid)

        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)  # dynamic_gap_egocircle_path_follower
        print("waiting for server")
        client.wait_for_server()
        print("Done!")

        goal = MoveBaseGoal()
        goal.target_pose = goal_pose
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = 'map_static'

        print("sending goal")
        client.send_goal(goal)
        print("waiting for result")
        #if self.dynamic_obstacles:
        #    self.roslaunch_obst_controller()

        r = rospy.Rate(5)
        start_time = rospy.Time.now()
        result = None
        keep_waiting = True
        counter = 0
        result = None
        while keep_waiting:
            try:
                state = client.get_state()
                if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
                    keep_waiting = False
                elif bumper_checker.collided:
                    keep_waiting = False
                    result = "BUMPER_COLLISION"
                elif rospy.Time.now() - start_time > rospy.Duration(600):
                    keep_waiting = False
                    result = "TIMED_OUT"
                else:
                    counter += 1
                    # if blocks, sim time cannot be 0
                    r.sleep()
            except:
                keep_waiting = "False"

        print(result)
        task_time = str(rospy.Time.now() - start_time)
        path_length = str(odom_accumulator.getPathLength())

        if record:
            print("Acquire Record Done")
            result_recorder.done()
            print("Acquired")

        if result is None:
            print("done!")
            print("getting goal status")
            print(client.get_goal_status_text())
            print("done!")
            print("returning state number")
            state = client.get_state()
            if state == GoalStatus.SUCCEEDED:
                result = "SUCCEEDED"
            elif state == GoalStatus.ABORTED:
                result = "ABORTED"
            elif state == GoalStatus.LOST:
                result = "LOST"
            elif state == GoalStatus.REJECTED:
                result = "REJECTED"
            elif state == GoalStatus.ACTIVE:
                result = "TIMED_OUT"
            else:
                result = "UNKNOWN"
        print(result)

        return {'result': result, 'time': task_time, 'path_length': path_length}

    def generate_start_goal(self, task):
        print('CALLING GENERATE START GOAL')
        # try:
        #     trans = self.tfBuffer.lookup_transform("map_static", 'world', rospy.Time())
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #     pass
        # print trans

        print(task["world"])
        trans = [0, 0]

        if task["world"] == "sector_laser":
            scenario = SectorScenario(task, "world")
            trans[0] = 10.217199
            trans[1] = 10.1176
        elif task["world"] == "office_laser":
            scenario = FourthFloorScenario(task, "world")
            trans[0] = 43.173023
            trans[1] = 30.696842
        elif task["world"] == "dense_laser":
            scenario = SparseScenario(task, "world")
            trans[0] = 9.955517
            trans[1] = 9.823917
        elif task["world"] == "campus_laser":
            scenario = CampusScenario(task, "world")
            self.trans[0] = 14.990204
            self.trans[1] = 13.294787
            self.valid_regions = scenario.valid_regions
            self.agent_bounds = [1, 29, 1, 29]
            #self.obstacle_spawns = scenario.obstacle_spawns
            #self.obstacle_goals = scenario.obstacle_goals
            # location [1,1] in map_static (need transform between map_static and known_map
        elif task["world"] == "empty_laser":
            scenario = EmptyScenario(task, "world")
            self.trans[0] = 13.630
            self.trans[1] = 13.499
            self.obstacles = scenario.obstacles
            self.agent_bounds = [5, 23, 5, 23]
            self.obstacle_spawns = scenario.obstacle_spawns
            self.obstacle_goals = scenario.obstacle_goals
            self.obstacle_backup_goals = scenario.obstacle_backup_goals
        if self.dynamic_obstacles:
            self.obstacle_goals = [x - self.trans for x in self.obstacle_goals]
            self.obstacle_backup_goals = [x - self.trans for x in self.obstacle_backup_goals]
            self.num_obsts = 5 #len(self.obstacle_spawns)
            #self.new_goal_list = np.zeros(self.num_obsts)

        start = scenario.getStartingPose()
        goal = scenario.getGoal()
        start.position.x += self.trans[0]
        start.position.y += self.trans[1]
        goal.pose.position.x += self.trans[0]
        goal.pose.position.y += self.trans[1]

        # add start and goal to obstacles list so that agents cannot spawn really close to rbt
        #self.obstacles.append([start.position.x - 2, start.position.y + 2, start.position.x + 2, start.position.y - 2])
        #self.obstacles.append([goal.position.x - 2, goal.position.y + 2, goal.position.x + 2, goal.position.y - 2])

        return (start, goal)

    def start_core(self):

        #env_prefix = "ROS_MASTER_URI="+ros_master_uri + " GAZEBO_MASTER_URI=" + gazebo_master_uri + " "

        my_command = "roscore -p " + str(self.ros_port)

        #my_env = os.environ.copy()
        #my_env["ROS_MASTER_URI"] = self.ros_master_uri
        #my_env["GAZEBO_MASTER_URI"] = self.gazebo_master_uri

        print "Starting core..."
        self.core = subprocess.Popen(my_command.split()) # preexec_fn=os.setsid
        print "Core started! [" + str(self.core.pid) + "]"

    def roslaunch_core(self):

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        #roslaunch.configure_logging(uuid)

        self.core = roslaunch.parent.ROSLaunchParent(
            run_id=uuid, roslaunch_files=[],
            is_core=True, port=self.ros_port
        )
        self.core.start()

    def roslaunch_controller(self, robot, controller_name, controller_args={}):

        #controller_path =
        print("RosLaunch controller")
        print(controller_args.items())

        rospack = rospkg.RosPack()
        path = rospack.get_path("dynamic_gap")

        # We'll assume Gazebo is launched are ready to go

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, True)
        #roslaunch.configure_logging(uuid)
        #print path

        #Remapping stdout to /dev/null
        # sys.stdout = open(os.devnull, "w")
        for key,value in controller_args.items():
            var_name = "GM_PARAM_"+ key.upper()
            value = str(value)
            os.environ[var_name] = value
            print("Setting environment variable [" + var_name + "] to '" + value + "'")
        print('controller launch file: ' + path + "/launch/" + controller_name + "_" + robot + "_controller.launch")


        cli_args = [path + "/launch/spawn_robot.launch",
                    'robot_namespace:=robot' + str(self.num_obsts),
                    'rbtx:=' + os.environ["GM_PARAM_RBT_X"],
                    'rbty:=' + os.environ["GM_PARAM_RBT_Y"],
                    'fov:=' + os.environ["GM_PARAM_RBT_FOV"]]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

        self.spawner_launch = roslaunch.parent.ROSLaunchParent(
            run_id=uuid, roslaunch_files=roslaunch_file,
            is_core=False, port=self.ros_port  # , roslaunch_strs=controller_args
        )
        self.spawner_launch.start()

        cli_args = [path + "/launch/" + controller_name + "_" + robot + "_controller.launch",
                    'robot_namespace:=robot' + str(self.num_obsts),
                    'robot_radius:=' + str(0.2)]

        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

        self.controller_launch = roslaunch.parent.ROSLaunchParent(
            run_id=uuid, roslaunch_files=roslaunch_file,
            is_core=False, port=self.ros_port #, roslaunch_strs=controller_args
        )
        self.controller_launch.start()
        # sys.stdout = sys.__stdout__

    def roslaunch_stdr(self, task):
        # print('CALLING ROSLAUNCH_STDR')
        if self.stdr_launch is not None:
            self.stdr_launch.shutdown()

        world = task["world"]
        fov = task["fov"]
        robot = task["robot"]

        # self.stdr_launch = world
        self.current_robot = robot

        map_num = "GM_PARAM_MAP_NUM"
        seed_num = str(task['seed'])
        os.environ[map_num] = seed_num
        print("Setting environment variable [" + map_num + "] to '" + seed_num + "'")

        fov = "GM_PARAM_RBT_FOV"
        seed_fov = str(task['fov'])
        os.environ[fov] = seed_fov

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, True)
        print(world)
        # launch_file_name = "stdr_" + robot + "_" + world + fov + "_world.launch"
        launch_file_name = "stdr_" + world + "_world.launch"
        path = rospack.get_path("dynamic_gap")
        # print('launch name: ', launch_file_name)

        with self.stdr_launch_mutex:
            print('world launch file: ', [path + "/launch/" + launch_file_name])
            self.stdr_launch = roslaunch.parent.ROSLaunchParent(
                run_id=uuid, roslaunch_files=[path + "/launch/" + launch_file_name],
                is_core=False #, roslaunch_strs=controller_args
            )
            self.stdr_launch.start()

        if self.dynamic_obstacles:
            with self.stdr_launch_mutex:
                path = rospack.get_path("dynamic_gap")
                print('num obsts: ', self.num_obsts)
                for i in range(0, self.num_obsts):
                    print('spawning robot' + str(i))
                    # spawn_msg = "robot" + str(i) + " moved to new pose"
                    ## ADDING ROBOT ##
                    #start = self.obstacle_spawns[i]
                    #goal = self.obstacle_goals[i]
                    # start = [1.0, 1.0]
                    start = self.agent_random_start_and_goal()
                    #os.system("rosrun stdr_robot robot_handler add " + path + "/stdr_robots/robots/agent.xml" + " " + str(
                    #        start[0]) + " " + str(start[1]) + " 0")
                    #os.system("rosrun stdr_robot robot_handler replace " + "robot" + str(i) + " " + str(
                    #    start[0]) + " " + str(start[1]) + " 0")
                    #while self.rosout_msg is not spawn_msg:
                    #    os.system("rosrun stdr_robot robot_handler replace " + "robot" + str(i) + " " + str(
                    #        start[0]) + " " + str(start[1]) + " 0")

                    # maybe add back check for spawning inside a wall?

                    ## GIVING GOAL ##
                    cli_args = [path + "/launch/spawn_robot.launch",
                                'robot_namespace:=robot' + str(i),
                                'rbtx:=' + str(start[0]),
                                'rbty:=' + str(start[1]),
                                'robot_file:=agent.xml']
                    roslaunch_args = cli_args[1:]
                    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

                    agent_spawn_parent = roslaunch.parent.ROSLaunchParent(
                        run_id=uuid, roslaunch_files=roslaunch_file,
                        is_core=False, port=self.ros_port  # , roslaunch_strs=controller_args
                    )
                    self.agent_launch.append(agent_spawn_parent)
                    self.agent_launch[i].start()
                    rospy.sleep(0.25)
                    #topic = "robot" + str(i) + "/move_base/status"

                    #rospy.Subscriber(topic, GoalStatusArray, self.obstacle_callback, topic, queue_size=5)

                    #self.send_goal(i, goal)
                        # give goal as well, what topic do you need to get feedback on if goal is accepted?

                    # spawn at 0,0
                    # while result = false, replace with random location. How to get result?

    def valid_random_pos(self, pos):
        for obstacle in self.obstacles:
            if obstacle[0] <= pos[0] <= obstacle[2] and obstacle[3] <= pos[1] <= obstacle[1]:
                # within obstacle
                return False
        return True

    def agent_random_start_and_goal(self):

        rand_region = self.valid_regions[np.random.randint(0, len(self.valid_regions))]
        start = [np.random.randint(rand_region[0], rand_region[2]),
                 np.random.randint(rand_region[3], rand_region[1])]
        '''
        while not self.valid_random_pos(start):
            start = [np.random.randint(self.agent_bounds[0], self.agent_bounds[1]),
                     np.random.randint(self.agent_bounds[2], self.agent_bounds[3])]
        # print("valid start: ", start)
        goal = [np.random.randint(self.agent_bounds[0], self.agent_bounds[1]),
                     np.random.randint(self.agent_bounds[2], self.agent_bounds[3])]
        while not self.valid_random_pos(goal):
            goal = [np.random.randint(self.agent_bounds[0], self.agent_bounds[1]),
                     np.random.randint(self.agent_bounds[2], self.agent_bounds[3])]
        goal[0] = goal[0] - self.trans[0]
        goal[1] = goal[1] - self.trans[1]
        '''
        # print("valid goal: ", goal)
        return start

    '''
    def obstacle_callback(self, msg, topic):
        robot_name = topic.split("/")[0]
        robot_id = int(robot_name[5:])
        if len(msg.status_list) > 0:
            status = msg.status_list[0].status
            # print('status for robot' + str(robot_id) + ": " + str(status))
            if status == GoalStatus.SUCCEEDED or status == GoalStatus.ABORTED or status == GoalStatus.LOST or status == GoalStatus.REJECTED:
                goal = [np.random.randint(self.agent_bounds[0], self.agent_bounds[1]),
                     np.random.randint(self.agent_bounds[2], self.agent_bounds[3])]
                while not self.valid_random_pos(goal):
                    goal = [np.random.randint(self.agent_bounds[0], self.agent_bounds[1]),
                     np.random.randint(self.agent_bounds[2], self.agent_bounds[3])]
                goal[0] = goal[0] - self.trans[0]
                goal[1] = goal[1] - self.trans[1]
                # print('valid goal: ', goal)
                self.send_goal(robot_id, goal)
            # need to change something here to add other goals

    def send_goal(self, i, goal):
        client = actionlib.SimpleActionClient('/robot' + str(i) + '/move_base', MoveBaseAction)
        # print("agent: waiting for server")
        client.wait_for_server()
        # print("agent: Done!")
        target_pose = PoseStamped()
        pose_msg = Pose()
        pose_msg.position.x = goal[0]
        pose_msg.position.y = goal[1]
        q = tf.transformations.quaternion_from_euler(0, 0, 0)
        pose_msg.orientation = Quaternion(*q)
        target_pose.pose = pose_msg
        goal = MoveBaseGoal()
        goal.target_pose = target_pose
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = 'known_map'
        client.send_goal(goal)
        # wait for result?
    '''
    def shutdown(self):
        self.is_shutdown = True

    # TODO: add conditional logic to trigger this
    def task_error(self, task):
        self.task_queue.put(task)
        self.task_queue.task_done()
        self.shutdown()

    def return_result(self,result):
        print "Returning completed task: " + str(result)
        self.result_queue.put(result)
        self.task_queue.task_done()


    def move_robot(self, location):
        print("Moving robot to")
        # print location

        # position
        os.environ["GM_PARAM_RBT_X"] = str(location.position.x)
        os.environ["GM_PARAM_RBT_Y"] = str(location.position.y)
        return


if __name__ == "__main__":
    master = MultiMasterCoordinator(1, record=False)
    start_time = time.time()
    master.start()
    master.addTasks()

    #master.singletask()
    master.waitToFinish()
    #rospy.spin()
    master.shutdown()
    end_time = time.time()
    print "Total time: " + str(end_time - start_time)
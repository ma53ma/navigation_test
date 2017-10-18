#!/usr/bin/env python

import rospy
import random
import sys, os, time
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion, Transform, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from copy import deepcopy

from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException, StaticTransformBroadcaster
import tf
#from pips_test import gazebo_driver

from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
import numpy as np

from gazebo_ros import gazebo_interface
import std_srvs.srv as std_srvs
  
import std_msgs.msg as std_msgs



#Copied from pips_test: gazebo_driver.py
# Load model xml from file
def load_model_xml(filename):
  if os.path.exists(filename):
      if os.path.isdir(filename):
          print "Error: file name is a path?", filename
          sys.exit(0)

      if not os.path.isfile(filename):
          print "Error: unable to open file", filename
          sys.exit(0)
  else:
      print "Error: file does not exist", filename
      sys.exit(0)

  f = open(filename,'r')
  model_xml = f.read()
  if model_xml == "":
      print "Error: file is empty", filename
      sys.exit(0)

  return model_xml

class GazeboDriver():
  # Copied from pips_test: gazebo_driver.py
  def barrel_points(self,xmin, ymin, xmax, ymax, grid_size, num_barrels):
    # Get a dense grid of points
    points = np.mgrid[xmin:xmax:grid_size, ymin:ymax:grid_size]
    points = points.swapaxes(0, 2)
    points = points.reshape(points.size / 2, 2)

    # Choose random indexes
    idx = self.random.sample(range(points.shape[0]), num_barrels)
    print idx

    # Generate offsets
    off = self.nprandom.rand(num_barrels, 2) * grid_size / 2.0

    # Compute barrel points
    barrels = points[idx] + off

    for barrel in barrels:
      yield barrel

  def statesCallback(self, data): #This comes in at ~100hz
    self.models = data


  def newScene(self):
    self.pause()
    self.resetRobot()
    self.moveBarrels(self.num_barrels)
    self.unpause()

  def setPose(self, model_name, pose):
    ## Check if our model exists yet
    if(model_name in self.models.name):
    
      state = ModelState(model_name=model_name, pose=pose)

      response = self.setModelState(state)

      if(response.success):
        rospy.loginfo("Successfully set model pose")
        return True
    
    rospy.loginfo("failed to set model pose")
    return False

  def pause(self):
    rospy.wait_for_service(self.pause_service_name)
    return self.pauseService()

  def unpause(self):
    rospy.wait_for_service(self.unpause_service_name)
    return self.unpauseService()

  def resetWorld(self):
    rospy.wait_for_service(self.reset_world_service_name)
    return self.resetWorldService()

  def setModelState(self, state):
    rospy.wait_for_service(self.set_model_state_service_name)
    return self.setModelStateService(state)

  def resetRobotImpl(self, pose):
    self.pause()
    p = Pose()
    p.position.x = pose[0]
    p.position.y = pose[1]
    p.position.z = pose[2]
    quaternion = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5])
    #print quaternion
    p.orientation.x = quaternion[0]
    p.orientation.y = quaternion[1]
    p.orientation.z = quaternion[2]
    p.orientation.w = quaternion[3]
    self.setPose('mobile_base', p)
    self.unpause()

  def resetOdom(self):
    self.odom_pub.publish()
    
  def moveRobot(self, pose):
    self.setPose(self.robotName, pose)

  def resetBarrels(self, n):
      name = None
      for i in range(n):
          name = "barrel{}".format(i)
          pose = self.poses[i]
          self.setPose(name, pose)
      
  #Adapted from pips_test: gazebo_driver.py
  def spawn_barrel(self, model_name, initial_pose):
    # Must be unique in the gazebo world - failure otherwise
    # Spawning on top of something else leads to bizarre behavior
    model_path = os.path.expanduser("~/.gazebo/models/first_2015_trash_can/model.sdf")
    model_xml = load_model_xml(model_path)
    robot_namespace = rospy.get_namespace()
    gazebo_namespace = "/gazebo"
    reference_frame = ""
    
    success = gazebo_interface.spawn_sdf_model_client(model_name, model_xml,
        robot_namespace, initial_pose, reference_frame, gazebo_namespace)

  def moveBarrelsTest(self,n, x, y):
    self.poses = []
    for i in range(n):
      name = "barrel{}".format(i)
      pose = Pose()
      pose.position.x = x[i]
      pose.position.y = y[i]
      pose.orientation.w = 1
      self.poses.append(pose)
      if not self.setPose(name, pose):
	    self.spawn_barrel(name, pose)
	
  def moveBarrels(self,n):
    self.poses = []
    for i, xy in enumerate(self.barrel_points(self.minx,self.miny,self.maxx,self.maxy,self.grid_spacing, n)):
      print i, xy
      name = "barrel{}".format(i)
      print name
      
      pose = Pose()
      pose.position.x = xy[0]
      pose.position.y = xy[1]

      pose.orientation.w = 1
      
      self.poses.append(pose)
      
      if not self.setPose(name,pose):
        self.spawn_barrel(name, pose)
      
  def shutdown(self):
    self.unpause()
    self.resetWorld()

  def run(self):
    rospy.spin()


  def reset(self, seed=None):
    if seed is not None:
      self.seed = seed
    self.random.seed(self.seed)
    self.nprandom = np.random.RandomState(self.seed)
  
  def getRandInt(self, lower, upper):
    a = range(lower, upper + 1)
    start = self.random.choice(a)
    a.remove(start)
    end = self.random.choice(a)
    output = [start, end]
    return output
  


  def __init__(self, as_node = True, seed=None):
    if as_node:
      rospy.init_node('gazebo_state_recorder')
    
      rospy.loginfo("gazebo_state_recorder node started")
    
    self.robotName = 'mobile_base'

    self.queue_size = 50
    self.num_barrels = 3
    self.minx = -3.5
    self.maxx = 0.5
    self.miny = 1.0
    self.maxy = 5.0
    self.grid_spacing = 1.0
    
    
    
    self.poses = []
    self.robotPose = Pose()


    self.random = random.Random()
    self.seed = 0
    self.random.seed(seed)
    self.nprandom = np.random.RandomState(seed)

    self.odom_pub = rospy.Publisher(
      '/mobile_base/commands/reset_odometry', std_msgs.Empty, queue_size=1)

    self.models = None
    
    self.set_model_state_service_name = 'gazebo/set_model_state'
    self.pause_service_name = 'gazebo/pause_physics'
    self.unpause_service_name = 'gazebo/unpause_physics'
    self.get_model_state_service_name = 'gazebo/get_model_state'
    self.reset_world_service_name = "gazebo/reset_world"


    #NOTE: removed all 'wait_for_service' statements here, so should probably
    # add them in front of each actual service call
    
    rospy.loginfo("Waiting for service...")
    #rospy.wait_for_service(self.get_model_state_service_name)
    self.setModelStateService = rospy.ServiceProxy(self.set_model_state_service_name, SetModelState)
    rospy.loginfo("Service found...")
    
    #rospy.wait_for_service(self.pause_service_name)
    self.pauseService = rospy.ServiceProxy(self.pause_service_name, std_srvs.Empty)
    rospy.loginfo("Service found...")
    
    self.resetWorldService = rospy.ServiceProxy(self.reset_world_service_name, std_srvs.Empty)
    rospy.loginfo("Service found...")

    #rospy.wait_for_service(self.unpause_service_name)
    self.unpauseService = rospy.ServiceProxy(self.unpause_service_name, std_srvs.Empty)
    rospy.loginfo("Service found...")
    
    self.stateSub = rospy.Subscriber('gazebo/model_states', ModelStates, self.statesCallback, queue_size=self.queue_size)
        #self.statePub = rospy.Publisher('gazebo_data', GazeboState, queue_size=self.queue_size)
    
    #self.resetWorldService()
    #self.unpauseService()
    
    #rospy.on_shutdown(self.shutdown)




  


if __name__ == '__main__':
  try:
    a = GazeboDriver()
    a.run()
  except rospy.ROSInterruptException:
    rospy.loginfo("exception")

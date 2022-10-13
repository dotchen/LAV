# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
from agents.navigation.agent import Agent
from agents.navigation.local_planner_behavior import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.types_behavior import Cautious, Aggressive, Normal
from agents.navigation.behavior_agent import BehaviorAgent
from agents.tools.misc import get_speed, positive, is_within_distance_ahead, is_within_distance, compute_distance
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

class Autopilot(BehaviorAgent):

    def __init__(self, vehicle, map_utils):
        super().__init__(vehicle, ignore_traffic_light=True, behavior='cautious')
        self.map_utils = map_utils

    def set_destination_waypoints(self, route, clean=False):
        self._local_planner.set_global_plan(route, clean)

    def set_destination_coarse(self, route, clean=False):
        route_trace = []
        for pointA, pointB in zip(route, route[1:]):
            waypointA = self._map.get_waypoint(pointA)
            waypointB = self._map.get_waypoint(pointB)
            
            route_trace += self._trace_route(waypointA, waypointB)

        self._local_planner.set_global_plan(route_trace, clean)

    def update_information(self, world):

        self.speed = get_speed(self.vehicle)
        self.speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self.speed_limit)
        self.direction = self._local_planner.target_road_option

        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW

        self.look_ahead_steps = int((self.speed_limit) / 10)

        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)
        if self.incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW

        self.is_at_traffic_light = self._vehicle.is_at_traffic_light()
        if self.ignore_traffic_light:
            self.light_state = "Green"
        else:
            # This method also includes stop signs and intersections.
            self.light_state = str(self.vehicle.get_traffic_light_state())


    def pedestrian_avoid_manager(self, location, waypoint):
        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self.direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._bh_is_pedestrian_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._bh_is_pedestrian_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._bh_is_pedestrian_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=60)
        
        return walker_state, walker, distance

    
    def _bh_is_vehicle_hazard(self, ego_wpt, ego_loc, vehicle_list,
                           proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):


        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        for target_vehicle in vehicle_list:

            target_vehicle_loc = target_vehicle.get_location()
            # If the object is not in our next or current lane it's not an obstacle

            if is_within_distance(target_vehicle_loc, ego_loc,
                                  self._vehicle.get_transform().rotation.yaw,
                                  6.0, 30, 0):
                return (True, target_vehicle, compute_distance(target_vehicle_loc, ego_loc))
            target_wpt = self._map.get_waypoint(target_vehicle_loc)
            if target_wpt.road_id != ego_wpt.road_id or \
                    target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=5)[0]
                if target_wpt.road_id != next_wpt.road_id or \
                        target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                    continue

            if is_within_distance(target_vehicle_loc, ego_loc,
                                  self._vehicle.get_transform().rotation.yaw,
                                  proximity_th, up_angle_th, low_angle_th):

                return (True, target_vehicle, compute_distance(target_vehicle_loc, ego_loc))

        return (False, None, -1)

    def _bh_is_pedestrian_hazard(self, ego_wpt, ego_loc, vehicle_list,
                           proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):
        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        for target_vehicle in vehicle_list:

            target_vehicle_loc = target_vehicle.get_location()
            # If the object is not in our next or current lane it's not an obstacle

            target_wpt = self._map.get_waypoint(target_vehicle_loc)
            if target_wpt.road_id != ego_wpt.road_id or \
                    target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=5)[0]
                if target_wpt.road_id != next_wpt.road_id or \
                        target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                    continue

            if not self.map_utils.is_point_on_sidewalk(target_vehicle_loc) and is_within_distance(target_vehicle_loc, ego_loc,
                                  self._vehicle.get_transform().rotation.yaw,
                                  proximity_th, up_angle_th, low_angle_th):

                return (True, target_vehicle, compute_distance(target_vehicle_loc, ego_loc))

        return (False, None, -1)
    
    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)
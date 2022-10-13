import numpy as np
from collections import deque
from .global_planner import GlobalRoutePlanner, RoadOption
from .global_planner_dao import GlobalRoutePlannerDAO


class LocalPlanner(object):
    def __init__(self, vehicle, resolution=15, threshold_before=2.5, threshold_after=5.0):

        # Max skip avoids misplanning when route includes both lanes.
        self._max_skip = 20
        self._threshold_before = threshold_before
        self._threshold_after = threshold_after

        self._vehicle = vehicle
        self._map = vehicle.get_world().get_map()
        self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(self._map, resolution))
        self._grp.setup()

        self._route = None
        self._waypoints_queue = deque(maxlen=20000)

        self.target = (None, None)
        self.checkpoint = (None, None)
        self.total_distance = float('inf')
        self.distance_to_goal = float('inf')
        self.distances = deque(maxlen=20000)

    def set_route(self, start, target):
        self._waypoints_queue.clear()

        self._route = self._grp.trace_route(start, target)

        self.distance_to_goal = 0.0

        prev = None

        for node in self._route:
            self._waypoints_queue.append(node)

            cur = node[0].transform.location

            if prev is not None:
                delta = np.sqrt((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2)

                self.distance_to_goal += delta
                self.distances.append(delta)

            prev = cur

        self.target = self._waypoints_queue[0]
        self.checkpoint = (
                self._map.get_waypoint(self._vehicle.get_location()),
                RoadOption.LANEFOLLOW)

        self.total_distance = self.distance_to_goal

    def run_step(self):
        assert self._route is not None

        u = self._vehicle.get_transform().location
        max_index = -1

        for i, (node, command) in enumerate(self._waypoints_queue):
            if i > self._max_skip:
                break

            v = node.transform.location
            distance = np.sqrt((u.x - v.x) ** 2 + (u.y - v.y) ** 2)

            if int(self.checkpoint[1]) == 4 and int(command) != 4:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            if distance < threshold:
                self.checkpoint = (node, command)
                max_index = i

        for i in range(max_index + 1):
            if self.distances:
                self.distance_to_goal -= self.distances[0]
                self.distances.popleft()

            self._waypoints_queue.popleft()

        if len(self._waypoints_queue) > 0:
            self.target = self._waypoints_queue[0]

    def calculate_timeout(self, fps=20):
        _numpy = lambda p: np.array([p.transform.location.x, p.transform.location.y])

        distance = 0
        node_prev = None

        for node_cur, _ in self._route:
            if node_prev is None:
                node_prev = node_cur

            distance += np.linalg.norm(_numpy(node_cur) - _numpy(node_prev))
            node_prev = node_cur

        timeout_in_seconds = ((distance / 1000.0) / 10.0) * 3600.0
        timeout_in_frames = timeout_in_seconds * fps

        return timeout_in_frames

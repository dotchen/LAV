from __future__ import print_function
import math
import itertools
import numpy.random as random

import carla
import py_trees

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.weather_sim import WeatherBehavior
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.timer import GameTime

from leaderboard.scenarios.route_scenario import convert_transform_to_location, convert_json_to_transform, convert_json_to_transform
from leaderboard.utils.route_manipulation import location_route_to_gps, _get_latlon_ref
from leaderboard.utils.route_parser import RouteParser
from .train_scenario import TrainScenario

class NoCrashTrainScenario(TrainScenario):
    category = "TrainScenario"
    
    def __init__(self, world, config, debug_mode=0, criteria_enable=True):

        # Overwrite
        self.config = config
        self.list_scenarios = []
        
        # Set route
        self._set_route()

        ego_vehicle = self._update_ego_vehicle()
        
        BasicScenario.__init__(self, name=config.name,
                                            ego_vehicles=[ego_vehicle],
                                            config=config,
                                            world=world,
                                            debug_mode=debug_mode>1,
                                            terminate_on_failure=False,
                                            criteria_enable=criteria_enable)

        # Use dynamic weather behavior
        for behavior in self.scenario.scenario_tree.children:
            if isinstance(behavior, WeatherBehavior):
                self.scenario.scenario_tree.replace_child(behavior, DynamicWeatherBehavior())
                break
        
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)
        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(
            config.town, self.route, world_annotations)

        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)
        self.list_scenarios = self._build_scenario_instances(world,
                                                             ego_vehicle,
                                                             self.sampled_scenarios_definitions,
                                                             scenarios_per_tick=10,
                                                             timeout=self.timeout,
                                                             debug_mode=debug_mode>1)


class DynamicWeatherBehavior(py_trees.behaviour.Behaviour):
    WEATHERS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.ClearSunset,
    ]

    def __init__(self, name="DynamicWeatherBehavior"):
        super().__init__(name=name)
        self._current_time = None

    def initialise(self):
        """
        Set current time to current CARLA time
        """
        self._current_time = GameTime.get_time()
        
    def update(self):
        new_time = GameTime.get_time()
        delta_time = new_time - self._current_time
        
        # Switch to a different weather every 10s
        if delta_time > 10:
            self._current_time = new_time
            CarlaDataProvider.get_world().set_weather(random.choice(self.WEATHERS))
            
        return py_trees.common.Status.RUNNING

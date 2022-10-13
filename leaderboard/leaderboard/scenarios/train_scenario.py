'''
Training scenario with random routes, vehicles & pedestrians
Email: dchen@cs.utexas.edu
'''

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
from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRoute
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.junction_crossing_route import SignalJunctionCrossingRoute, NoSignalJunctionCrossingRoute

from leaderboard.scenarios.route_scenario import RouteScenario, convert_transform_to_location, convert_json_to_transform, convert_json_to_transform, compare_scenarios
from leaderboard.utils.route_manipulation import location_route_to_gps, _get_latlon_ref
from leaderboard.utils.route_parser import RouteParser


TRAINSCENARIO = ["TrainScenario"]

NUMBER_CLASS_TRANSLATION = {
    "Scenario1": ControlLoss,
    # "Scenario2": FollowLeadingVehicle,
    "Scenario3": DynamicObjectCrossing,
    "Scenario4": VehicleTurningRoute,
    # "Scenario5": OtherLeadingVehicle,
    # "Scenario6": ManeuverOppositeDirection,
    "Scenario7": SignalJunctionCrossingRoute,
    "Scenario8": SignalJunctionCrossingRoute,
    "Scenario9": SignalJunctionCrossingRoute,
    "Scenario10": NoSignalJunctionCrossingRoute
}

class TrainScenario(RouteScenario):
    
    """
    Training scenario with random routes, vehicles & pedestrians
    """
    
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
                self.scenario.scenario_tree.replace_child(behavior, DynamicWeatherBehavior(self.ego_vehicles))
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
                                                             
    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """

        # fix the random seed for reproducibility
        # rgn = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        def select_scenario(list_scenarios):
            # priority to the scenarios with higher number: 10 has priority over 9, etc.
            higher_id = -1
            selected_scenario = None
            for scenario in list_scenarios:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1

                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario

            return selected_scenario

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            # scenario_choice = select_scenario(possible_scenarios)
            scenario_choice = random.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = random.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios
                                                             
    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions,
                                  scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []

        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger_position']['x'],
                                     scenario['trigger_position']['y'],
                                     scenario['trigger_position']['z']) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                        color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

        for scenario_number, definition in enumerate(scenario_definitions):
            # Get the class possibilities for this scenario number
            if definition['name'] not in NUMBER_CLASS_TRANSLATION:
                continue
            
            print (definition['name'])
            
            scenario_class = NUMBER_CLASS_TRANSLATION[definition['name']]

            # Create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            # Create an actor configuration for the ego-vehicle trigger position
            
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                          ego_vehicle.get_transform(),
                                                                          'hero')]
            route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_configuration.route_var_name = route_var_name
            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration,
                                                   criteria_enable=False, timeout=timeout)
                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

            except Exception as e:
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                # import pdb; pdb.set_trace()
                continue

            scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def _set_route(self, hop_resolution=1.0):

        world = CarlaDataProvider.get_world()
        dao = GlobalRoutePlannerDAO(world.get_map(), hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        spawn_points = CarlaDataProvider._spawn_points
        # start, target = random.choice(spawn_points, size=2)
        start_idx, target_idx = random.choice(len(spawn_points), size=2)

        # DEBUG
        # start_idx, target_idx = 57, 87
        # start_idx, target_idx = 2, 18
        # print (start_idx, target_idx)
        start = spawn_points[start_idx]
        target = spawn_points[target_idx]

        route = grp.trace_route(start.location, target.location)
        self.route = [(w.transform,c) for w, c in route]

        CarlaDataProvider.set_ego_vehicle_route_waypoint(route)
        CarlaDataProvider.set_ego_vehicle_route([(w.transform.location, c) for w, c in route])
        # CarlaDataProvider.set_ego_vehicle_coarse_route([w.transform.location for w, c in route])

        gps_route = location_route_to_gps(self.route, *_get_latlon_ref(world))
        self.config.agent.set_global_plan(gps_route, self.route)

        self.timeout = self._estimate_route_timeout()

    def _initialize_actors(self, config):

        # initialize background vehicles
        super(TrainScenario, self)._initialize_actors(config)
        
        # initialize background pedestrians
        town_amount = {
            'Town01': 200,
            'Town02': 160,
            'Town03': 200,
            'Town04': 240,
            'Town05': 200,
            'Town06': 240,
            'Town07': 160,
            'Town08': 240,
            'Town09': 360,
            'Town10HD': 200,
        }

        amount = town_amount[config.town] if config.town in town_amount else 0
        
        blueprints = CarlaDataProvider._blueprint_library.filter('walker.pedestrian.*')
        spawn_points = []
        while len(spawn_points) < amount:
            spawn_point = carla.Transform()
            loc = CarlaDataProvider.get_world().get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
        pedestrians = CarlaDataProvider.handle_actor_batch(batch)
        
        batch = []
        walker_controller_bp = CarlaDataProvider._blueprint_library.find('controller.ai.walker')
        for pedestrian in pedestrians:
            batch.append(
                carla.command.SpawnActor(walker_controller_bp, carla.Transform(), pedestrian)
            )
        
        pedestrian_controllers = CarlaDataProvider.handle_actor_batch(batch)
        CarlaDataProvider.get_world().set_pedestrians_cross_factor(1.0)
        for controller in pedestrian_controllers:
            controller.start()
            controller.go_to_location(CarlaDataProvider.get_world().get_random_location_from_navigation())
            controller.set_max_speed(1.2 + random.random())
            
        for actor in itertools.chain(pedestrians, pedestrian_controllers):
            if actor is None:
                continue

            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)
            
            self.other_actors.append(actor)
            
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)


class DynamicWeatherBehavior(py_trees.behaviour.Behaviour):
    WEATHERS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.WetCloudySunset,
        carla.WeatherParameters.MidRainSunset,
        carla.WeatherParameters.HardRainSunset,
        carla.WeatherParameters.SoftRainSunset,
    ]

    def __init__(self, ego_vehicles, name="DynamicWeatherBehavior"):
        super().__init__(name=name)
        self._current_time = None
        self._ego_vehicles = ego_vehicles
        self._vehicle_lights_on = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        self._vehicle_lights_off = carla.VehicleLightState.NONE
        
    def initialise(self):
        """
        Set current time to current CARLA time
        """
        self._current_time = GameTime.get_time()
        
    def update(self):
        new_time = GameTime.get_time()
        delta_time = new_time - self._current_time
        
        # Switch to a different weather every 5s
        if delta_time > 10:
            self._current_time = new_time
            weather = random.choice(self.WEATHERS)
            
            # Randomize sun angles
            weather.sun_azimuth_angle = random.uniform(0,360)
            weather.sun_altitude_angle = random.uniform(-50,50)
            
            # Random raining
            weather.cloudiness = random.uniform(0,90)
            if random.uniform() > 0.5:
                weather.precipitation_deposits = random.uniform(0,50)
                weather.precipitation = 0
                weather.fog_density = 0
                weather.wetness = 0
            else:
                weather.precipitation_deposits = random.uniform(30,100)
                weather.precipitation = random.uniform(0,80)
                weather.fog_density = random.uniform(0,80)
                weather.wetness = random.uniform(0,100)

            if weather.sun_altitude_angle < 0:
                for vehicle in self._ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights_on))
            else:
                for vehicle in self._ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights_off))
        
            CarlaDataProvider.get_world().set_weather(weather)
        
        return py_trees.common.Status.RUNNING

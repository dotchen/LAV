import copy
import logging
import weakref
import numpy as np
import os
import time
import math
from threading import Thread

from queue import Queue
from queue import Empty

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from .map_utils import Wrapper as map_utils
from .pretty_map_utils import Wrapper as pretty_map_utils


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame

class StitchCameraReader:
    def __init__(self, bp_library, vehicle, sensor_spec, reading_frequency=1.0):
        
        fov = int(sensor_spec['fov'])
        # Hack
        self.yaws = [('left', -fov+5), ('center', 0), ('right', fov-5)]
        self.sensor_type = 'sensor.camera.'+str(sensor_spec['type'].split('.')[-1])

        bp = bp_library.find(self.sensor_type)
        
        bp.set_attribute('image_size_x', str(sensor_spec['width']))
        bp.set_attribute('image_size_y', str(sensor_spec['height']))
        bp.set_attribute('fov', str(sensor_spec['fov']))
        bp.set_attribute('lens_circle_multiplier', str(3.0))
        bp.set_attribute('lens_circle_falloff', str(3.0))
        if 'rgb' in sensor_spec['type']:
            bp.set_attribute('chromatic_aberration_intensity', str(0.5))
            bp.set_attribute('chromatic_aberration_offset', str(0))


        self.sensors = []
        self.datas = {}
        
        sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z'])
        for prefix, yaw in self.yaws:
            sensor_rotation = carla.Rotation(
                pitch=sensor_spec['pitch'],
                roll=sensor_spec['roll'],
                yaw=sensor_spec['yaw']+yaw
            )
        
            sensor = CarlaDataProvider.get_world().spawn_actor(bp, carla.Transform(sensor_location, sensor_rotation), vehicle)
            sensor.listen(self.__class__.on_camera_func(weakref.ref(self), prefix))
            self.sensors.append(sensor)
            
            
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()
            
    @staticmethod
    def on_camera_func(weakself, prefix):
        def on_camera(event):
            self = weakself()
            array = np.frombuffer(event.raw_data, dtype=np.dtype("uint8"))
            array = copy.deepcopy(array)
            array = np.reshape(array, (event.height, event.width, 4))

            self.datas[prefix] = array[...,2] if 'semantic' in self.sensor_type else array[...,:3]

        return on_camera

    def ready(self):
        for prefix, _ in self.yaws:
            if prefix not in self.datas:
                return False
        
        return True
    
    @property
    def stitched(self):
        images = []
        for prefix, _ in self.yaws:
            images.append(self.datas[prefix])
        
        return np.concatenate(images, axis=1)
    
    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                
                if not self.ready():
                    continue
                    
                current_time = GameTime.get_time()
                
                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self.stitched, GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False
                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False
    
    def destroy(self):
        self._run_ps = False
        for sensor in self.sensors:
            sensor.destroy()
        
        self.datas.clear()


class CollisionReader:
    def __init__(self, bp_library, vehicle, reading_frequency=1.0):
        self._collided = False
        bp = bp_library.find('sensor.other.collision')
        self.sensor = CarlaDataProvider.get_world().spawn_actor(bp, carla.Transform(), vehicle)
        self.sensor.listen(lambda event: self.__class__.on_collision(weakref.ref(self), event))
        
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()
        
    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self._collided, GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False
                else:
                    time.sleep(0.001)
    
    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False
        self.sensor.destroy()
        
    @staticmethod
    def on_collision(weakself, data):
        self = weakself()
        self._collided = True

class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()

    def __call__(self):
        pass

    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False

class PrettyMapReader(BaseReader):
    def __init__(self, vehicle, reading_frequency=1.0):
        super().__init__(vehicle, reading_frequency)

        client = CarlaDataProvider.get_client()
        world = CarlaDataProvider.get_world()
        town_map = CarlaDataProvider.get_map()
        
        # Initialize map
        pretty_map_utils.init(client, world, town_map, vehicle)
        
    def __call__(self):
        
        pretty_map_utils.tick()
        map_obs = pretty_map_utils.get_observations()
        labels = get_pretty_birdview(map_obs)

        return labels

class MapReader(BaseReader):
    def __init__(self, vehicle, reading_frequency=1.0):
        super().__init__(vehicle, reading_frequency)

        client = CarlaDataProvider.get_client()
        world = CarlaDataProvider.get_world()
        town_map = CarlaDataProvider.get_map()

        # Initialize map
        map_utils.init(client, world, town_map, vehicle)
        
    def __call__(self):
        
        map_utils.tick()
        map_obs = map_utils.get_observations()
        labels = get_birdview(map_obs)
        
        return labels

class ObjectsReader(BaseReader):
    
    MAX_CONNECTION_ATTEMPTS = 10
    MAX_RADIUS = 50

    def _get_location(self, actors_list):
        return [[v.get_location().x, v.get_location().y] for v in actors_list]

    def _get_rotation(self, actors_list):
        return [v.get_transform().rotation.yaw for v in actors_list]

    def _get_speed(self, actors_list):
        return [np.linalg.norm([v.get_velocity().x,v.get_velocity().y,v.get_velocity().z]) for v in actors_list]

    def _get_bbox(self, actors_list):
        return [[v.bounding_box.extent.x, v.bounding_box.extent.y] for v in actors_list]

    def _get_type(self, actors_list):
        """
        0 for walkers
        1 for vehicles
        """
        def get_type(v):
            if 'vehicle' in v.type_id:
                return 1
            else:
                return 0

        return [get_type(v) for v in actors_list]

    def _get_id(self, actors_list):
        return [v.id for v in actors_list]

    def __call__(self):
        
        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                ego_loc = self._vehicle.get_location()
                actors_list = list(filter(
                    lambda v: v.get_location().distance(ego_loc) < self.MAX_RADIUS \
                    and v.id != self._vehicle.id \
                    and ('vehicle' in v.type_id or 'pedestrian' in v.type_id), 
                    CarlaDataProvider.get_world().get_actors()
                ))
                # vehicles_list = CarlaDataProvider.get_world().get_actors().filter("vehicle.*")
                # vehicles_list = list(filter(lambda v:v.id != self._vehicle.id and v.get_location().distance(self._vehicle.get_location())<self.MAX_RADIUS, vehicles_list))

                # walkers_list = CarlaDataProvider.get_world().get_actors().filter("*pedestrian*")
                # walkers_list = list(filter(lambda w:w.get_location().distance(self._vehicle.get_location())<self.MAX_RADIUS, walkers_list))

                actors_list = [self._vehicle] + actors_list
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        return {
            'id':  self._get_id(actors_list),
            'loc': self._get_location(actors_list),
            'ori': self._get_rotation(actors_list),
            'spd': self._get_speed(actors_list),
            'bbox':self._get_bbox(actors_list),
            'type':self._get_type(actors_list),
        }

class SpeedometerReader(BaseReader):
    """
    Sensor to measure the speed of the vehicle.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed
        
    def _get_location(self, transform=None):
        if not transform:
            transform = self._vehicle.get_transform()
        
        loc = transform.location
        return [loc.x, loc.y, loc.z]
    
    def _get_rotation(self, transform=None):
        if not transform:
            transform = self._vehicle.get_transform()
        
        rot = transform.rotation
        return [rot.roll, rot.pitch, rot.yaw]
        
    def _get_red_light(self, transform=None):
        if not transform:
            transform = self._vehicle.get_transform()
        
        light_id = self._vehicle.get_traffic_light().id if self._vehicle.get_traffic_light() is not None else -1
        lights_list = CarlaDataProvider.get_world().get_actors().filter("*traffic_light*")
        light_state = str(self._vehicle.get_traffic_light_state())
        
        for traffic_light in lights_list:
            if light_id != traffic_light.id or light_state not in ['Red', 'Yellow']:
                continue
            
            waypoint = CarlaDataProvider.get_map().get_waypoint(transform.location)
            if not waypoint.is_junction:
                return True

        return False

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                lights_list = CarlaDataProvider.get_world().get_actors().filter("*traffic_light*")
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        return {
            'spd': self._get_forward_speed(transform=transform, velocity=velocity),
            'loc': self._get_location(transform=transform),
            'rot': self._get_rotation(transform=transform),
            # 'red': self._get_red_light(transform=transform),
        }


class OpenDriveMapReader(BaseReader):
    def __call__(self):
        return {'opendrive': CarlaDataProvider.get_map().to_opendrive()}


class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.CollisionEvent):
            self._parse_collision_cb(data, self.tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)
        
    def _parse_collision_cb(self, event, tag):
        collision = event.intensity > 20
        self._data_provider.update_sensor(tag, collision, event.frame)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 60
        # self._queue_timeout = 1e6

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None


    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor

        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, timestamp):
        if tag not in self._sensors_objects:
            raise SensorConfigurationInvalid("The sensor with tag [{}] has not been created!".format(tag))

        self._new_data_buffers.put((tag, timestamp, data))

    def get_data(self):
        try: 
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):

                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    break

                sensor_data = self._new_data_buffers.get(True, self._queue_timeout)
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")

        return data_dict


def get_birdview(observations):
    birdview = [
            observations['road'],
            observations['lane'],
            observations['stop'],
            observations['traffic'],
            observations['vehicle'],
            observations['pedestrian'],
            observations['waypoints'][0],
            observations['waypoints'][1],
            observations['waypoints'][2],
            observations['waypoints'][3],
            observations['waypoints'][4],
            observations['waypoints'][5],
            ]

    birdview = [x if x.ndim == 3 else x[...,None] for x in birdview]
    birdview = np.concatenate(birdview, 2)

    return birdview

def get_pretty_birdview(observations):
    birdview = [
            observations['road'],
            observations['vehicle'],
            # observations['stop'],
            observations['pedestrian'],
            observations['waypoints'][0],
            observations['waypoints'][1],
            observations['waypoints'][2],
            observations['waypoints'][3],
            observations['waypoints'][4],
            observations['waypoints'][5],
            observations['solid_lane'],
            observations['broken_lane'],
            # observations['traffic'],
            
            ]

    birdview = [x if x.ndim == 3 else x[...,None] for x in birdview]
    birdview = np.concatenate(birdview, 2)

    return birdview


def is_within_distance_ahead(target_transform, current_transform, max_distance):

    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0

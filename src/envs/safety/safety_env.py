import numpy as np
import random
import enum
import glfw
import gym

from safety_gym.envs.engine import Engine
from mujoco_py import MjViewer, const

class zone(enum.Enum):
    JetBlack = 0
    White    = 1
    Blue     = 2
    Green    = 3
    Red      = 4
    Yellow   = 5
    Cyan     = 6
    Magenta  = 7

    def __lt__(self, sth):
        return self.value < sth.value

    def __str__(self):
        return self.name[0]

GROUP_ZONE = 7

class SafetyEnv(Engine):
    """
    This environment is a modification of the Safety-Gym's environment.
    There is no "goal circle" but rather a collection of zones that the
    agent has to visit or to avoid in order to finish the task.

    For now we only support the 'point' robot.
    """
    def __init__(self, zones:list, use_fixed_map:float, timeout:int, config=dict):
        self.DEFAULT.update({
            'observe_zones': False,
            'zones_num': 0,  # Number of hazards in an environment
            'zones_placements': None,  # Placements list for hazards (defaults to full extents)
            'zones_locations': [],  # Fixed locations to override placements
            'zones_keepout': 0.4,  # Radius of hazard keepout for placement
            'zones_size': 0.3,  # Radius of hazards
        })

        self.zones = zones
        self.zone_types = list(set(zones))
        self.zone_types.sort()
        self.use_fixed_map = use_fixed_map
        self._rgb = {
            zone.JetBlack: [0, 0, 0, 1],
            zone.Blue    : [0, 0, 1, 1],
            zone.Green   : [0, 1, 0, 1],
            zone.Cyan    : [0, 1, 1, 1],
            zone.Red     : [1, 0, 0, 1],
            zone.Magenta : [1, 0, 1, 1],
            zone.Yellow  : [1, 1, 0, 1],
            zone.White   : [1, 1, 1, 1]
        }
        self.zone_rgbs = np.array([self._rgb[haz] for haz in self.zones])

        parent_config = {
            'robot_base': 'xmls/point.xml',
            'task': 'none',
            'lidar_num_bins': 8,
            'observation_flatten': False,
            'observe_zones': True,
            'zones_num': len(zones),
            'num_steps': timeout
        }
        parent_config.update(config)

        super().__init__(parent_config)

    @property
    def zones_pos(self):
        ''' Helper to get the zones positions from layout '''
        return [self.data.get_body_xpos(f'zone{i}').copy() for i in range(self.zones_num)]

    def build_observation_space(self):
        super().build_observation_space()

        if self.observe_zones:
            for zone_type in self.zone_types:
                self.obs_space_dict.update({f'zones_lidar_{zone_type}': gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)})

        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(self.obs_space_dict)

    def build_placements_dict(self):
        super().build_placements_dict()

        if self.zones_num: #self.constrain_hazards:
            self.placements.update(self.placements_dict_from_object('zone'))

    def build_world_config(self):
        world_config = super().build_world_config()

        for i in range(self.zones_num):
            name = f'zone{i}'
            geom = {'name': name,
                    'size': [self.zones_size, 1e-2],#self.zones_size / 2],
                    'pos': np.r_[self.layout[name], 2e-2],#self.zones_size / 2 + 1e-2],
                    'rot': self.random_rot(),
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_ZONE,
                    'rgba': self.zone_rgbs[i] * [1, 1, 1, 0.25]} #0.1]}  # transparent
            world_config['geoms'][name] = geom

        return world_config

    def build_obs(self):
        obs = super().build_obs()

        if self.observe_zones:
            for zone_type in self.zone_types:
                ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                pos_in_type = list(np.array(self.zones_pos)[ind])

                obs[f'zones_lidar_{zone_type}'] = self.obs_lidar(pos_in_type, GROUP_ZONE)

        return obs


    def render_lidars(self):
        offset = super().render_lidars()

        if self.render_lidar_markers:
            for zone_type in self.zone_types:
                if f'zones_lidar_{zone_type}' in self.obs_space_dict:
                    ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                    pos_in_type = list(np.array(self.zones_pos)[ind])

                    self.render_lidar(pos_in_type, np.array([self._rgb[zone_type]]), offset, GROUP_ZONE)
                    offset += self.render_lidar_offset_delta

        return offset

    def reset(self):
        if (self.use_fixed_map):
            self.seed(self._seed - 1)

        return super().reset()

    def get_events(self):
        events = ""
        for h_inedx, h_pos in enumerate(self.zones_pos):
            h_dist = self.dist_xy(h_pos)
            if h_dist <= self.zones_size:
                # We assume the agent to be in one zone at a time
                events += str(self.zones[h_inedx])

        return events

    def get_propositions(self):
        return [str(i) for i in self.zone_types]




class PlayEnv(SafetyEnv):
    def __init__(self, zones=[zone.Blue, zone.Blue, zone.Green, zone.Red, zone.Yellow], use_fixed_map=False, timeout=10000, config={}):
        super().__init__(zones, use_fixed_map, timeout, config)

    def show_text(self, text):
        self.viewer.show_text(text)

    def render(self, mode='human'):
        if self.viewer is None:
            self._old_render_mode = 'human'
            self.viewer = CustomViewer(self.sim)
            self.viewer.cam.fixedcamid = -1
            self.viewer.cam.type = const.CAMERA_FREE

            self.viewer.render_swap_callback = self.render_swap_callback
            # Turn all the geom groups on
            self.viewer.vopt.geomgroup[:] = 1
            self._old_render_mode = mode

        super().render()

    def obs(self):
        obs = super().obs()

        if not self.viewer is None:
            obs['key_pressed'] = self.viewer.consume_key()

        return obs


class CustomViewer(MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        self.key_pressed = None
        self.custom_text = None

    def show_text(self, text):
        self.custom_text = text

    def consume_key(self):
        ret = self.key_pressed
        self.key_pressed = None

        return ret

    def key_callback(self, window, key, scancode, action, mods):
        self.key_pressed = key
        if action == glfw.RELEASE:
            self.key_pressed = -1

        super().key_callback(window, key, scancode, action, mods)

    def _create_full_overlay(self):
        if (self.custom_text): self.add_overlay(const.GRID_TOPRIGHT, self.custom_text, "")
        super()._create_full_overlay()

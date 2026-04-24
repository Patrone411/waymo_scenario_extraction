import tensorflow as tf
import numpy as np
from feature_extraction.tools.lane_graph import LaneGraph
from collections import defaultdict

ACTOR_MAP={
    0:"Unset", 
    1: "Vehicle",
    2: "Pedestrian",
    3: "Cyclist",
    4: "Other"
}

# scnario id
scenario_id = {
    'scenario/id': tf.io.FixedLenFeature([1], tf.string, default_value=None)
}

# Field Definition
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/id':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/id':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/future/state':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/valid':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/id':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/x':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
    'traffic_light_state/future/y':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
    'traffic_light_state/future/z':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
}

features_description = {}
features_description.update(scenario_id)
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

class Scenario:
    def __init__(self, example, do_setup: bool = False, lane_graph_lane_types=[1,2,3]):

        assert isinstance(do_setup, bool), 'do_setup must be bool'
        self.example = example

        self.roadgraph_xyz = example['roadgraph_samples/xyz']
        self.roadgraph_types = example['roadgraph_samples/type'].squeeze()
        self.roadgraph_ids = example['roadgraph_samples/id'].squeeze()
        self.roadgraph_valid = example['roadgraph_samples/valid'].squeeze()
        self.roadgraph_dir = example['roadgraph_samples/dir']
        self.scenario_id = example['scenario/id']
        roadgraph_lanetypes = lane_graph_lane_types

        # Build lane graph on init
        self.lane_graph = LaneGraph(
            do_setup,
            self.roadgraph_xyz,
            self.roadgraph_dir,
            self.roadgraph_types,
            self.roadgraph_ids,
            self.roadgraph_valid,
            lane_types=roadgraph_lanetypes
        )

        map_mask = self.roadgraph_valid == 1
        self.roadgraph_xyz = self.roadgraph_xyz[map_mask]
        self.roadgraph_types = self.roadgraph_types[map_mask]
        self.roadgraph_ids = self.roadgraph_ids[map_mask]

        #types for points in roadgraph
        lane_types ={
            'LaneCenter-Freeway': 1,
            'LaneCenter-SurfaceStreet': 2,
            'LaneCenter-BikeLane' :3,
            'RoadLine-BrokenSingleWhite': 6,
            'RoadLine-SolidSingleWhite': 7, 
            'RoadLine-SolidDoubleWhite': 8, 
            'RoadLine-BrokenSingleYellow': 9, 
            'RoadLine-BrokenDoubleYellow': 10, 
            'Roadline-SolidSingleYellow': 11, 
            'Roadline-SolidDoubleYellow': 12, 
            'RoadLine-PassingDoubleYellow': 13, 
            'RoadEdgeBoundary': 15, 
            'RoadEdgeMedian': 16, 
            'StopSign': 17, 
            'Crosswalk': 18, 
            'SpeedBump': 19,
        }

        #colors fortypes for points in roadgraph
        self.map_line_types = {
            1: 'gray', 2: 'gray', 3: 'gray',
            6: 'black', 7: 'black', 8: 'black',
            9: 'yellow', 10: 'yellow', 11: 'yellow',
            12: 'yellow', 13: 'yellow',
            15: 'black', 16: 'black',
            17: 'red', 18: 'green', 19: 'blue'
        }

        include_types = list(self.map_line_types.keys())
        include_mask = np.isin(self.roadgraph_types, include_types)

        self.roadgraph_xyz = self.roadgraph_xyz[include_mask]
        self.roadgraph_types = self.roadgraph_types[include_mask]
        self.roadgraph_ids = self.roadgraph_ids[include_mask]

        # Actor states
        self.actor_ids = example['state/id']
        self.actor_idx_to_id = {idx: aid for idx, aid in enumerate(self.actor_ids)}
        self.actor_types = example['state/type']



        # helpers to mapped from tagged waymo data to actor ids
        self.actor_id_to_type_indexed = {}
        type_counters = defaultdict(lambda: -1)

        for aid, atype in zip(self.actor_ids, self.actor_types):
            type_counters[atype] += 1
            idx = type_counters[atype]
            type_name = ACTOR_MAP.get(int(atype), "Other")     # e.g. "Vehicle"
            type_key  = type_name.lower()
            self.actor_id_to_type_indexed[aid] = f"{type_key}_{idx}"  # e.g. vehicle_01
        self.actor_type_indexed_to_id = {v: k for k, v in self.actor_id_to_type_indexed.items()}


        self.full_x = np.concatenate([
            example['state/past/x'],
            example['state/current/x'],
            example['state/future/x']
        ], axis=1)

        self.full_y = np.concatenate([
            example['state/past/y'],
            example['state/current/y'],
            example['state/future/y']
        ], axis=1)

        print(f"created {self.name}")

    @property
    def name(self):
        '''returns the scenario id as a string'''
        return self.scenario_id[0].decode('utf-8')
    
    def setup(self):
        if self.lane_graph:
            self.lane_graph.setup()
            
        print(f"setup {self.name}")
   
    def get_actor_id_from_index(self, idx):
        """Returns the actor ID given its index."""
        if 0 <= idx < len(self.actor_ids):
            return self.actor_ids[idx]
        else:
            raise IndexError(f"Index {idx} out of range (max {len(self.actor_ids) - 1})")


""" module docstring """
# pylint: disable=import-error, no-name-in-module
from external.waymo_motion_scenario_mining.utils.stripped_actor_activities import per_actor_minimal
from external.waymo_motion_scenario_mining.utils.stripped_inter_actor_tags import build_inter_actor_position_and_ttc
from external.waymo_motion_scenario_mining.utils.stripped_environ_elements import EnvironmentElementsWaymo

from feature_extraction.tools.scenario import Scenario
from feature_extraction.tools.helpers.segment_polygon_handling import run_for_all_segments
from feature_extraction.tools.helpers.condense_actors import kept_actors_from_per_actor_minimal
from feature_extraction.tools.helpers.actor_per_segment import compute_segment_openscenario_coords
from feature_extraction.tools.helpers.env_elements_per_segment import process_env_elements_segment_wise



def env_elements_per_segment(parsed, processed_segs, debug:bool= False):
    if debug:print("environment elements start")
    env = EnvironmentElementsWaymo(parsed)
    env(eval_mode=False) 
    env_res = process_env_elements_segment_wise(processed_segs, env)
    if debug:print(env_res)
    if debug:print("environment elements done")
    return env_res

def process_actor_activities_per_segment(condensed_actors, processed_segs, debug: bool = False):
    if debug:print("actor_activities_per_segment start")
    per_segment_actor_activities = compute_segment_openscenario_coords(
        condensed=condensed_actors,
        processed_segs=processed_segs,
        dt=0.1,
        ds=0.5,           
    ) # process per segment actor data
    if debug:print("actor_activities_per_segment done")

    return per_segment_actor_activities

def process_inter_actor_activities(condensed_actors, debug: bool = False):
    if debug:print("inter actor start")

    inter_actor_relation = build_inter_actor_position_and_ttc(
        condensed_actors["actor_activities"],   
        dt=0.1,
    )
    if debug:print("inter actor relations done")

    return inter_actor_relation

def process_road_segments(lane_graph, road_segs, debug:bool = False):
    if debug:print("segments processing start ")
    processed_segs = run_for_all_segments(lane_graph, road_segs, show_plot=False)
    if debug:print("segments processing done ")

    return processed_segs


def result_dict_from_scenario(scenario, debug:bool = True):
    parsed = scenario.example
    scene_id = parsed['scenario/id'].item().decode("utf-8")
    if debug: print("executing scene id :", scene_id)
    lane_graph = scenario.lane_graph
    sequences = lane_graph.sequences
    root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
    road_segs = lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)
    processed_segs = process_road_segments(lane_graph=lane_graph, road_segs=road_segs, debug=debug)

    valid_keys = [
        k for k, v in processed_segs.items()
        if v.get("valid", v.get("reference_line") is not None)
    ]
    if debug and len(valid_keys) != len(processed_segs):
        dropped = [k for k in processed_segs.keys() if k not in valid_keys]
        print(f"Dropping {len(dropped)} segments without a reference_line:", 
              dropped[:10], "..." if len(dropped) > 10 else "")

    # Keep keys in sync across *everything* downstream
    road_segs = {k: road_segs[k] for k in valid_keys}
    processed_segs = {k: processed_segs[k] for k in valid_keys}


    actors = per_actor_minimal(parsed, eval_mode=False) # gets minimal preprocessed actor data from waymo data
    condensed = kept_actors_from_per_actor_minimal(road_segments=road_segs, per_actor=actors, min_steps=5) #filters out actors present in road segments

    inter_actor_activities = process_inter_actor_activities(condensed, debug=debug)
    actor_activities_per_segment = process_actor_activities_per_segment(condensed_actors=condensed, processed_segs=processed_segs, debug=debug)
    per_segment_env = env_elements_per_segment(parsed, processed_segs)
    result_dict = { 
    "scene_id": scene_id,
    "road_segments": road_segs,
    "general_actor_data": {
        k: v for k, v in condensed.items() if k != "per_segment_payloads"
    },
    "inter_actor_activities": inter_actor_activities,
    "segment_actor_data": actor_activities_per_segment,
    "segment_env_elements": per_segment_env,
    "processed_road_segments": processed_segs,
    }
    return result_dict
from feature_extraction.tools.scenario_processor import result_dict_from_scenario   # <-- anpassen!


def process_scenario(scenario):

    result = result_dict_from_scenario(scenario, debug=True)

    return result
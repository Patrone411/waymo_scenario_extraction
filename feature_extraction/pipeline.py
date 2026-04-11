from feature_extraction.tools.scenario import Scenario
from feature_extraction.tools.scenario_processor import result_dict_from_scenario   # <-- anpassen!


def process_scenario(example):

    scenario = Scenario(example)
    print("scenario insantiated")
    result = result_dict_from_scenario(scenario, debug=False)

    return result
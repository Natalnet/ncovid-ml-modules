import logger
import sys, json, os.path

sys.path.append("../")

assert os.path.isfile("../doc/configure.json"), "Config file unreached"

def add_all_configures_to_globals(configure_dict: dict) -> None:
    for dict_item in configure_dict.items():
        globals()[dict_item[0]] = dict_item[1]
        if isinstance(dict_item[1], dict):
            add_all_configures_to_globals(dict_item[1])
        else:
            pass

def overwrite(configure_dict: dict) -> None:
    add_all_configures_to_globals(configure_dict)

with open("../doc/configure.json") as json_file:
    try:
        configure_dict = json.load(json_file)
        globals().update(configure_dict)
        add_all_configures_to_globals(configure_dict)
            
    except Exception as e:
        logger.error_log("configs_manner.py", "loading cofigurations", f"Error: {e}.")


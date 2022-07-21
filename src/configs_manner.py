from numpy import inner
import logger
import sys, json, os.path

sys.path.append("../")

assert os.path.isfile("../doc/configure.json"), "Config file unreached"


def add_all_configures_to_globals(configure_dict: dict) -> None:
    try:
        for dict_item in configure_dict.items():
            add_variable_to_globals(dict_item[0], dict_item[1])
            if isinstance(dict_item[1], dict):
                add_all_configures_to_globals(dict_item[1])
            else:
                pass

    except Exception as e:
        logger.error_log(
            "configs_manner.py", "add all variables to globals", f"Error: {e}."
        )


def add_variable_to_globals(var_name: str, value) -> None:
    try:
        if value == ("True" or "False"):
            value = eval(value)
        globals()[var_name] = value
        update_variable_value_in_dictionaries(var_name, value)

    except Exception as e:
        logger.error_log("configs_manner.py", "add variable to globals", f"Error: {e}.")


def update_variable_value_in_dictionaries(
    new_var, new_value, inner_dict=globals()
) -> None:

    try:
        for item, value in inner_dict.items():
            if isinstance(value, dict):
                if new_var in value:
                    value[new_var] = new_value

    except Exception as e:
        logger.error_log(
            "configs_manner.py",
            "uptade variable value in sub-dictionaries",
            f"Error: {e}.",
        )


def overwrite(configure_dict: dict) -> None:
    try:
        add_all_configures_to_globals(configure_dict)

    except Exception as e:
        logger.error_log("configs_manner.py", "owerwrite configures", f"Error: {e}.")


def combine_configures_names_and_add_to_globals(
    configures_to_combine: dict, desired_new_name: str
) -> None:

    try:
        new_variable_value = combine_configures_names(configures_to_combine)
        add_variable_to_globals(desired_new_name, new_variable_value)

    except Exception as e:
        logger.error_log(
            "configs_manner.py",
            "combine configures names and add to globals",
            f"Error: {e}.",
        )


def combine_configures_names(configures_to_combine: dict) -> str:
    try:
        first_of_configures_names = str(configures_to_combine[0])
        others_configures_names = configures_to_combine[1:]
        new_combined_configure_name = first_of_configures_names

        for configure_name in others_configures_names:
            new_combined_configure_name += str(configure_name)

        return new_combined_configure_name

    except Exception as e:
        logger.error_log(
            "configs_manner.py", "combine configures names", f"Error: {e}."
        )


with open("../doc/configure.json") as json_file:
    try:
        configure_dict = json.load(json_file)
        globals().update(configure_dict)
        add_all_configures_to_globals(configure_dict)

        combine_configures_names_and_add_to_globals(
            [data_path, model_path], "model_path"
        )
        combine_configures_names_and_add_to_globals(
            [docs_path, glossary_file], "glossary_file"
        )

        combine_configures_names_and_add_to_globals(
            [data_path, metadata_path], "metadata_path"
        )

    except Exception as e:
        logger.error_log(
            "configs_manner.py",
            "loading cofigurations and add to globals",
            f"Error: {e}.",
        )

import logger
import sys, json, os.path

sys.path.append("../")


assert os.path.isfile("../doc/configure.json"), "Config file unreached"


def collect_Autoregressive():
    infos = {}
    infos["model_p"] = data["model_configs"]["Autoregressive"]["p"]
    infos["model_d"] = data["model_configs"]["Autoregressive"]["d"]
    infos["model_q"] = data["model_configs"]["Autoregressive"]["q"]
    return infos


def collect_Epidemiological():
    infos = {}
    infos["model_s_initial"] = data["model_configs"]["Epidemiological"]["s_initial"]
    infos["model_i_initial"] = data["model_configs"]["Epidemiological"]["i_initial"]
    infos["model_e_initial"] = data["model_configs"]["Epidemiological"]["e_initial"]
    infos["model_r_initial"] = data["model_configs"]["Epidemiological"]["r_initial"]
    infos["model_d_initial"] = data["model_configs"]["Epidemiological"]["d_initial"]
    return infos


def collect_Artificial():
    infos = {}
    infos["model_nodes"] = data["model_configs"]["Artificial"]["nodes"]
    infos["model_epochs"] = data["model_configs"]["Artificial"]["epochs"]
    infos["model_dropout"] = data["model_configs"]["Artificial"]["dropout"]
    infos["model_batch_size"] = data["model_configs"]["Artificial"]["batch_size"]
    infos["model_earlystop"] = data["model_configs"]["Artificial"]["earlystop"]
    infos["model_is_output_in_input"] = eval(
        data["model_configs"]["Artificial"]["is_output_in_input"]
    )
    infos["data_is_accumulated_values"] = eval(
        data["model_configs"]["Artificial"]["data_configs"]["is_accumulated_values"]
    )
    infos["data_is_apply_moving_average"] = eval(
        data["model_configs"]["Artificial"]["data_configs"]["is_apply_moving_average"]
    )
    infos["data_window_size"] = data["model_configs"]["Artificial"]["data_configs"][
        "window_size"
    ]
    infos["data_test_size_in_days"] = data["model_configs"]["Artificial"][
        "data_configs"
    ]["data_test_size_in_days"]
    infos["data_type_norm"] = data["model_configs"]["Artificial"]["data_configs"]["type_norm"]
    infos["repo"] = data["model_configs"]["Artificial"]["data_configs"]["repo"]
    infos["path"] = data["model_configs"]["Artificial"]["data_configs"]["path"]
    infos["input_features"] = data["model_configs"]["Artificial"]["data_configs"]["input_features"]
    infos["output_features"] = data["model_configs"]["Artificial"]["data_configs"]["output_features"]
    infos["date_begin"] = data["model_configs"]["Artificial"]["data_configs"]["date_begin"]
    infos["date_end"] = data["model_configs"]["Artificial"]["data_configs"]["date_end"]
    infos["moving_average_window_size"] = data["model_configs"]["Artificial"]["data_configs"]["moving_average_window_size"]

    return infos


def overwrite_Autoregressive(metadata):
    pass


def overwrite_Epidemiological(metadata):
    pass


def overwrite_Artificial(metadata):
    model_infos["model_nodes"] = metadata["model_configs"]["Artificial"]["nodes"]
    model_infos["model_epochs"] = metadata["model_configs"]["Artificial"]["epochs"]
    model_infos["model_dropout"] = metadata["model_configs"]["Artificial"]["dropout"]
    model_infos["model_batch_size"] = metadata["model_configs"]["Artificial"][
        "batch_size"
    ]
    model_infos["model_earlystop"] = metadata["model_configs"]["Artificial"][
        "earlystop"
    ]
    model_infos["model_is_output_in_input"] = eval(
        metadata["model_configs"]["Artificial"]["is_output_in_input"]
    )
    model_infos["data_is_accumulated_values"] = eval(
        metadata["model_configs"]["Artificial"]["data_configs"]["is_accumulated_values"]
    )
    model_infos["data_is_apply_moving_average"] = eval(
        metadata["model_configs"]["Artificial"]["data_configs"][
            "is_apply_moving_average"
        ]
    )
    model_infos["data_window_size"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["window_size"]
    model_infos["data_test_size_in_days"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["data_test_size_in_days"]
    model_infos["data_type_norm"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["type_norm"]
    
    model_infos["repo"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["repo"]
    model_infos["path"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["path"]
    model_infos["input_features"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["input_features"]
    model_infos["output_features"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["output_features"]
    model_infos["date_begin"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["date_begin"]
    model_infos["date_end"] = metadata["model_configs"]["Artificial"][
        "data_configs"
    ]["date_end"]
    model_infos["moving_average_window_size"] = metadata["model_configs"]["Artificial"]["data_configs"]["moving_average_window_size"]


def overwrite(metadata: dict = None):
    try:
        model_infos["model_id"] = metadata["model_configs"]["model_id"]
        model_path_remote = metadata["folder_configs"]["model_path_remote"]
        model_infos["model_type"] = metadata["model_configs"]["type_used"]

        getattr(sys.modules[__name__], f"overwrite_{model_type}")(metadata)

        logger.debug_log(
            "configs_manner.py", "overwite configurations", "Configurations overwrited"
        )

    except Exception as e:
        logger.error_log("configs_manner.py", "overwite configurations", f"Error: {e}.")


with open("../doc/configure.json") as json_file:
    try:
        data = json.load(json_file)

        doc_folder = data["folder_configs"]["docs_path"]
        data_path = data["folder_configs"]["data_path"]
        model_path = (
            data["folder_configs"]["data_path"] + data["folder_configs"]["model_path"]
        )
        model_path_remote = data["folder_configs"]["model_path_remote"]
        glossary_file = (
            data["folder_configs"]["docs_path"]
            + data["folder_configs"]["glossary_file"]
        )

        model_type = data["model_configs"]["type_used"]
        model_subtype = data["model_configs"][model_type]["model"]
        model_is_predicting = eval(data["model_configs"]["is_predicting"])

        model_infos = getattr(sys.modules[__name__], f"collect_{model_type}")()
        logger.debug_log(
            "configs_manner.py", "loading configurations", "Configurations loaded"
        )

    except Exception as e:
        logger.error_log("configs_manner.py", "loading configurations", f"Error: {e}.")

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
    infos["data_type_norm"] = data["model_configs"]["Artificial"]["data_configs"][
        "type_norm"
    ]
    return infos


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


import comet_llm
import comet_ml
import re
import configparser
import const as const


def init_and_login_comet():
    print("Comet SDK initializer")
    comet_llm.init()
    print("Comet SDK initialized")
    print("Comet SDK login")
    comet_ml.login()
    print("Comet SDK logged in")


def create_experiment(comet_config):
    if not comet_config:
        comet_config = get_comet_configuration()
    experiment = comet_ml.Experiment(
        api_key=comet_config["api_key"],
        workspace=comet_config["workspace"],
        project_name=comet_config["project_name"],
    )
    experiment.set_name(get_next_experiment_name(comet_config=comet_config))
    return experiment


def get_comet_last_experiment(comet_config=None):
    if not comet_config:
        comet_config = get_comet_configuration()
    experiments = get_all_project_experiments(comet_config)
    if len(experiments) > 0:
        return experiments[len(experiments) - 1]
    else:
        return None


def get_comet_experiment_by_name(experiment_name="", comet_config=None):
    if not experiment_name:
        return None
    try:
        int(experiment_name)
        experiment_name = f"experiment_{experiment_name}"
    except:
        pass
    if not comet_config:
        comet_config = get_comet_configuration()

    experiments = get_all_project_experiments(comet_config=comet_config)
    for experiment in experiments:
        if experiment.name == experiment_name:
            return experiment
    return None


def get_all_project_experiments(comet_config):
    if not comet_config:
        comet_config = get_comet_configuration()
    comet_api = comet_ml.API()
    workspace = comet_config["workspace"]
    project_name = comet_config["project_name"]
    experiments = comet_api.get_experiments(
        workspace=workspace, project_name=project_name
    )
    return experiments


def get_comet_configuration():
    config = configparser.ConfigParser()
    config.read(const.COMET_CONFIG)

    # Extract the Comet configuration parameters
    api_key = config.get("comet", "api_key")
    workspace = config.get("comet", "workspace")
    project_name = config.get("comet", "project_name")
    local_path = config.get("comet", "local_path")

    return {
        "api_key": api_key,
        "workspace": workspace,
        "project_name": project_name,
        "local_path": local_path,
    }


def get_next_experiment_name(comet_config=None):
    if not comet_config:
        comet_config = get_comet_configuration()

    experiments = get_all_project_experiments(comet_config=comet_config)
    experiment_names = [exp.name for exp in experiments]
    pattern = re.compile(r"experiment_(\d+)")
    last_number = 0
    for name in experiment_names:
        match = pattern.match(name)
        if match:
            number = int(match.group(1))
            last_number = max(last_number, number)
    new_experiment_number = last_number + 1
    new_experiment_name = f"experiment_{new_experiment_number}"
    print(f"New Experiment Name: {new_experiment_name}")
    return new_experiment_name

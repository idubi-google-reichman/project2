import app_const as const
from utils.profile_execution import execute_profile
import tensorflow as tf


# Function to get device pointer
def get_active_device_name():
    if tf.config.list_physical_devices("GPU"):
        return "/device:GPU:0"  # Assuming the first GPU is desired
    else:
        return "/device:CPU:0"


def execute_all_profiles(active_device_name):
    # for profile_name in const.PROFILES:
    #     print(f"Executing profile: {profile_name}")
    #     profile = const.PROFILES[profile_name]
    #     execute_profile(device=active_device, profile=profile)
    print(f"Executing profile: {const.PROFILES['2']}")
    execute_profile(device_name=active_device_name, profile=const.PROFILES["2"])


device_name = get_active_device_name()
execute_all_profiles(active_device_name=device_name)

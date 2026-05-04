import robosuite as suite
from robosuite import load_composite_controller_config

config = load_composite_controller_config(controller="BASIC")
print(config)

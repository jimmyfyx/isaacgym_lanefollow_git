from isaacgym import gymapi

'''
Loading assets helper functions
'''


def load_terrasentia(gym, sim):
    """Load Terrasentia asset from local directory.

    :param gym: The gym object
    :type gym: gym class
    :param sim: The simulator object
    :type gym: simulator class

    :return asset: The robot asset object in gym API
    :type asset: asset class
    """
    asset_root = "resources/isaac_gym_urdf_files/terra_description/urdf"
    asset_file = "terrasentia2022.urdf"

    # Terrasentia asset options
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = True

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return asset


def load_plants(gym, sim):
    """Load all plants asset from local directory, including corn, sorghum, and tobacco.

    :param gym: The gym object
    :type gym: gym class
    :param sim: The simulator object
    :type gym: simulator class

    :return asset_name_to_handler: A dictionary with every key as a plant variant name, and every value as
    the corresponding plant asset object
    :type asset_name_to_handler: dict[str -> asset obj]
    """
    asset_name_to_handler = {}  # Map asset name to asset handler

    # Load all corn variants
    corn_var_num = 21
    for var_type in range(corn_var_num):
        asset_folder = f"corn_variant_{str(var_type)}"
        asset_root = f"resources/isaac_gym_urdf_files/terra_worlds/models/{asset_folder}"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True

        asset_file = f"model.urdf"
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        asset_name_to_handler[asset_folder] = asset

    # Load all sorghum variants
    sorghum_var_num = 9
    for var_type in range(sorghum_var_num):
        asset_folder = f"sorghum_variant_{str(var_type)}"
        asset_root = f"resources/isaac_gym_urdf_files/terra_worlds/models/{asset_folder}"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True

        asset_file = f"model.urdf"
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        asset_name_to_handler[asset_folder] = asset

    # Load all tobacco variants
    tobacco_var_num = 9
    for var_type in range(tobacco_var_num):
        asset_folder = f"tobacco_variant_{str(var_type)}"
        asset_root = f"resources/isaac_gym_urdf_files/terra_worlds/models/{asset_folder}"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True

        asset_file = f"model.urdf"
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        asset_name_to_handler[asset_folder] = asset

    return asset_name_to_handler


def load_ground(gym, sim):
    """Load the ground skin from local directory as a URDF file.

    :param gym: The gym object
    :type gym: gym class
    :param sim: The simulator object
    :type gym: simulator class

    :return asset: The ground asset object in gym API
    :type asset: asset class
    """
    asset_root = "resources"
    asset_file = "carpet.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return asset

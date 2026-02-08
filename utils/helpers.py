import yaml

def get_conf(conf_path):
    with open(conf_path) as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return conf
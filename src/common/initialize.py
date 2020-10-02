import os
from importlib import import_module
import torch
from common.util import get_args


def initialize(models, reload, dir, model_path):
    for name, model in models.items():
        reload_model = reload and has_models(model_path)
        print(reload_model)
        if reload_model:
            models[name] = load_last_model(model, name, dir)
    return models


def infer_iteration(name, reload, model_path, save_path):
    resume = reload and has_models(model_path)
    if not resume:
        return 0
    names = list(filter_name(name, save_path))
    epochs = map(parse_model_id, names)
    return max(epochs) + 1


def has_models(path):
    return len(os.listdir(path)) > 0


def get_last_model(model_type, dir):
    names = list(filter_name(model_type, dir))
    last_name = max(names, key=parse_model_id)
    return os.path.join(dir, 'model', last_name)


def load_last_model(model, model_type, dir):
    path = get_last_model(model_type, dir)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model


def sort_name(names):
    return sorted(names, key=parse_model_id)


def parse_model_id(path):
    return int(path.split(':')[-1].split('.')[0])


def filter_name(name, dir):
    model_dir = os.path.join(dir, 'model')
    return filter(lambda x: name == x.split(':')[0], os.listdir(model_dir))


def define_last_model(model_type, model_path, model_name, **kwargs):
    model_definition = import_module('.'.join(('models', model_type, 'train')))
    model_parameters = get_args(model_path)
    model_parameters.update(kwargs)

    models = model_definition.define_models(**model_parameters)
    model = models[model_name]
    return load_last_model(model, model_name, model_path)

from importlib import import_module
import os
import argparse


def parse_args(_models_):
    root = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='evaluation')
    subparsers.required = True
    for model_name, _ in _models_.items():
        if not 'evaluate' in os.listdir(os.path.join(root, 'models', model_name)):
            continue

        for name in os.listdir(os.path.join(root, 'models', model_name, 'evaluate')):
            if name[-3:] != '.py':
                continue
            name = name[:-3]
            module = import_module('.'.join(('models', model_name, 'evaluate', name)))
            if not hasattr(module, 'parse_args') and not hasattr(module, 'execute'):
                continue
            model_parser = subparsers.add_parser(f'{model_name}-{name}')
            module.parse_args(model_parser)
            model_parser.set_defaults(func=module.execute)
    return parser.parse_args()


if __name__ == '__main__':
    root = os.path.dirname(os.path.realpath(__file__))
    models_root = os.path.join(root, 'models')
    models_path = [os.path.join(root, f) for f in os.listdir(models_root)]
    models_path = [m for m in models_path if not os.path.isfile(m)]
    models_name = [p.split('/')[-1] for p in models_path]
    _models_ = {name: import_module('.'.join(('models', name))) for name in models_name}
    args = parse_args(_models_)
    args.func(args)

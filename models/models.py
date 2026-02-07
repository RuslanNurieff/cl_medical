from utils.model_registry import register_model, MODEL_REGISTRY


@register_model("STFPM")
def _create_stfpm(config):
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs

    teacher = CustomFeatureExtractor(
        config['model']["backbone"],
        config['model']["layers"],
        frozen=True,
    )
    student = CustomFeatureExtractor(
        config['model']["backbone"],
        config['model']["layers"],
        frozen=False,
    )
    model = STFPM(teacher, student)
    training_args = STFPMTrainArgs(
        epochs=config['train']["epochs"],
        batch_size=config['train']["batch_size"],
        optimizer=config['model']["optimizer"],
        loss_function=config['model']["loss_function"],
    )

    model.to(config['train']['device'])
    training_args.init_train(model)
    return model, training_args


@register_model("RD4AD")
def _create_rd4ad(config):
    from moviad.models.rd4ad.rd4ad import RD4AD, RD4ADTrainArgs

    model = RD4AD(config['model']["backbone"],
                  input_size=config['model']["input_size"],)
    training_args = RD4ADTrainArgs(
        epochs=config['train']["epochs"],
        batch_size=config['train']["batch_size"],
        optimizer=config['model']["optimizer"],
        loss_function=config['model']["loss_function"],
    )

    model.to(config['train']["device"])
    training_args.init_train(model)
    return model, training_args


def get_model(config):
    name = config['model']['name'].upper()
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODEL_REGISTRY[name](config)

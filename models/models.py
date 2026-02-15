from utils.model_registry import register_model, MODEL_REGISTRY


@register_model("STFPM")
def _create_stfpm(config):
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs

    teacher = CustomFeatureExtractor(
        config['model']["backbone"],
        config['model']["layers"],
        config['train']['device'],
        frozen=True,
    )
    student = CustomFeatureExtractor(
        config['model']["backbone"],
        config['model']["layers"],
        config['train']['device'],
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

@register_model("CFA")
def _create_cfa(config):
    from moviad.models.cfa.cfa import CFA, CFATrainArgs
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

    feature_extractor = CustomFeatureExtractor(
        config['model']["backbone"],
        config['model']["layers"],
        config['train']['device'],
        frozen=True,
    )

    model = CFA(feature_extractor=feature_extractor,
                  backbone=config['model']['backbone'],)
    training_args = CFATrainArgs(
        epochs=config['train']["epochs"],
        batch_size=config['train']["batch_size"],
        optimizer=config['model']["optimizer"],
        loss_function=config['model']["loss_function"],
    )

    model.to(config['train']["device"])
    training_args.init_train(model)
    return model, training_args

@register_model("PatchCore")
def _create_patchcore(config):
    from moviad.models.patchcore.patchcore import PatchCore 
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.training_args import TrainingArgs

    feature_extractor = CustomFeatureExtractor(config['model']['backbone'],
                                               config['model']['layers'],
                                               config['train']['device'],
                                               frozen=True,)
    model = PatchCore(feature_extractor)
    training_args = TrainingArgs(epochs=config['train']['epochs'],
                                 batch_size=config['train']['batch_size'],)
    
    model.to(config['train']['device'])
    return model, training_args

@register_model("FastFlow")
def _create_fastflow(config):
    from moviad.models.fastflow.fastflow import create_fastflow, FastFlowTrainArgs

    model = create_fastflow(img_shape=config['model']['input_size'],
                            backbone_name=config['model']["backbone"],
                            device=config['train']["device"],)
    training_args = FastFlowTrainArgs(
        epochs=config['train']["epochs"],
        batch_size=config['train']["batch_size"],
        optimizer=config['model']["optimizer"],
        loss_function=config['model']["loss_function"],
    )

    model.to(config['train']["device"])
    training_args.init_train(model)
    return model, training_args

@register_model("Padim")
def _create_padim(config):
    from moviad.models.padim.padim import Padim, PadimTrainArgs

    model = Padim(backbone_model_name=config['model']['backbone'],
                  layers_idxs=config['model']['layers'],
                  device=config['train']['device'])
    training_args = PadimTrainArgs(
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

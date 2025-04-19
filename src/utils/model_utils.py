from src.config.config import MODEL_NAMES
from src.services.codet5_model import Codet5Model
from src.services.starcoder_model import StarCoderModel

def load_model_by_type(model_name):
    """
    Instantiates and returns a model object based on the specified model name.
    
    Returns an instance of either Codet5Model or StarCoderModel, depending on the
    provided model_name. If model_name does not match a supported model, returns
    None.
    """
    if model_name == MODEL_NAMES.CODET5:
        return Codet5Model(model_name)
    elif model_name == MODEL_NAMES.STARCODER2_3B:
        return StarCoderModel(model_name)

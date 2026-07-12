from src.val.BERT import val_BERT
from src.val.GPT2 import val_GPT2

def get_val(model_name, state_dict_full, logger, num_val_samples=40):
    """
    Returns:
        (success: bool, val_loss: float)
    """
    if model_name == 'BERT':
        return val_BERT(state_dict_full, logger, num_val_samples)
    elif model_name == 'GPT2':
        return val_GPT2(state_dict_full, logger, num_val_samples)
    else:
        return False, float('inf')
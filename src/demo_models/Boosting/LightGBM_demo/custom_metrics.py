import numpy as np


def rmsle(y_true, y_pred):
    """
    self-defined eval metric
    f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
    Root Mean Squared Logarithmic Error (RMSLE)

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
    """
    rmsle_result = np.sqrt(
        np.mean(
            np.power(np.log1p(y_pred) - np.log1p(y_true), 2)
        )
    )
    
    return 'RMSLE', rmsle_result, False


def rae(y_true, y_pred):
    """
    self-defined eval metric
    f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
    Relative Absolute Error (RAE)

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    rae_result = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true))

    return 'RAE', rae_result, False



def binary_error(preds, train_data):
    """
    # self-defined eval metric
    # f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    # binary error

    # NOTE: when you do customized loss function, the default prediction value is margin
    # This may make built-in evalution metric calculate wrong results
    # For example, we are doing log likelihood loss, the prediction is score before logistic transformation
    # Keep this in mind when you use the customization

    Args:
        preds ([type]): [description]
        train_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'error', np.mean(labels != (preds > 0.5)), False



def accuracy(preds, train_data):
    """
    # another self-defined eval metric
    # f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    # accuracy
    # NOTE: when you do customized loss function, the default prediction value is margin
    # This may make built-in evalution metric calculate wrong results
    # For example, we are doing log likelihood loss, the prediction is score before logistic transformation
    # Keep this in mind when you use the customization

    Args:
        preds ([type]): [description]
        train_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'accuracy', np.mean(labels == (preds > 0.5)), True


def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood



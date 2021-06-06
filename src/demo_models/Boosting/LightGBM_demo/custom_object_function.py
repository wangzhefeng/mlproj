

def loglikelihood(pred, train_data):
    """
    self-defined objective function
    f(preds: array, train_data: Dataset) -> grad: array, hess: array
    log likelihood loss

    Args:
        pred ([type]): [description]
        train_data ([type]): [description]
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    
    return grad, hess
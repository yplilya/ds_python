def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(prediction)):
        tp += prediction[i] and ground_truth[i]
        fp += prediction[i] and not ground_truth[i]
        tn += not prediction[i] and not ground_truth[i]
        fn += not prediction[i] and ground_truth[i]
    
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    # https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = sum(prediction == ground_truth) / len(prediction)
    return accuracy

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    false_negatives = 0
    false_positives = 0
    true_positives = 0
    correct = 0

    num_samples = prediction.shape[0]
    for i in range(num_samples) :
        if prediction[i] == ground_truth[i] :
            correct += 1
            if prediction[i] :
                true_positives += 1
        else :
            if prediction[i] :
                false_positives += 1
            else :
                false_negatives += 1  

    if true_positives + false_positives != 0 :
        precision = true_positives / (true_positives + false_positives)
    else :
        precision = 0    

    if true_positives + false_negatives != 0 :    
        recall = true_positives / (true_positives + false_negatives)
    else :
        recall = 0    
    
    if precision != 0 and recall != 0 :
        f1 = 2 * (precision * recall) / (precision + recall)
    else :    
        f1 = 0

    accuracy = correct / num_samples
    
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
    num_samples = prediction.shape[0]
    correct = 0
    for i in range(num_samples) :
        if prediction[i] == ground_truth[i] :
            correct += 1

    accuracy = correct / num_samples
    
    return accuracy

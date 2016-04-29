function [ train_input, train_classes, test_input, test_classes] = GenerateDatasetMNIST()
    train_input = loadMNISTImages('mnist-train-images-ubyte');
    train_input = train_input';
    test_input = loadMNISTImages('mnist-test-images-ubyte');
    test_input = test_input';
    
    output_train = loadMNISTLabels('mnist-train-labels-ubyte');
    output_test = loadMNISTLabels('mnist-test-labels-ubyte');

    test_classes = LabelsToSparse(output_test, 10);
    train_classes = LabelsToSparse(output_train, 10);
end

function [sparseLabel] = LabelsToSparse(labels, max_label)
    sparseLabel = zeros(size(labels,1), max_label);
    
    inds=[(1:length(labels))' labels+1];
    
    inds = sub2ind(size(sparseLabel),inds(:,1),inds(:,2));
    
    sparseLabel(inds) = 1;
end


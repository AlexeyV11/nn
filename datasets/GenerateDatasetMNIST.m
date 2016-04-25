function [ input_train, output_train, output_train_sparse, input_test, output_test, output_test_sparse] = GenerateDatasetMNIST()
    input_train = loadMNISTImages('mnist-train-images-ubyte');
    input_train = input_train';
    input_test = loadMNISTImages('mnist-test-images-ubyte');
    input_test = input_test';
    
    output_train = loadMNISTLabels('mnist-train-labels-ubyte');
    output_test = loadMNISTLabels('mnist-test-labels-ubyte');

    output_test_sparse = LabelsToSparse(output_test, 10);
    output_train_sparse = LabelsToSparse(output_train, 10);
end

function [sparseLabel] = LabelsToSparse(labels, max_label)
    sparseLabel = zeros(size(labels,1), max_label);
    
    inds=[(1:length(labels))' labels+1];
    
    inds = sub2ind(size(sparseLabel),inds(:,1),inds(:,2));
    
    sparseLabel(inds) = 1;
end


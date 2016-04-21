function [ input_train, output_train, input_test, output_test ] = GenerateDatasetMNIST()
    input_train = loadMNISTImages('mnist-train-images-ubyte');
    input_train = input_train';
    input_test = loadMNISTImages('mnist-test-images-ubyte');
    input_test = input_test';
    
    output_train = loadMNISTLabels('mnist-train-labels-ubyte');
    output_test = loadMNISTLabels('mnist-test-labels-ubyte');
end


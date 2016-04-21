function [ input_train, output_train, input_test, output_test ] = GenerateDatasetXOR()
    input_train = [0 0; 0 1; 1 0; 1 1];
    output_train = [0 ; 1; 1; 0];
    
    input_test = input_train;
    output_test = output_train;
end


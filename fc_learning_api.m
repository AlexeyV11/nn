function [ output_args ] = fc_learning_api( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here

    addpath('datasets');
    addpath('layers');
    addpath('nn');
    
    hidden_neurons_count = 2;
    output_neurons_count = 1;
    
    [input_train, output_train, input_test, output_test] = GenerateDatasetXOR();
    
    rng(0,'v5uniform');

    nn = network();
    
    nn = nn.addLayer(LayerInput(2));
    nn = nn.addLayer(LayerFC(2,hidden_neurons_count));
    nn = nn.addLayer(LayerActivationSigmoid());
    nn = nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count));
    nn = nn.addLayer(LayerActivationSigmoid());
    nn = nn.addLayer(LossEuclidean(output_neurons_count));
    
    learning_rate = 5.0;
        
    for i = 1:2000
        output_train_current = nn.forwardPropogate(input_train);
        loss = nn.computeLoss(output_train_current, output_train);
        nn.backPropagate(output_train_current, output_train, learning_rate);
    end
    
    
    output_test_current = nn.forwardPropogate(input_train);
    
    disp(output_test_current);
    
end


function [ output_args ] = fc_learning_api( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here

    startup();
    
    hidden_neurons_count = 2;
    output_neurons_count = 1;
    
    [input_train, output_train, input_test, output_test] = GenerateDatasetXOR();
    
    rng(0,'v5uniform');

    learningRate = 5.0;
    minibatchSize = 4;
    gradientUpdater = GradientUpdaterSimple(learningRate, minibatchSize);
    
    nn = network();
    
    nn.addLayer(LayerInput(2), gradientUpdater);
    nn.addLayer(LayerFC(2,hidden_neurons_count,WeightFillerUniform(0.8)), gradientUpdater);
    nn.addLayer(LayerActivationSigmoid(), gradientUpdater);
    nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count,WeightFillerUniform(0.8)), gradientUpdater);
    nn.addLayer(LayerActivationSigmoid(), gradientUpdater);
    lossLayer = LossEuclidean(output_neurons_count);
    
    
    for i = 1:2000
        output_train_current = nn.forwardPropogate(input_train);
        %loss = lossLayer.computeLoss(output_train_current, output_train);
        nn.backPropagate(lossLayer.computeDerivative(output_train_current, output_train));
    end
    
    
    output_test_current = nn.forwardPropogate(input_train);
    
    disp(output_test_current);
    
end


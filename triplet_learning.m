function [ output_args ] = triplet_learning( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here
    startup();
    
    [train_input, train_classes, test_input,  test_classes] = GenerateDatasetMNIST();
    
    hidden_neurons_count = 50;
    input_dim = size(train_input,2);
    output_neurons_count = 10;
    
    learningRate = 0.05;
    momentum = 0.9;
    weightDecay = 0.0005;
    
    nn = network();
    nn.addLayer(LayerInput(input_dim), {});
    nn.addLayer(LayerFC(input_dim,hidden_neurons_count,WeightFillerGaussian(0.001)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    nn.addLayer(LayerActivationSigmoid,  {});
    nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count,WeightFillerGaussian(0.001)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    nn.addLayer(LayerActivationSigmoid,  {});
            
    minibatchSize = 64;
    epochs = 50;
    margin = 0.3;
    
    trainTripletLossNetwork(nn, epochs, minibatchSize, margin, train_input, train_classes);

    %output_train_full = nn.forwardPropogate(train_input);
    %[~, ind_train] = max(output_train_full{end}');
    %[~, ind_train_gt] = max(train_classes');
    %accuracy_train = (sum(ind_train == ind_train_gt)) / numel(ind_train);

    %output_test_full = nn.forwardPropogate(test_input);
    %[~, ind_test] = max(output_test_full{end}');
    %[~, ind_test_gt] = max(test_classes');
    %accuracy_test = (sum(ind_test == ind_test_gt)) / numel(ind_test);

    %disp(['train accuracy : ' num2str(accuracy_train) ' test accuracy : ' num2str(accuracy_test)]);
end


function [ output_args ] = fc_learning_api( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here
    startup();
    
    [input_train, output_train_labels, output_train, input_test, output_test_labels, output_test] = GenerateDatasetMNIST();
    
    
    hidden_neurons_count = 50;
    output_neurons_count = 10;
    input_dim = size(input_train,2);
    
    rng(0,'v5uniform');
    
    
    learningRate = 0.05;
    momentum = 0.9;
    weightDecay = 0.0005;
    
    nn = network();
    
    nn.addLayer(LayerInput(input_dim), {});
    nn.addLayer(LayerFC(input_dim,hidden_neurons_count,WeightFillerGaussian(0.001)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    nn.addLayer(LayerActivationRELU,  {});
    nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count,WeightFillerGaussian(0.001)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    nn.addLayer(LayerActivationRELU,  {});
    
        
    epochs = 10;
    minibatchSize = 64;
    
    trainSoftmaxNetwork(nn, epochs, minibatchSize, input_train, output_train);

    output_train_full = nn.forwardPropogate(input_train);
    
    [val_train, ind_train] = max(output_train_full{end}');
    ind_train = ind_train'-1;
    accuracy_train = (sum(ind_train == output_train_labels)) / numel(ind_train);

    output_test_full = nn.forwardPropogate(input_test);
    
    [val_test, ind_test] = max(output_test_full{end}');
    ind_test = ind_test'-1;
    accuracy_test = (sum(ind_test == output_test_labels)) / numel(ind_test);

    %disp(['epoch ' num2str(epoch-1)]);
    %disp(['train loss : ' num2str(sum(loss_full_train)/numel(loss_full_train)) ' test loss : ' num2str(sum(loss_full_test)/numel(loss_full_test))]);
    disp(['train accuracy : ' num2str(accuracy_train) ' test accuracy : ' num2str(accuracy_test)]);
end


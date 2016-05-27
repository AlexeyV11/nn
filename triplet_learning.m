function [ output_args ] = triplet_learning( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here
    startup();
    
    [train_input, train_classes, test_input,  test_classes] = GenerateDatasetMNIST();
    
    hidden_neurons_count = 50;
    input_dim = size(train_input,2);
    output_neurons_count = 10;
    
    learningRate = 0.1;
    momentum = 0.9;
    weightDecay = 0.0005;
    
    nn = network();
    nn.addLayer(LayerInput(input_dim), {});
    nn.addLayer(LayerFC(input_dim,hidden_neurons_count,WeightFillerGaussian(0.001)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    nn.addLayer(LayerActivationSigmoid,  {});
    %nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count,WeightFillerGaussian(0.001)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    %nn.addLayer(LayerActivationSigmoid,  {});
            
    minibatchSize = 64;
    epochs = 5;
    margin = 0.3;
    
    dataProvider = TripletDataProvider(train_input, train_classes);
    trainTripletLossNetwork(nn, epochs, minibatchSize, margin, dataProvider);
    
    inputs = zeros(10,784);
    for i=1:10
        inputs(i,:) = dataProvider.features{i}{1};
    end
    
    outputs = nn.forwardPropogate(inputs);
    outputs = outputs{end};
    
    right = 0;
    wrong = 0;
    for digits=1:10
        for sampleInd=1:numel( dataProvider.features{digits})
            sampleFeature = dataProvider.features{digits}{sampleInd};
            sampleOutput = nn.forwardPropogate(sampleFeature);
            sampleOutput = sampleOutput{end};
            
            [~, ind] = min(pdist2(sampleOutput,outputs));
            
            if(ind == digits)
                right = right + 1;
            else
                wrong = wrong + 1;
            end
        end
        
    end
    
    disp(num2str(right));
    disp(num2str(wrong));
        
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


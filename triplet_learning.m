function [ output_args ] = triplet_learning( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here
    startup_nn();
    
    [train_input, train_classes, test_input,  test_classes] = GenerateDatasetMNIST();
    
    test_softmax(train_input, train_classes);
    test_triplet(train_input, train_classes);
end


function [] = test_softmax(train_input, train_classes)

    hidden_neurons_count = 50;
    output_neurons_count = 10;
    input_dim = size(train_input,2);
    
    rng(0,'v5uniform');
    
    
    learningRate = 0.1;
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
    
    trainSoftmaxNetwork(nn, epochs, minibatchSize, train_input, train_classes);
    
    
    output_train_full = nn.forwardPropogate(train_input);
    [~, ind_train] = max(output_train_full{end}');
    [~, ind_train_gt] = max(train_classes');
    accuracy_train = (sum(ind_train == ind_train_gt)) / numel(ind_train);

    disp(['train accuracy : ' num2str(accuracy_train)]);
    
    
    [~,labels] = max(train_classes');
     
    features = cell(1,max(labels));
    for i=1:numel(features)
        features{i} = {};
    end
 
    for i=1:numel(labels)
        features{labels(i)}{end+1} = train_input(i,:);
    end

    feats = struct();
    feats.features = features;
    
    dataProvider = TripletGeneratorRandom(feats);
    evaluate_softmax(nn, dataProvider.features);
end


function [] = evaluate_softmax(nn, features)
    inputs = zeros(10,784);
    for i=1:10
        inputs(i,:) = features{i}{1};
    end
    
    outputs = nn.forwardPropogate(inputs);
    outputs = outputs{end-2};
    
    right = 0;
    wrong = 0;
    for digits=1:10
        for sampleInd=1:numel( features{digits})
            sampleFeature = features{digits}{sampleInd};
            sampleOutput = nn.forwardPropogate(sampleFeature);
            sampleOutput = sampleOutput{end-2};
            
            [~, ind] = min(pdist2(sampleOutput,outputs));
            
            if(ind == digits)
                right = right + 1;
            else
                wrong = wrong + 1;
            end
        end
        
    end
    
    disp('softmax');
    disp(num2str(right));
    disp(num2str(wrong));
end


function [] = test_triplet(train_input, train_classes)
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
    %nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count,WeightFillerGaussian(0.001)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    %nn.addLayer(LayerActivationSigmoid,  {});
            
    minibatchSize = 64;
    epochs = 5;
    margin = 0.3;
    
    [~,labels] = max(train_classes');
     
    features = cell(1,max(labels));
    for i=1:numel(features)
        features{i} = {};
    end
 
    for i=1:numel(labels)
        features{labels(i)}{end+1} = train_input(i,:);
    end

    feats = struct();
    feats.features = features;
    
    
    dataProvider = TripletGeneratorRandom(feats);
    trainTripletLossNetwork(nn, epochs, minibatchSize, margin, dataProvider);
    
    evaluate_triplet(nn, dataProvider.features);
end

function [] = evaluate_triplet(nn, features)
    inputs = zeros(10,784);
    for i=1:10
        inputs(i,:) = features{i}{1};
    end
    
    outputs = nn.forwardPropogate(inputs);
    outputs = outputs{end};
    
    right = 0;
    wrong = 0;
    for digits=1:10
        for sampleInd=1:numel( features{digits})
            sampleFeature = features{digits}{sampleInd};
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
    
    disp('triplet');
    disp(num2str(right));
    disp(num2str(wrong));
end

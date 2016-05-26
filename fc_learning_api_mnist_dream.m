function [ output_args ] = fc_learning_api( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here
    startup();
    
    [train_input, train_classes, test_input,  test_classes] = GenerateDatasetMNIST();
    
    hidden_neurons_count = 50;
    output_neurons_count = 10;
    input_dim = size(train_input,2);
    
    rng(0,'v5uniform');
    
    
    learningRate = 0.5;
    momentum = 0.9;
    weightDecay = 0.0005;
    
    nn = network();
    
    nn.addLayer(LayerInput(input_dim), {});
    nn.addLayer(LayerFC(input_dim,hidden_neurons_count,WeightFillerGaussian(0.1)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    nn.addLayer(LayerActivationSigmoid,  {});
    nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count,WeightFillerGaussian(0.1)),  GradientUpdaterUsingMomentumAndWeightDecay(learningRate, momentum, weightDecay));
    nn.addLayer(LayerActivationSigmoid,  {});
    
        
    epochs = 5;
    minibatchSize = 64;
    
    trainSoftmaxNetwork(nn, epochs, minibatchSize, train_input, train_classes);

    %rand_input = rand(1,784);
    rand_input = train_input(2,:);
    lossLayer = LossEuclidean(size(train_input,2));
    
    answers = [0 0 0 0 0 0 0 0 0 1];
        
    for i=1:100
        img = reshape(rand_input, [28 28]);
        
        img = imresize(img, 4, 'nearest');
        imshow(img);
        pause(0.2);
        
        outputs = nn.forwardPropogate(rand_input);
        output_last = outputs{end};
        
        loss = lossLayer.computeLoss(output_last, answers);
        
        grads = nn.backPropagate1(outputs, lossLayer.computeDerivative(output_last, answers));
        
        
        rand_input = rand_input - grads{1};
        
        disp(output_last);
    end
        
end


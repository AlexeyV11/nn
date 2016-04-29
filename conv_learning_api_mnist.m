function [ output_args ] = conv_learning_api_mnist( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here
    startup();
    
    [input_train, output_train_labels, output_train, input_test, output_test_labels, output_test] = GenerateDatasetMNIST();
    
    
    hidden_neurons_count = 50;
    output_neurons_count = 10;
    input_dim = size(input_train,2);
    
    rng(0,'v5uniform');
    
    
    learningRate = 0.5;
    minibatchSize = 64;
    
    gradientUpdater = GradientUpdaterSimple(learningRate, minibatchSize);

    nn = network();
    
    nn.addLayer(LayerInput(input_dim), gradientUpdater);
    
    conv_kernels_1 = 8;
    nn.addLayer(LayerConv2([28,28,1],5,conv_kernels_1,WeightFillerGaussian(0.001)), gradientUpdater);
    nn.addLayer(LayerActivationRELU, gradientUpdater);
    
    %conv_kernels_2 = 32;
    %nn.addLayer(LayerConv2([28,28,conv_kernels_1],3,conv_kernels_2,WeightFillerGaussian(0.001)), gradientUpdater);
    %nn.addLayer(LayerActivationRELU, gradientUpdater);
    
    fc_inputs_1 = 28*28*conv_kernels_1;
    nn.addLayer(LayerFC(fc_inputs_1,hidden_neurons_count,WeightFillerGaussian(0.01)), gradientUpdater);
    nn.addLayer(LayerActivationSigmoid, gradientUpdater);
    nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count,WeightFillerGaussian(0.01)), gradientUpdater);
    nn.addLayer(LayerActivationSigmoid, gradientUpdater);
    
    lossLayer = LossSoftmax(output_neurons_count);
    
        
    for epoch = 1:20
        
        
        itersCount = floor(size(output_train,1)/minibatchSize);
        
        for iters = 1:itersCount 
            samples = input_train((iters-1)*minibatchSize+1:iters*minibatchSize,:);
            answers = output_train((iters-1)*minibatchSize+1:iters*minibatchSize,:);
            
            output_train_batch = nn.forwardPropogate(samples);
            loss = lossLayer.computeLoss(output_train_batch, answers);
            disp(['loss : ' num2str(sum(loss)/numel(loss))]);
            
            nn.backPropagate(lossLayer.computeDerivative(output_train_batch, answers));
        end
        
        pause(1);
        

    end
    
    aaa = 0;
end


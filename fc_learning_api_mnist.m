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
    minibatchSize = 64;
    
    gradientUpdater = GradientUpdaterSimple(learningRate, minibatchSize);

    nn = network();
    
    nn.addLayer(LayerInput(input_dim), gradientUpdater);
    nn.addLayer(LayerFC(input_dim,hidden_neurons_count,WeightFillerGaussian(0.001)), gradientUpdater);
    nn.addLayer(LayerActivationRELU, gradientUpdater);
    nn.addLayer(LayerFC(hidden_neurons_count,output_neurons_count,WeightFillerGaussian(0.001)), gradientUpdater);
    nn.addLayer(LayerActivationRELU, gradientUpdater);
    
    lossLayer = LossSoftmax(output_neurons_count);
        
    for epoch = 1:500
        if(rem(epoch, 1) == 0)
            output_train_full = nn.forwardPropogate(input_train);
            loss_full_train = lossLayer.computeLoss(output_train_full, output_train);

            [val_train, ind_train] = max(output_train_full');
            ind_train = ind_train'-1;
            accuracy_train = (sum(ind_train == output_train_labels)) / numel(ind_train);

            output_test_full = nn.forwardPropogate(input_test);
            loss_full_test = lossLayer.computeLoss(output_test_full, output_test);

            [val_test, ind_test] = max(output_test_full');
            ind_test = ind_test'-1;
            accuracy_test = (sum(ind_test == output_test_labels)) / numel(ind_test);

            disp(['epoch ' num2str(epoch-1)]);
            disp(['train loss : ' num2str(sum(loss_full_train)/numel(loss_full_train)) ' test loss : ' num2str(sum(loss_full_test)/numel(loss_full_test))]);
            disp(['train accuracy : ' num2str(accuracy_train) ' test accuracy : ' num2str(accuracy_test)]);
        end
        
        itersCount = floor(size(output_train,1)/minibatchSize);
        
        for iters = 1:itersCount 
            samples = input_train((iters-1)*minibatchSize+1:iters*minibatchSize,:);
            answers = output_train((iters-1)*minibatchSize+1:iters*minibatchSize,:);
            
            output_train_batch = nn.forwardPropogate(samples);
            loss = lossLayer.computeLoss(output_train_batch, answers);
            
            nn.backPropagate(lossLayer.computeDerivative(output_train_batch, answers));
        end

        
    end
end


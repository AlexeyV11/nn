function [ output_args ] = fc_learning_api( input_args )
%FC_LEARNING_API Summary of this function goes here
%   Detailed explanation goes here

    addpath('datasets');
    addpath('layers');
    
    hidden_neurons_count = 2;
    output_neurons_count = 1;
    
    [input_train, output_train, input_test, output_test] = GenerateDatasetXOR();
    
    rng(0,'v5uniform');
    
    layers = {};
    layers{end+1} = LayerInput(2);
    layers{end+1} = LayerFC(2,hidden_neurons_count);
    layers{end+1} = LayerActivationSigmoid();
    layers{end+1} = LayerFC(hidden_neurons_count,output_neurons_count);
    layers{end+1} = LayerActivationSigmoid();
    layers{end+1} = LossEuclidean(output_neurons_count);
    
    learning_rate = 5.0;
        
    for i = 1:2000
        % forward pass
        forwardOutput = input_train;
        for l=1:numel(layers)
            if(find(strcmp(superclasses(layers{l}), 'LayerInterface')))
                forwardOutput = layers{l}.feedForward(forwardOutput);
            else
                loss = layers{l}.feedForward(forwardOutput, output_train);
            end
        end

        %backward pass
        backwardOutput = [];
        for l=numel(layers):-1:1
            if(find(strcmp(superclasses(layers{l}), 'LayerInterface')))
                backwardOutput = layers{l}.backPropagate(backwardOutput, learning_rate);
            else
                backwardOutput = layers{l}.backPropagate(forwardOutput, output_train);
            end
        end
    end
    
    % forward pass
    forwardOutput = input_train;
    for l=1:numel(layers)
        if(find(strcmp(superclasses(layers{l}), 'LayerInterface')))
            forwardOutput = layers{l}.feedForward(forwardOutput);
        else
            loss = layers{l}.feedForward(forwardOutput, output_train);
        end
    end
    
    disp(forwardOutput);
    
end


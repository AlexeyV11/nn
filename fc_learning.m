

function [] = fc_learning


    addpath('datasets');
    learning_rate = 5.0;
    
    %[input_train, output_train, input_test, output_test] = GenerateDatasetMNIST();
    [input_train, output_train, input_test, output_test] = GenerateDatasetXOR();
    
    
    hidden_neurons_count = 2;
    output_neurons_count = 1;
    
    rng(0,'v5uniform');
    INIT_EPISLON = 0.8;

    W_input_to_hidden = rand(size(input_train,2)+1, hidden_neurons_count) * (2*INIT_EPISLON) - INIT_EPISLON;
    assert(isequal(size(W_input_to_hidden), [size(input_train,2)+1 hidden_neurons_count]));
    
    W_hidden_to_output = rand(hidden_neurons_count+1, output_neurons_count) * (2*INIT_EPISLON) - INIT_EPISLON;
    assert(isequal(size(W_hidden_to_output), [hidden_neurons_count+1 output_neurons_count]));
    
    for iter = 1:2000
        % https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    
        input = input_train;%(ind,:);
        target = output_train;%(ind,:);
        
        bias = -ones(size(input,1), 1);
        
        % forward pass
        hidden_i = [input bias] * W_input_to_hidden; %sum(W_input_to_hidden' .* repmat([input bias], [hidden_neurons_count 1]),2);%
        hidden_o = activation(hidden_i);

        output_i = [hidden_o bias] * W_hidden_to_output; %sum(W_hidden_to_output' .*  repmat([hidden_o bias], [output_neurons_count 1]),2);% + hidden_bias * W_hidden_to_output_bias;
        output_o = activation(output_i);

        e = (target - output_o) .* (target - output_o) / 2;
        %disp(sum(e));
        
        % backward pass
        De_Dotput_o = (output_o - target);
        Do_Doutput_i = activation_der(output_i);
        De_Doutput_i = De_Dotput_o .* Do_Doutput_i;
        % delta_hidden=delta_output*W2.*dsigmoid(Z2);

        d_W_hidden_to_output = De_Doutput_i' * [hidden_o bias];
        d_W_hidden_to_output = d_W_hidden_to_output / size(target,1);
        
        dE_Dhidden_o = De_Doutput_i * W_hidden_to_output';
        do_Dhidden_i = activation_der([hidden_i bias]);
        dE_Dhidden_i = do_Dhidden_i .* dE_Dhidden_o;
        dE_Dhidden_i = dE_Dhidden_i(:,1:end-1);
        
        %De_Doutput_i
        %dE_Dhidden_i
        d_W_input_to_hidden = dE_Dhidden_i' * [input bias];
        d_W_input_to_hidden = d_W_input_to_hidden / size(target,1);
        
        W_input_to_hidden = W_input_to_hidden - learning_rate * d_W_input_to_hidden';
        W_hidden_to_output = W_hidden_to_output - learning_rate * d_W_hidden_to_output';

        %WWW = [W_input_to_hidden' ; W_hidden_to_output']; 
        %WWW
        
        %DW = [d_W_input_to_hidden ; d_W_hidden_to_output];
        %DW
    end
    
     %%%%%%%%%%%%%%%%%%
     %%%% test
    target = output_test;
    input = input_test;

    bias = -ones(size(input,1), 1);

    % forward pass
    hidden_i = [input bias] * W_input_to_hidden;
    hidden_o = activation(hidden_i);

    output_i = [hidden_o bias] * W_hidden_to_output;
    output_o = activation(output_i);

    e = (target - output_o) .* (target - output_o) / 2;

    disp(output_o)
    disp(e)
    
end


function [y] = activation(x)
    y = 1 ./ (1 + exp(-x));
end

function [y] = activation_der(x)
    y = activation(x) .* (1 - activation(x));
end
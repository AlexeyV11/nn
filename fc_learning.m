

function [] = fc_learning

    % https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    learning_rate = 0.5;
    
    input_values = [0.05 0.10];
    target_values = [0.01 0.99];
    
    input_bias = 1;
    hidden_neurons_count = 2;
    
    W_input_to_hidden_bias = 0.35;
    W_input_to_hidden = [0.15 0.2; 0.25 0.30; W_input_to_hidden_bias W_input_to_hidden_bias];
    
    assert(isequal(size(W_input_to_hidden), [numel(input_values)+1 hidden_neurons_count]));
    
    
    output_neurons_count = 2;
    hidden_bias = 1;
    
    W_hidden_to_output_bias = 0.6;
    W_hidden_to_output = [0.40 0.45; 0.50 0.55; W_hidden_to_output_bias W_hidden_to_output_bias];
    
    assert(isequal(size(W_hidden_to_output), [output_neurons_count+1 hidden_neurons_count]));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    while(true)
        pause(0.2);
        
        % forward pass
        hidden_i = sum(W_input_to_hidden' .* repmat([input_values 1], [hidden_neurons_count 1]),2);% + input_bias * W_input_to_hidden_bias;
        hidden_o = activation(hidden_i)';

        output_i = sum(W_hidden_to_output' .*  repmat([hidden_o 1], [output_neurons_count 1]),2);% + hidden_bias * W_hidden_to_output_bias;
        output_o = activation(output_i)';

        e = (target_values - output_o) .* (target_values - output_o) / 2;

        disp(sum(e));
        
        % backward pass
        De_Dotput_o = (output_o - target_values);
        Do_Doutput_i = activation_der(output_o);
        De_Doutput_i = De_Dotput_o .* Do_Doutput_i;

        d_W_hidden_to_output = repmat(De_Doutput_i, [numel([hidden_o 1]) 1])' .* repmat([hidden_o 1], [numel(De_Doutput_i) 1]);

        dE_Dhidden_o = W_hidden_to_output * De_Doutput_i';
        do_Dhidden_i = activation_der([hidden_o 1]);
        dE_Dhidden_i = do_Dhidden_i .* dE_Dhidden_o';
        dE_Dhidden_i = dE_Dhidden_i(1:end-1);

        d_W_input_to_hidden = repmat(dE_Dhidden_i, [numel([input_values 1]) 1])' .* repmat([input_values 1], [numel(dE_Dhidden_i) 1]);

        W_input_to_hidden = W_input_to_hidden - learning_rate * d_W_input_to_hidden';
        W_hidden_to_output = W_hidden_to_output - learning_rate * d_W_hidden_to_output';

    end

end

function [y] = activation(x)
    y = 1 ./ (1 + exp(-x));
end

function [y] = activation_der(x)
    y = x .* (1 - x);
end
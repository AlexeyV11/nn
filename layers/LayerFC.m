classdef LayerFC < LayerInterface
    properties
    end
    
    properties (Access = 'private')        
        weights;
        activationsPrevWithBias;
        neuronsPrevCount;
        neuronsCurrentCount;
    end
    
    methods (Access = 'public')
        function obj = LayerFC(neuronsPrevCount, neuronsCurrentCount)
            INIT_EPISLON = 0.8;
            obj.weights = rand(neuronsPrevCount+1, neuronsCurrentCount) * (2*INIT_EPISLON) - INIT_EPISLON;
            
            obj.neuronsPrevCount = neuronsPrevCount;
            obj.neuronsCurrentCount = neuronsCurrentCount;
        end
        
        function [result] = feedForward(obj, activationsPrev)
            bias = -ones(size(activationsPrev,1), 1);
            obj.activationsPrevWithBias = [activationsPrev bias];
            result = obj.activationsPrevWithBias * obj.weights; %sum(W_input_to_hidden' .* repmat([input bias], [hidden_neurons_count 1]),2);%
        end
        
        function [gradientToPrev] = backPropagate(obj, gradientToCurrent, learningRate)
            % backpropogate gradient
            gradientToPrev = gradientToCurrent * obj.weights';
            % remove bias column
            gradientToPrev = gradientToPrev(:,1:end-1);
            
            % update current weights
            dWeights = gradientToCurrent' * obj.activationsPrevWithBias;
            obj.weights = obj.weights - dWeights' * learningRate / size(obj.activationsPrevWithBias,1);
            
            %obj.weights
        end
        
    end    
end


classdef LayerFC < LayerInterface
    properties
    end
    
    properties (Access = 'private')        
        weights;
        neuronsPrevCount;
        neuronsCurrentCount;
    end
    
    methods (Access = 'public')
        function obj = LayerFC(neuronsPrevCount, neuronsCurrentCount,weightFiller)
            obj.weights = weightFiller.generateWeights([neuronsPrevCount+1, neuronsCurrentCount]);
            
            obj.neuronsPrevCount = neuronsPrevCount;
            obj.neuronsCurrentCount = neuronsCurrentCount;
        end
        
        function [activationsCurrent] = feedForward(obj, activationsPrev)
            bias = -ones(size(activationsPrev,1), 1);
            activationsPrevWithBias = [activationsPrev bias];
            activationsCurrent = activationsPrevWithBias * obj.weights; %sum(W_input_to_hidden' .* repmat([input bias], [hidden_neurons_count 1]),2);%
        end
        
        function [gradientToPrev, gradientCurrent] = backPropagate(obj, gradientToCurrent, activationsPrev)
            % backpropogate gradient
            gradientToPrev = gradientToCurrent * obj.weights';
            % remove bias column
            gradientToPrev = gradientToPrev(:,1:end-1);
            
            % update current weights
            bias = -ones(size(activationsPrev,1), 1);
            activationsPrevWithBias = [activationsPrev bias];
            
            gradientCurrent = gradientToCurrent' * activationsPrevWithBias;

            gradientCurrent = gradientCurrent / size(activationsPrev,1);
            %obj.weights = gradientUpdater.update(obj.weights, dWeights);
            %obj.weights = obj.weights - dWeights' * learningRate / size(obj.activationsPrevWithBias,1);
            %obj.weights
        end
        
        function [] = update(obj, gradientUpdater, gradient)
            obj.weights = gradientUpdater.update(obj.weights, gradient);
        end
    end    
end


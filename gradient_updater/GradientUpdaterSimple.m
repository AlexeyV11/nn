classdef GradientUpdaterSimple < GradientUpdaterInterface
    properties
    end
    
    properties (Access = 'private')
        learningRate
        minibatchSize
    end
    
    methods (Access = 'public')
        function obj = GradientUpdaterSimple(learningRate, minibatchSize)
            obj.learningRate = learningRate;
            obj.minibatchSize = minibatchSize;
        end
        
        function [weights] = update(obj, weights, dw)
            weights = weights - dw' * obj.learningRate / obj.minibatchSize;
        end
        
        
    end    
end


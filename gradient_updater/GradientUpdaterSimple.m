classdef GradientUpdaterSimple < GradientUpdaterInterface
    properties
    end
    
    properties (Access = 'private')
        learningRate
    end
    
    methods (Access = 'public')
        function [] = setLearningRate(obj, lr)
            obj.learningRate = lr;
        end
        
        function obj = GradientUpdaterSimple(learningRate)
            obj.learningRate = learningRate;
        end
        
        function [weights] = update(obj, weights, dw)
            weights = weights - dw' * obj.learningRate;
        end
        
        
    end    
end


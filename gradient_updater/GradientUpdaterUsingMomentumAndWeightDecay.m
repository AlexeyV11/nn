classdef GradientUpdaterUsingMomentumAndWeightDecay < GradientUpdaterInterface
    properties
    end
    
    properties (Access = 'private')
        learningRate
        minibatchSize
        mu
        weightDecay
        
        weightsV
    end
    
    methods (Access = 'public')
        function obj = GradientUpdaterUsingMomentumAndWeightDecay(learningRate, minibatchSize, mu, weightDecay)
            obj.learningRate = learningRate;
            obj.minibatchSize = minibatchSize;
            obj.mu = mu; % momentum
            obj.weightDecay = weightDecay;
        end
        
        function [weights] = update(obj, weights, dw)
            if(isempty(obj.weightsV))
                obj.weightsV = zeros(size(dw'));
            end
            
            obj.weightsV =  obj.mu * obj.weightsV - dw' * obj.learningRate / obj.minibatchSize - obj.weightDecay * obj.learningRate * weights;
            weights = weights + obj.weightsV;
        end
        
        
    end    
end


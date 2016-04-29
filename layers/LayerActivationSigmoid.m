classdef LayerActivationSigmoid < LayerInterface
    properties
    end
    
    properties (Access = 'private')
    end
    
    methods (Access = 'public')
        function obj = LayerActivationSigmoid(obj)
        end
        
        function [activationsCurrent] = feedForward(obj, activationsPrev)
            activationsCurrent = 1 ./ (1 + exp(-activationsPrev));
        end
        
        function [gradientToPrev, gradientCurrent] = backPropagate(obj, gradientToCurrent, activationsPrev)
            result = 1 ./ (1 + exp(-activationsPrev));
            gradientToPrev = result .* (1 - result);
            gradientToPrev = gradientToPrev .* gradientToCurrent;

            gradientCurrent = {};
        end
        
        function [] = update(obj, gradientUpdater, gradient)
        end
    end    
end


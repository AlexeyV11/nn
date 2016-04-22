classdef LayerActivationSigmoid < LayerInterface
    properties
    end
    
    properties (Access = 'private')
        activationsPrev % this stuff is used in backpropogation stage
    end
    
    methods (Access = 'public')
        function obj = LayerActivationSigmoid(obj)
        end
        
        function [result] = feedForward(obj, activationsPrev)
            obj.activationsPrev = activationsPrev;
            result = 1 ./ (1 + exp(-activationsPrev));
        end
        
        function [gradientToPrev] = backPropagate(obj, gradientToCurrent, learningRate)
            result = 1 ./ (1 + exp(-obj.activationsPrev));
            gradientToPrev = result .* (1 - result);
            gradientToPrev = gradientToPrev .* gradientToCurrent;
        end
        
    end    
end


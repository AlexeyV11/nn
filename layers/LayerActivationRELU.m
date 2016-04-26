classdef LayerActivationRELU < LayerInterface
    properties
    end
    
    properties (Access = 'private')
        activationsPrev % this stuff is used in backpropogation stage
    end
    
    methods (Access = 'public')
        function obj = LayerActivationRELU(obj)
        end
        
        function [result] = feedForward(obj, activationsPrev)
            obj.activationsPrev = activationsPrev;
            result = max(activationsPrev, 0);
        end
        
        function [gradientToPrev] = backPropagate(obj, gradientToCurrent, learningRate)
            gradientToPrev = (obj.activationsPrev > 0) .* gradientToCurrent;
        end
        
    end    
end


classdef LayerActivationRELU < LayerInterface
    properties
    end
    
    properties (Access = 'private')
    end
    
    methods (Access = 'public')
        function obj = LayerActivationRELU(obj)
        end
        
        function [activationsCurrent] = feedForward(obj, activationsPrev)
            activationsCurrent = max(activationsPrev, 0);
        end
        
        function [gradientToPrev, gradientCurrent] = backPropagate(obj, gradientToCurrent, activationsPrev)
            gradientToPrev = (activationsPrev > 0) .* gradientToCurrent;
            gradientCurrent = {};
        end
        
        function [] = update(obj, gradientUpdater, gradient)
        end
    end    
end


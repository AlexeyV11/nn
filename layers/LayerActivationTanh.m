classdef LayerActivationTanh < LayerInterface
    properties
    end
    
    properties (Access = 'private')
    end
    
    methods (Access = 'public')
        function obj = LayerActivationTanh(obj)
        end
        
        function [activationsCurrent] = feedForward(obj, activationsPrev)
            activationsCurrent = 2./(1+exp(-2*activationsPrev))-1;
        end
        
        function [gradientToPrev, gradientCurrent] = backPropagate(obj, gradientToCurrent, activationsPrev)
            gradientToPrev = (1-(obj.feedForward(activationsPrev)).^2) .* gradientToCurrent;
            gradientCurrent = {};
        end
        
        function [] = update(obj, gradientUpdater, gradient)
        end
    end    
end


classdef LayerInput < LayerInterface
    properties
    end
    
    properties (Access = 'private')        
    end
    
    methods (Access = 'public')
        function obj = LayerInput(obj, inputDimension)
        end
        
        function [activationsCurrent] = feedForward(obj, activationsPrev)
            activationsCurrent = activationsPrev;
        end
        
        function [gradientToPrev, gradientCurrent] = backPropagate(obj, gradientToCurrent, activationsPrev)
            gradientToPrev = {};
            gradientCurrent = gradientToCurrent;
        end
        
        function [] = update(obj, gradientUpdater, gradient)
        end

    end    
end


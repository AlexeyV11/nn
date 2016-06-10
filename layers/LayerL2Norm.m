classdef LayerL2Norm < LayerInterface
    properties
    end
    
    properties (Access = 'private')        
    end
    
    methods (Access = 'public')
        function obj = LayerInput(obj, inputDimension)
        end
        
        function [activationsCurrent] = feedForward(obj, activationsPrev)
            factors = sqrt(sum(activationsPrev.*activationsPrev,2));
            activationsCurrent = activationsPrev ./ repmat(factors,1,size(activationsPrev,2));
        end
        
        function [gradientToPrev, gradientCurrent] = backPropagate(obj, gradientToCurrent, activationsPrev)
            
            factors = sqrt(sum(activationsPrev.*activationsPrev,2));
            gradientToPrev = gradientToCurrent .* repmat(factors,1,size(activationsPrev,2));
            gradientCurrent = {};
        end
        
        function [] = update(obj, gradientUpdater, gradient)
        end

    end    
end


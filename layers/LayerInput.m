classdef LayerInput < LayerInterface
    properties
    end
    
    properties (Access = 'private')        
    end
    
    methods (Access = 'public')
        function obj = LayerInput(obj, inputDimension)
        end
        
        function [result] = feedForward(obj, activationsPrev)
            result = activationsPrev;
        end
        
        function [gradientToPrev] = backPropagate(obj, gradientToCurrent, gradientUpdater)
            gradientToPrev = {};
        end
        
    end    
end


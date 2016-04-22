classdef LossEuclidean < LossInterface
    properties
    end
    
    properties (Access = 'private')        
        featureDimension;
    end
    
    methods (Access = 'public')
        function obj = LossEuclidean(dimension)
            obj.featureDimension = dimension;
        end
        
        function [result] = feedForward(obj, activationsPrev, activationsTarget)
            result = (activationsPrev - activationsTarget) .* (activationsPrev - activationsTarget) / 2;
        end
        
        function [gradientToPrev] = backPropagate(obj, activationsPrev, activationsTarget)
            gradientToPrev = (activationsPrev - activationsTarget);
        end
        
    end    
end


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
        
        function [result] = computeLoss(obj, activationsPrev, activationsTarget)
            result = (activationsPrev - activationsTarget) .* (activationsPrev - activationsTarget) / 2;
            result = sum(result, 2);
        end
        
        function [gradientToPrev] = computeDerivative(obj, activationsPrev, activationsTarget)
            gradientToPrev = (activationsPrev - activationsTarget);
        end
        
    end    
end


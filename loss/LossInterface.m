classdef LossInterface
    properties
    end
    
    properties (Access = 'private')        
    end
    methods(Abstract, Access = public)
        [result] = computeLoss(obj, activationsPrev, activationsTarget)
        [gradientToPrev] = computeDerivative(obj, activationsPrev, activationsTarget)
    end    
end


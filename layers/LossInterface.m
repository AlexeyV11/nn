classdef LossInterface
    properties
    end
    
    properties (Access = 'private')        
    end
    methods(Abstract, Access = public)
        [result] = feedForward(obj, activationsPrev, activationsTarget)
        [gradientToPrev] = backPropagate(obj, activationsPrev, activationsTarget)
    end    
end


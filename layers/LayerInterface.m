classdef (Abstract) LayerInterface < handle
    %BASELAYER Abstract class for Layer
    % Check format on reference project:
    % http://cs.stanford.edu/people/karpathy/convnetjs/
    % https://github.com/karpathy/convnetjs 
    % https://databoys.github.io/Feedforward/
    % http://scs.ryerson.ca/~aharley/neural-networks/
    
    properties (Abstract)
    end
    
    methods(Abstract, Access = public)
        [activationsCurrent] = feedForward(obj, activationsPrev);
        
        [gradientToPrev, gradientCurrent] = backPropagate(obj, gradientToCurrent, activationsPrev);
        
        update(obj, gradientUpdater, gradient);
        
        % Return the layer type
        %[type] = getType(obj); 
        
        % Get text description
        %[descText] = getDescription(obj);
        
        % Get number of neurons
        %[numNeurons] = getNumNeurons(obj);
    end
    
end


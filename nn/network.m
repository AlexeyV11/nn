classdef network
    %NETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = 'private')        
        layers;
    end
    
    methods (Access = 'public')
        function obj = network()
            obj.layers = {};
        end
        
        function [obj] = addLayer(obj, layer)
            obj.layers{end+1} = layer;
        end
        
        function [output] = forwardPropogate(obj, input)
            % forward pass
            output = input;
            for l=1:numel(obj.layers)-1
                if(find(strcmp(superclasses(obj.layers{l}), 'LayerInterface')))
                    output = obj.layers{l}.feedForward(output);
                end
            end
        end
        
        function [loss] = computeLoss(obj, outputForward, groundTrooth)
            loss = obj.layers{end}.feedForward(outputForward, groundTrooth);
        end
        
        function [backwardOutput] = backPropagate(obj, forwardOutput, groundTrooth, lerningRate)
            %backward pass
            backwardOutput = [];
            for l=numel(obj.layers):-1:1
                if(find(strcmp(superclasses(obj.layers{l}), 'LayerInterface')))
                    backwardOutput = obj.layers{l}.backPropagate(backwardOutput, lerningRate);
                else
                    backwardOutput = obj.layers{l}.backPropagate(forwardOutput, groundTrooth);
                end
            end
        end
        
        
        
    end
    
end


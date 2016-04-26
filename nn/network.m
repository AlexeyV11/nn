classdef network < handle
    %NETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = 'private')        
        layers;
        gradientUpdaters;
    end
    
    methods (Access = 'public')
        function obj = network()
            obj.layers = {};
            obj.gradientUpdaters = {};
        end
        
        function [] = addLayer(obj, layer, gradientUpdater)
            obj.layers{end+1} = layer;
            obj.gradientUpdaters{end+1} = gradientUpdater;
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
        
        function [loss] = computeLoss(obj, forwardOutput, groundTrooth)
            loss = obj.layers{end}.feedForward(forwardOutput, groundTrooth);
        end
        
        function [backwardOutput] = backPropagate(obj, forwardOutput, groundTrooth)
            %backward pass
            backwardOutput = [];
            for l=numel(obj.layers):-1:1
                if(find(strcmp(superclasses(obj.layers{l}), 'LayerInterface')))
                    backwardOutput = obj.layers{l}.backPropagate(backwardOutput, obj.gradientUpdaters{l});
                else
                    backwardOutput = obj.layers{l}.backPropagate(forwardOutput, groundTrooth);
                end
            end
        end
        
        
        
    end
    
end


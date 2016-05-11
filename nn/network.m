classdef network < handle
    %NETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = 'public')        
        layers;
        gradientUpdaters;
    end
    
    methods (Access = 'public')
        function obj = network()
            obj.layers = {};
            obj.gradientUpdaters = {};
        end
        
        function [] = setLayersLearningRate(obj, lr)
            for i=1:numel(obj.gradientUpdaters)
                if ~isempty(obj.gradientUpdaters{i})
                    obj.gradientUpdaters{i}.setLearningRate(lr);
                end
            end
        end
        
        function [] = addLayer(obj, layer, gradientUpdater)
            obj.layers{end+1} = layer;
            obj.gradientUpdaters{end+1} = gradientUpdater;
        end
        
        function [outputs] = forwardPropogate(obj, input)
            
            outputs = cell(1, numel(obj.layers)+1);
            % forward pass
            outputs{1} = input;
            for l=1:numel(obj.layers)
                if(find(strcmp(superclasses(obj.layers{l}), 'LayerInterface')))
                    outputs{l+1} = obj.layers{l}.feedForward(outputs{l});
                end
            end
        end
        
        function [gradients] = backPropagate(obj, outputs, lossDerivative)
            %backward pass
            gradients =  cell(1, numel(obj.layers));

            backwardOutput = lossDerivative;
            for l=numel(obj.layers):-1:1
                if(find(strcmp(superclasses(obj.layers{l}), 'LayerInterface')))
                    [backwardOutput, gradients{l}] = obj.layers{l}.backPropagate(backwardOutput, outputs{l});
                end
            end
            
            for l=numel(obj.layers):-1:1
                obj.layers{l}.update(obj.gradientUpdaters{l}, gradients{l});
            end
        end
        
        
        
    end
    
end


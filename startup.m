function [ output_args ] = startup( input_args )
%STARTUP Summary of this function goes here
%   Detailed explanation goes here
    addpath('datasets');
    addpath('layers');
    addpath('loss');
    addpath('weight_filler');
    addpath('gradient_updater');
    addpath('nn');
    addpath('training_funcs');
end


function [ output_args ] = startup_nn( input_args )
%STARTUP Summary of this function goes here
%   Detailed explanation goes here
    loc = fileparts(mfilename('fullpath'));

    addpath(genpath(fullfile(loc,'datasets')));
    addpath(genpath(fullfile(loc,'layers')));
    addpath(genpath(fullfile(loc,'loss')));
    addpath(genpath(fullfile(loc,'weight_filler')));
    addpath(genpath(fullfile(loc,'gradient_updater')));
    addpath(genpath(fullfile(loc,'nn')));
    addpath(genpath(fullfile(loc,'training_funcs')));
    addpath(genpath(fullfile(loc,'data_provider')));
end


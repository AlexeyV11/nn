classdef TripletGeneratorAP < handle
    properties
    end
    
    properties (Access = 'public')
        features
        samplesCount
        feat_matrix
        
        labels_shift
        labels_count
        samples_for_label_min_count
        samples_fot_label_max_count 
    end
    
    methods (Access = 'public')
        
%         function [obj] = TripletDataProvider(input_features, output_labels)
%             [~,labels] = max(output_labels');
%     
%             obj.features = cell(1,max(labels));
%             for i=1:numel(obj.features)
%                 obj.features{i} = {};
%             end
% 
%             for i=1:numel(labels)
%                 obj.features{labels(i)}{end+1} = input_features(i,:);
%             end
%             
%             obj.prepareUsingFeatures();
%         end
        
        function [obj] = TripletGeneratorAP(feats)
            obj.features = feats.features;
            obj.prepareUsingFeatures();
        end
        
        
        function [] = prepareUsingFeatures(obj)

            obj.samplesCount = 0;

            for i=1:numel(obj.features)
                obj.samplesCount = obj.samplesCount + numel(obj.features{i});
            end
            
            obj.feat_matrix = zeros(obj.samplesCount, numel(obj.features{1}{1}));

            %%%%%%
            ff = cell(1, numel(obj.features));
            for i=1:numel(obj.features)
                ff{i} = cell2mat(obj.features{i}');
            end
            
            obj.feat_matrix = cell2mat(ff');
            
            clear ff;
            %%%%%%
            
            %%%%%%%%%%%%%%%%%
%             tic
%             ind = 1;
%             for i = 1:numel(obj.features)
%                 for j = 1:numel(obj.features{i})
%                     feat = obj.features{i}{j};
%                     obj.feat_matrix(ind,:) = feat(1,:);
% 
%                     ind = ind + 1;
%                 end
%             end
%             
%             toc
            %%%%%%%%%%%%%%%%%%%
            
            
            %%%%% database stats
            label_shift_current = 0;
            obj.labels_shift = [];
            obj.labels_count = [];

            obj.samples_for_label_min_count = 100000000000;
            obj.samples_fot_label_max_count = -1;

            for i = 1:numel(obj.features)
                obj.labels_shift(end+1) = label_shift_current;
                obj.labels_count(end+1) = numel(obj.features{i});
                label_shift_current = label_shift_current + numel(obj.features{i});

                obj.samples_for_label_min_count = min(obj.samples_for_label_min_count, numel(obj.features{i}));
                obj.samples_fot_label_max_count = max(obj.samples_fot_label_max_count, numel(obj.features{i}));

            end
        end
        
        function [count] = getSamplesCount(obj)
            count = obj.samplesCount;
        end
        function [anchor_feats, positive_feats, negative_feats] = getMinibatch(obj, positivePairsCount, negativeSamplesCount, nn, lossTriplet)
            debug_log = 0;

           

            positivePersonLabel = randi([1 numel(obj.features)],1,positivePairsCount);

            positive_sizes = obj.labels_count(positivePersonLabel);
            positive_shifts = obj.labels_shift(positivePersonLabel);
            positive_arr_index = positive_shifts + rem(randi([0 obj.samples_fot_label_max_count],1,positivePairsCount), positive_sizes) + 1;
            anchor_arr_index = positive_shifts + rem(randi([0 obj.samples_fot_label_max_count],1,positivePairsCount), positive_sizes) + 1;

            negativePersonLabel = randi([1 numel(obj.features)], 1, negativeSamplesCount);

            if(debug_log)
                disp(['person index check ' num2str(sum(positivePersonLabel == negativePersonLabel)==0)]);
            end

            
            if(debug_log)

                disp(['arr index check 1 ' num2str(sum(positive_arr_index > obj.labels_shift(positivePersonLabel)) == positivePairsCount)]);
                disp(['arr index check 2 ' num2str(sum(positive_arr_index <= obj.labels_shift(positivePersonLabel)+obj.labels_count(positivePersonLabel)) == positivePairsCount)]);
                disp(['arr index check 3 ' num2str(sum(anchor_arr_index > obj.labels_shift(positivePersonLabel)) == positivePairsCount)]);
                disp(['arr index check 4 ' num2str(sum(anchor_arr_index <= obj.labels_shift(positivePersonLabel)+obj.labels_count(positivePersonLabel)) == positivePairsCount)]);
            end

            negative_sizes = obj.labels_count(negativePersonLabel);
            negative_shifts = obj.labels_shift(negativePersonLabel);
            negative_arr_index = negative_shifts + rem(randi([0 obj.samples_fot_label_max_count],1, negativeSamplesCount), negative_sizes) + 1;

            if(debug_log)
                disp(['arr index check 5 ' num2str(sum(negative_arr_index > obj.labels_shift(negativePersonLabel)) == positivePairsCount)]);
                disp(['arr index check 6 ' num2str(sum(negative_arr_index <= obj.labels_shift(negativePersonLabel)+obj.labels_count(negativePersonLabel)) == positivePairsCount)]);
            end

            
            %inds = positive_arr_index_1' ~= positive_arr_index_2';
            %triplets = triplets(inds,:);

            anchor_feats = obj.feat_matrix(anchor_arr_index',:);
            positive_feats = obj.feat_matrix(positive_arr_index',:);
            negative_feats = obj.feat_matrix(negative_arr_index',:);
            
            anchorOutputs =     nn.forwardPropogate(anchor_feats);
            positiveOutputs =   nn.forwardPropogate(positive_feats);
            negativeOutputs =   nn.forwardPropogate(negative_feats);

            positivePersonLabel = vectorDublicate(positivePersonLabel', negativeSamplesCount);
            anchorOutputs = vectorDublicate(anchorOutputs{end}, negativeSamplesCount);
            positiveOutputs = vectorDublicate(positiveOutputs{end}, negativeSamplesCount);
            
            
            % fix dublication to repmat
            negativePersonLabel = repmat(negativePersonLabel', [positivePairsCount 1]);
            negativeOutputs = repmat(negativeOutputs{end}, [positivePairsCount 1]);
            
            loss = lossTriplet.computeLoss(anchorOutputs, positiveOutputs, negativeOutputs);
            loss(positivePersonLabel == negativePersonLabel) = 0;
            
            
            
            loss2d = reshape(loss,negativeSamplesCount,[]);
            [lossVal, ind] = SelectElementsForRows(loss2d);

            lossInd = ind;%(0:positivePairsCount-1)*negativeSamplesCount+ind;

            %loss(lossInd) == val;
            
            indPos = floor((lossInd - 1) / negativeSamplesCount) + 1;
            indNeg = rem(lossInd - 1, negativeSamplesCount) + 1;
            
            anchor_feats = anchor_feats(indPos,:);
            positive_feats = positive_feats(indPos,:);
            negative_feats = negative_feats(indNeg,:);
            
            anchor_feats(lossVal <= 0, :) = [];
            positive_feats(lossVal <= 0, :) = [];
            negative_feats(lossVal <= 0, :) = [];
            
        end
    end    
    
    
end

function [vec] = vectorDublicate(x, n)
        vec = kron(x, ones(n,1));
        %vec=x';
        %vec=repmat(vec,1,n)'; 
        %vec=vec(:)';
end

function [finalVals, finalInds] = SelectElementsForRows(loss2d)
    %matrix = randi([0 1], 10, 10);
    matrix = (loss2d>0);
    
    [r, c] = ind2sub(size(matrix), find(matrix == 1));

    coords = [r c];
    coordsPermuted = coords(randperm(length(r)),:);

    %[~,sorted_inds] = sort(coordsPermuted(:,2));
    %coordsSorted = coordsPermuted(sorted_inds,:);

    [~, uInd] = unique(coordsPermuted(:,2));

    finalCoords = coordsPermuted(uInd,:);
    
    finalInds = sub2ind(size(matrix),finalCoords(:,1),finalCoords(:,2));
    finalVals = loss2d(finalInds);
    
    %r = finalCoords(:,1);
    %c = finalCoords(:,2);
    
end


function [finalVals, finalInds] = SelectMaxElementsForRows(loss2d)
    [finalVals, r] = max(loss2d);
    
    finalInds = (0:size(loss2d,2)-1)*size(loss2d,1)+r;
end


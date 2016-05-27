classdef TripletDataProvider < handle
    properties
    end
    
    properties (Access = 'private')
        features
        samplesCount
        feat_matrix
    end
    
    methods (Access = 'public')
        
        function [obj] = TripletDataProvider(input_features, output_labels)
            [~,labels] = max(output_labels');
    
            obj.features = cell(1,max(labels));
            for i=1:numel(obj.features)
                obj.features{i} = {};
            end

            for i=1:numel(labels)
                obj.features{labels(i)}{end+1} = input_features(i,:);
            end
            
            obj.samplesCount = size(output_labels,1);
            
            
            
            obj.feat_matrix = zeros(obj.samplesCount, 784);

            ind = 1;
            for i = 1:numel(obj.features)
                for j = 1:numel(obj.features{i})
                    feat = obj.features{i}{j};
                    obj.feat_matrix(ind,:) = feat(1,:);

                    ind = ind + 1;
                end
            end

        end
        
        function [count] = getSamplesCount(obj)
            count = obj.samplesCount;
        end
        function [anchor_feats, positive_feats, negative_feats] = getMinibatch(obj, triplets_count)
            debug_log = 0;

            count = 0;
            label_shift = [];
            label_count = [];

            feat_for_person_min = 100000000000;
            feat_for_person_max = -1;

            for i = 1:numel(obj.features)
                label_shift(end+1) = count;
                label_count(end+1) = numel(obj.features{i});
                count = count + numel(obj.features{i});

                feat_for_person_min = min(feat_for_person_min, numel(obj.features{i}));
                feat_for_person_max = max(feat_for_person_max, numel(obj.features{i}));

            end

            positive_person_index = randi([1 numel(obj.features)],1,triplets_count);

            negative_person_index = positive_person_index + randi([0 numel(obj.features) - 2],1,triplets_count);
            negative_person_index = rem(negative_person_index, numel(obj.features))+1;

            if(debug_log)
                disp(['person index check ' num2str(sum(positive_person_index == negative_person_index)==0)]);
            end

            positive_sizes = label_count(positive_person_index);
            positive_shifts = label_shift(positive_person_index);
            positive_arr_index_1 = positive_shifts + rem(randi([0 feat_for_person_max],1,triplets_count), positive_sizes) + 1;
            positive_arr_index_2 = positive_shifts + rem(randi([0 feat_for_person_max],1,triplets_count), positive_sizes) + 1;

            if(debug_log)

                disp(['arr index check 1 ' num2str(sum(positive_arr_index_1 > label_shift(positive_person_index)) == triplets_count)]);
                disp(['arr index check 2 ' num2str(sum(positive_arr_index_1 <= label_shift(positive_person_index)+label_count(positive_person_index)) == triplets_count)]);
                disp(['arr index check 3 ' num2str(sum(positive_arr_index_2 > label_shift(positive_person_index)) == triplets_count)]);
                disp(['arr index check 4 ' num2str(sum(positive_arr_index_2 <= label_shift(positive_person_index)+label_count(positive_person_index)) == triplets_count)]);
            end

            negative_sizes = label_count(negative_person_index);
            negative_shifts = label_shift(negative_person_index);
            negative_arr_index = negative_shifts + rem(randi([0 feat_for_person_max],1,triplets_count), negative_sizes) + 1;

            if(debug_log)
                disp(['arr index check 5 ' num2str(sum(negative_arr_index > label_shift(negative_person_index)) == triplets_count)]);
                disp(['arr index check 6 ' num2str(sum(negative_arr_index <= label_shift(negative_person_index)+label_count(negative_person_index)) == triplets_count)]);
            end

            triplets = [positive_arr_index_1' positive_arr_index_2' negative_arr_index'];

            inds = positive_arr_index_1' ~= positive_arr_index_2';
            triplets = triplets(inds,:);

            anchor_feats = obj.feat_matrix(triplets(:,1),:);
            positive_feats = obj.feat_matrix(triplets(:,2),:);
            negative_feats = obj.feat_matrix(triplets(:,3),:);


            %disp(['setLearningRate' num2str(obj.learningRate)]);
        end
    end    
end


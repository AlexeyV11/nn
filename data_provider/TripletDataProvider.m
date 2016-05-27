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

            label_shift_current = 0;
            labels_shift = [];
            labels_count = [];

            samples_for_label_min_count = 100000000000;
            samples_fot_label_max_count = -1;

            for i = 1:numel(obj.features)
                labels_shift(end+1) = label_shift_current;
                labels_count(end+1) = numel(obj.features{i});
                label_shift_current = label_shift_current + numel(obj.features{i});

                samples_for_label_min_count = min(samples_for_label_min_count, numel(obj.features{i}));
                samples_fot_label_max_count = max(samples_fot_label_max_count, numel(obj.features{i}));

            end

            positive_person_index = randi([1 numel(obj.features)],1,triplets_count);

            negative_person_index = positive_person_index + randi([0 numel(obj.features) - 2],1,triplets_count);
            negative_person_index = rem(negative_person_index, numel(obj.features))+1;

            if(debug_log)
                disp(['person index check ' num2str(sum(positive_person_index == negative_person_index)==0)]);
            end

            positive_sizes = labels_count(positive_person_index);
            positive_shifts = labels_shift(positive_person_index);
            positive_arr_index_1 = positive_shifts + rem(randi([0 samples_fot_label_max_count],1,triplets_count), positive_sizes) + 1;
            positive_arr_index_2 = positive_shifts + rem(randi([0 samples_fot_label_max_count],1,triplets_count), positive_sizes) + 1;

            if(debug_log)

                disp(['arr index check 1 ' num2str(sum(positive_arr_index_1 > labels_shift(positive_person_index)) == triplets_count)]);
                disp(['arr index check 2 ' num2str(sum(positive_arr_index_1 <= labels_shift(positive_person_index)+labels_count(positive_person_index)) == triplets_count)]);
                disp(['arr index check 3 ' num2str(sum(positive_arr_index_2 > labels_shift(positive_person_index)) == triplets_count)]);
                disp(['arr index check 4 ' num2str(sum(positive_arr_index_2 <= labels_shift(positive_person_index)+labels_count(positive_person_index)) == triplets_count)]);
            end

            negative_sizes = labels_count(negative_person_index);
            negative_shifts = labels_shift(negative_person_index);
            negative_arr_index = negative_shifts + rem(randi([0 samples_fot_label_max_count],1,triplets_count), negative_sizes) + 1;

            if(debug_log)
                disp(['arr index check 5 ' num2str(sum(negative_arr_index > labels_shift(negative_person_index)) == triplets_count)]);
                disp(['arr index check 6 ' num2str(sum(negative_arr_index <= labels_shift(negative_person_index)+labels_count(negative_person_index)) == triplets_count)]);
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


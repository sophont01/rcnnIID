function[feature] = normalizeSiftFeatures(feature)


for i=1:size(feature,1)
    for j=1:size(feature,2)
        feature_norm = norm(reshape(double(feature(i,j,:)),[],size(feature,3)));
        if feature_norm == 0
            feature(i,j,:) = 0;
        else
            feature(i,j,:) = feature(i,j,:)./feature_norm;
        end
        
    end
end



end
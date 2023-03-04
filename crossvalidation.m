clear all

dirIn = 'data/';
datasetnames = {'multispectral_day01.mat','multispectral_day06.mat','multispectral_day13.mat','multispectral_day20.mat','multispectral_day28.mat';
    'annotation_day01.png','annotation_day06.png','annotation_day13.png','annotation_day20.png','annotation_day28.png'};
imagenames =  {"color_day01.png","color_day06.png","color_day13.png","color_day20.png","color_day28.png"};
short_names = {'day01', 'day06', 'day13', 'day20', 'day28'};

%%
fprintf("\n\nCross validation : error rates on annotations (multispectral LDA)\n");
fprintf("\nTraining\t\tTest set\n");
fprintf("  set\t\t\tday01\t\tday06\t\tday13\t\tday20\t\tday28\t\taverage\n");


for training_set = 1:5
average_error = 0;
fprintf(short_names{training_set});
fprintf("\t\t");

[multiIm, annotationIm] = loadMulti(datasetnames{1,training_set} , datasetnames{2,training_set}, dirIn);
[fatTrainingValues, fatR, fatC] = getPix(multiIm, annotationIm(:,:,2));
[meatTrainingValues, meatR, meatC] = getPix(multiIm, annotationIm(:,:,3));

% estimated values
m_fat = length(fatR);
mu_fat = mean(fatTrainingValues);
m_meat = length(meatR);
mu_meat = mean(meatTrainingValues);

sigma_fat = (fatTrainingValues - mu_fat).*(fatTrainingValues - mu_fat);
sigma_fat = sum(sigma_fat) / (m_fat-1);

sigma_meat = (meatTrainingValues - mu_meat).*(meatTrainingValues - mu_meat);
sigma_meat = sum(sigma_meat) / (m_meat-1);

% covariance
covar = zeros(19,19,2);
for band1 = 1:19
    for band2 = band1 : 19
        covar(band1,band2,1) = sum((fatTrainingValues(:,band1) - mu_fat(band1)) .* (fatTrainingValues(:,band2) - mu_fat(band2))) / (m_fat-1);
        covar(band1,band2,2) = sum((meatTrainingValues(:,band1) - mu_meat(band1)) .* (meatTrainingValues(:,band2) - mu_meat(band2))) / (m_meat-1);
        covar(band2,band1,:) = covar(band1,band2,:);
    end
end

Pooled = covar(:,:,1)*(m_fat-1);
Pooled = Pooled + covar(:,:,2)*(m_meat-1);
Pooled = Pooled / (m_fat + m_meat -2);

% mesure the error-rates
for test_set = 1:5
    
    [multiIm2, annotationIm2] = loadMulti(datasetnames{1,test_set} , datasetnames{2,test_set}, dirIn);
    [fat_values, ~, ~] = getPix(multiIm2, annotationIm2(:,:,2));
    [meat_values, ~, ~] = getPix(multiIm2, annotationIm2(:,:,3));
    
    errors_fat = 0;
    n_fat = length(fat_values);
    errors_meat = 0;
    n_meat = length(meat_values);
    
    for i = 1:n_fat
        Sfat = fat_values(i,:)/Pooled * transpose(mu_fat) - 1/2 * mu_fat/Pooled *transpose(mu_fat);
        Smeat = fat_values(i,:)/Pooled * transpose(mu_meat) - 1/2 * mu_meat/Pooled *transpose(mu_meat);
        errors_fat = errors_fat + (Sfat < Smeat);
    end
    
    for i = 1:n_meat
        Sfat = meat_values(i,:)/Pooled * transpose(mu_fat) - 1/2 * mu_fat/Pooled *transpose(mu_fat);
        Smeat = meat_values(i,:)/Pooled * transpose(mu_meat) - 1/2 * mu_meat/Pooled *transpose(mu_meat);
        errors_meat = errors_meat +  (Sfat > Smeat);
    
    end
    if test_set == training_set
        fprintf("\t----\t");
    else
        fprintf("\t%.1f %%\t",100*(errors_meat + errors_fat)/(n_fat + n_meat));
        average_error = average_error + (errors_meat + errors_fat)/(n_fat + n_meat);
    end
end
fprintf("\t%.2f %%\n",25*average_error);
end
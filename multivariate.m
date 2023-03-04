clear all

dirIn = 'data/';

[multiIm, annotationIm] = loadMulti('multispectral_day01.mat' , 'annotation_day01.png', dirIn);

[fatTrainingValues, fatR, fatC] = getPix(multiIm, annotationIm(:,:,2));
[meatTrainingValues, meatR, meatC] = getPix(multiIm, annotationIm(:,:,3));


%% estimated values
m_fat = length(fatR);
mu_fat = mean(fatTrainingValues);
m_meat = length(meatR);
mu_meat = mean(meatTrainingValues);

sigma_fat = (fatTrainingValues - mu_fat).*(fatTrainingValues - mu_fat);
sigma_fat = sum(sigma_fat) / (m_fat-1);

sigma_meat = (meatTrainingValues - mu_meat).*(meatTrainingValues - mu_meat);
sigma_meat = sum(sigma_meat) / (m_meat-1);


%% covariance


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

%% discriminant classification (day 01)
[salami_values, salami_rows, salami_columns] = getPix(multiIm, annotationIm(:,:,1));

class = zeros(1,length(salami_values));

for i = 1:length(salami_values(:,1))

    Sfat = salami_values(i,:)/Pooled * transpose(mu_fat) - 1/2 * mu_fat/Pooled *transpose(mu_fat);
    Smeat = salami_values(i,:)/Pooled * transpose(mu_meat) - 1/2 * mu_meat/Pooled *transpose(mu_meat);

    class(i) = Sfat > Smeat;

end

%% plot


binary_fat = zeros(514);
for i = 1:length(salami_values)
    binary_fat(salami_rows(i),salami_columns(i)) = class(i);
end
binary_meat = zeros(514);
for i = 1:length(salami_values)
    binary_meat(salami_rows(i),salami_columns(i)) = 1 - class(i);
end


color_img = imread(dirIn + "color_day01.png");
color_img2 = imread(dirIn + "color_day01.png");
color_img3 = imread(dirIn + "color_day01.png");

figure
subplot(1,3,1)
imshow(color_img)

color_img2(:,:,1) = color_img2(:,:,1).*uint8(binary_fat + annotationIm(:,:,2));
color_img2(:,:,2) = color_img2(:,:,2).*uint8(binary_fat + annotationIm(:,:,2));
color_img2(:,:,3) = color_img2(:,:,3).*uint8(binary_fat + annotationIm(:,:,2));
subplot(1,3,2)
imshow(color_img2)


color_img3(:,:,1) = color_img3(:,:,1).*uint8(binary_meat + annotationIm(:,:,3));
color_img3(:,:,2) = color_img3(:,:,2).*uint8(binary_meat + annotationIm(:,:,3));
color_img3(:,:,3) = color_img3(:,:,3).*uint8(binary_meat + annotationIm(:,:,3));
subplot(1,3,3)
imshow(color_img3)


%% classify other datasets

datasetnames = {'multispectral_day01.mat','multispectral_day06.mat','multispectral_day13.mat','multispectral_day20.mat','multispectral_day28.mat';
    'annotation_day01.png','annotation_day06.png','annotation_day13.png','annotation_day20.png','annotation_day28.png'};
imagenames =  {"color_day01.png","color_day06.png","color_day13.png","color_day20.png","color_day28.png"};

test_set = 3;

[multiIm2, annotationIm2] = loadMulti(datasetnames{1,test_set} , datasetnames{2,test_set}, dirIn);
[salami_values, salami_rows, salami_columns] = getPix(multiIm2, annotationIm2(:,:,1));

class = zeros(1,length(salami_values));

for i = 1:length(salami_values(:,1))

    Sfat = salami_values(i,:)/Pooled * transpose(mu_fat) - 1/2 * mu_fat/Pooled *transpose(mu_fat);
    Smeat = salami_values(i,:)/Pooled * transpose(mu_meat) - 1/2 * mu_meat/Pooled *transpose(mu_meat);

    class(i) = Sfat > Smeat;

end

%% plot 

binary_fat = zeros(514);
for i = 1:length(salami_values)
    binary_fat(salami_rows(i),salami_columns(i)) = class(i);
end
binary_meat = zeros(514);
for i = 1:length(salami_values)
    binary_meat(salami_rows(i),salami_columns(i)) = 1 - class(i);
end


color_img = imread(dirIn + imagenames{test_set});
color_img2 = color_img;
color_img3 = color_img;

figure
subplot(1,3,1)
imshow(color_img)
title(imagenames(test_set))

color_img2(:,:,1) = color_img2(:,:,1).*uint8(binary_fat + annotationIm2(:,:,2));
color_img2(:,:,2) = color_img2(:,:,2).*uint8(binary_fat + annotationIm2(:,:,2));
color_img2(:,:,3) = color_img2(:,:,3).*uint8(binary_fat + annotationIm2(:,:,2));
subplot(1,3,2)
imshow(color_img2)


color_img3(:,:,1) = color_img3(:,:,1).*uint8(binary_meat + annotationIm2(:,:,3));
color_img3(:,:,2) = color_img3(:,:,2).*uint8(binary_meat + annotationIm2(:,:,3));
color_img3(:,:,3) = color_img3(:,:,3).*uint8(binary_meat + annotationIm2(:,:,3));
subplot(1,3,3)
imshow(color_img3)

%% mesure the error rate on all datasets 
fprintf('\n\n--- Multivariate LDA classifier --- \n\n')
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

fprintf(strcat('\nTest on dataset\t', datasetnames{1,test_set},'\n'));
fprintf('Errors on fat : %d out of %d  (%.1f%%)\n',errors_fat, n_fat, 100*errors_fat/n_fat);
fprintf('Errors on meat : %d out of %d  (%.1f%%)\n',errors_meat,n_meat, 100*errors_meat/n_meat);
fprintf('Total number of errors : %d out of %d  (%.1f%%)\n',errors_meat+errors_fat,(n_fat + n_meat), 100*(errors_meat + errors_fat)/(n_fat + n_meat));

%fprintf('Average error percentage : %.2f%%\n' ,50*(errors_fat/n_fat + errors_meat/n_meat));
end
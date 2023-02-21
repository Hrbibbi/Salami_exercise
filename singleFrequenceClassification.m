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

%% Find threshold for one band at a time
t = zeros(1,19);
for band = 1:19
    
    i = floor(mu_meat(band));

    p_fat = exp(- (i - mu_fat(band))^2 / (2*sigma_fat(band)^2)) / (sigma_fat(band)*sqrt(2*pi));
    p_meat = exp(- (i - mu_meat(band))^2 / (2*sigma_meat(band)^2)) / (sigma_meat(band)*sqrt(2*pi));

    while p_meat > p_fat

        i = i + 1;
        p_fat = exp(- (i - mu_fat(band))^2 / (2*sigma_fat(band)^2)) / (sigma_fat(band)*sqrt(2*pi));
        p_meat = exp(- (i - mu_meat(band))^2 / (2*sigma_meat(band)^2)) / (sigma_meat(band)*sqrt(2*pi));
    end
    t(band) = i;
end

%% Error rate

false_negatives = sum(fatTrainingValues < t) / m_fat;

false_positives = sum(meatTrainingValues > t) / m_meat;

best_band = find(false_negatives + false_positives == min(false_negatives + false_positives),1);

%% Classify the hole salami

[salami_values, salami_rows, salami_columns] = getPix(multiIm, annotationIm(:,:,1));

% apply the threshold on the best bandwidth
salami_values = salami_values(:,best_band);
mask = salami_values > t(best_band);

binary_fat = zeros(514);
for i = 1:length(salami_values)
    binary_fat(salami_rows(i),salami_columns(i)) = mask(i);
end

color_img = imread(dirIn + "color_day01.png");

figure
subplot(1,3,1)
imshow(color_img)
subplot(1,3,2)
imshow(binary_fat)



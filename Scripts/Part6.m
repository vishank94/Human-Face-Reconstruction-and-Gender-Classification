%clc;
%close all;
%clear all;

root1 = 'C:\Users\VishankBhatia\Desktop\231A\Project1\Part6';
cd(root1);

dir_1 = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\female_face\*.bmp';
dir_2 = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\male_face\*.bmp';

female_faces = dir(dir_1);
male_faces = dir(dir_2);
appearance_data = zeros(173, 65536);
app_cnt = 1;

for i=1:85 %85 females
    image_name = strcat(female_faces(i).folder,'\',female_faces(i).name);
    image = double(imread(image_name));
    appearance_data(app_cnt,:) = reshape(image, [1,256*256]);
    app_cnt = app_cnt +1;
end

for i=1:88 %88 males
    image_name = strcat(male_faces(i).folder,'\',male_faces(i).name);
    image = double(imread(image_name));
    appearance_data(app_cnt,:) = reshape(image, [1,256*256]);
    app_cnt = app_cnt+1;
end


%Computing Fisher Face for Key Point

dir_1 = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\female_landmark_87\*.txt';
dir_2 = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\male_landmark_87\*.txt';
female_landmarks = dir(dir_1);
male_landmarks = dir(dir_2);
landmarks_data = zeros(173, 174);
geom_cnt = 1;

for i=1:85 %75 females
    landmarks_name = strcat(female_landmarks(i).folder,'\',female_landmarks(i).name);
    data = dlmread(landmarks_name);
    data = reshape(data, [1, 174]);
    landmarks_data(geom_cnt,:) = data;
    geom_cnt=geom_cnt+1;
end

for i=1:88 %78 males
    landmarks_name = strcat(male_landmarks(i).folder,'\',male_landmarks(i).name);
    data = dlmread(landmarks_name);
    data = reshape(data, [1, 174]);
    landmarks_data(geom_cnt,:) = data;
    geom_cnt = geom_cnt+1;
end

save('appearance_data.mat', 'appearance_data', '-double');
save('landmarks_data.mat', 'landmarks_data', '-double');

load('appearance_data.mat');
load('landmarks_data.mat');

global_male_point = [];
global_female_point = [];
global_male_landmark_point = [];
global_female_landmark_point = [];

for pick_ten=1:10:80
    test_indices =[pick_ten:pick_ten+9, pick_ten+85:pick_ten+94]; % goes into test
    train_indices = setdiff(1:173, test_indices); % goes into train
    
    train_data_appearance = appearance_data(train_indices,:);
    test_data_appearance = appearance_data(test_indices,:);
    train_data_landmarks = landmarks_data(train_indices, :);
    test_data_landmarks = landmarks_data(test_indices, :);
    
    average_landmarks = mean(train_data_landmarks);
    
    for i=1:153
        image = reshape(train_data_appearance(i,:),[256, 256]);
        org_landmarks = reshape(train_data_landmarks(i,:),[87, 2]);
        desired_landmarks = reshape(average_landmarks, [87, 2]);
        warped_image = warpImage_kent(image, org_landmarks, desired_landmarks);
        train_data_appearance(i,:) = reshape(warped_image, [1, 256*256]);
    end
    for i=1:20
        image = reshape(test_data_appearance(i,:),[256, 256]);
        org_landmarks = reshape(test_data_landmarks(i,:),[87, 2]);
        desired_landmarks = reshape(average_landmarks, [87, 2]);
        warped_image = warpImage_kent(image, org_landmarks, desired_landmarks);
        test_data_appearance(i,:) = reshape(warped_image, [1, 256*256]);
    end
    
    %Computing for Face
    average_face_vec = mean(train_data_appearance);  
    train_appearance_mean_subtract =  transpose(train_data_appearance - average_face_vec);
    %computing covariance matrix
    [V, D] = eig(transpose(train_appearance_mean_subtract)* train_appearance_mean_subtract);
    [D, ind] = sort(diag(D),'descend');
    V = V(:, ind);
    U = train_appearance_mean_subtract*V;
    for col=1:size(U,2)
        U(:,col) = U(:,col)/norm(U(:,col));
    end
    first_20_eigenvectors = U(:,1:20);
    train_projected_data = transpose(first_20_eigenvectors) * train_appearance_mean_subtract;
    female_projected_data = transpose(train_projected_data(:,1:75));
    male_projected_data = transpose(train_projected_data(:,76:153));

    mean_female = mean(female_projected_data);
    mean_male = mean(male_projected_data);

    S_1 = length(female_projected_data)*cov(female_projected_data);
    S_2 = length(male_projected_data)*cov(male_projected_data);
    S_w = S_1+S_2;
    optimal_line = inv(S_w)*(transpose(mean_male) - transpose(mean_female));

    test_mean_subtract =  transpose(test_data_appearance - average_face_vec);
    test_projected_data = transpose(first_20_eigenvectors) * test_mean_subtract;

    test_projected_data = transpose(test_projected_data);

    female_point = zeros(1,10);
    male_point = zeros(1,10);

    k=1;
    for test_image=1:10
        f = test_projected_data(test_image,:);
        sign_f = transpose(optimal_line)*transpose(f);
        female_point(1,k) = sign_f;
        k = k+1;
    end
    k=1;
    for test_image=11:20
        m = test_projected_data(test_image,:);
        sign_m = transpose(optimal_line)*transpose(m);
        male_point(1,k) = sign_m;
        k = k+1;
    end
    
    % Computing for Landmarks
    train_landmark_mean_subtract =  transpose(train_data_landmarks - average_landmarks);

    %computing covariance matrix
    [V, D] = eig(transpose(train_landmark_mean_subtract)*train_landmark_mean_subtract);
    [D, ind] = sort(diag(D),'descend');
    V = V(:, ind);
    U = train_landmark_mean_subtract*V;
    for col=1:size(U,2)
        U(:,col) = U(:,col)/norm(U(:,col));
    end
    first_20_eigenvectors = U(:,1:20);

    train_projected_data = transpose(first_20_eigenvectors) * train_landmark_mean_subtract;
    female_projected_data = transpose(train_projected_data(:,1:75));
    male_projected_data = transpose(train_projected_data(:,76:153));

    mean_female = mean(female_projected_data);
    mean_male = mean(male_projected_data);

    S_1 = length(female_projected_data)*cov(female_projected_data);
    S_2 = length(male_projected_data)*cov(male_projected_data);
    S_w = S_1+S_2;
    optimal_line = inv(S_w)*(transpose(mean_male) - transpose(mean_female));

    test_mean_subtract =  transpose(test_data_landmarks - average_landmarks);
    test_projected_data = transpose(first_20_eigenvectors) * test_mean_subtract;

    test_projected_data = transpose(test_projected_data);

    female_landmark_point = zeros(1,10);
    male_landmark_point = zeros(1,10);

    k=1;
    for test_image=1:10
        f = test_projected_data(test_image,:);
        sign_f = transpose(optimal_line)*transpose(f);
        female_landmark_point(1,k) = sign_f;
        k = k+1;
    end
    k=1;
    for test_image=11:20
        m = test_projected_data(test_image,:);
        sign_m = transpose(optimal_line)*transpose(m);
        male_landmark_point(1,k) = sign_m;
        k = k+1;
    end

    global_male_point = horzcat(global_male_point, male_point);
    global_female_point = horzcat(global_female_point, female_point);
    global_male_landmark_point = horzcat(global_male_landmark_point, male_landmark_point);
    global_female_landmark_point = horzcat(global_female_landmark_point, female_landmark_point);
    break;
end
set(0,'defaultfigurecolor',[1 1 1]);
L=figure('Position', [400, 400, 1500, 5000]);
zoom on;
plot(global_male_point, global_male_landmark_point,'b+', 'LineWidth',1.5);
hold on;
plot(global_female_point, global_female_landmark_point,'r+', 'LineWidth',1.5);
%plot([-5 ,5], [0 ,0], 'yellow');
plot([-0.1, 0.08], [0.018, -0.005], 'g-');
title('Fisher Discrimination of faces using 20 eigenvectors of landmarks and appearance', 'interpreter','latex','FontSize',15);
lg = legend('Male Face projection', 'Female Face projection');
set(lg, 'interpreter','latex','FontSize',15, 'Location', 'northwest');
set(gca,'LineWidth',1);
grid on;
saveas(L,sprintf('FLD_landmarks_appearance'),'png');
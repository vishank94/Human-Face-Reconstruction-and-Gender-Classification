%clc;
%close all;
%clear all;

root1 = 'C:\Users\VishankBhatia\Desktop\231A\Project1\Part5';
cd(root1);

dir_1 = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\female_face\*.bmp';
dir_2 = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\male_face\*.bmp';

female_faces = dir(dir_1);
male_faces = dir(dir_2);
train_data = zeros(153, 65536);
test_data = zeros(20, 65536);
train_cnt = 1;
test_cnt = 1;

for i=1:75 %75 females
    image_name = strcat(female_faces(i).folder,'\',female_faces(i).name);
    image = double(imread(image_name));
    train_data(train_cnt,:) = reshape(image, [1,256*256]);
    train_cnt=train_cnt+1;
end

for i=76:85 %10 females
    image_name = strcat(female_faces(i).folder,'\',female_faces(i).name);
    image = double(imread(image_name));
    test_data(test_cnt,:) = reshape(image, [1,256*256]);
    test_cnt=test_cnt+1;
end

figure(1);
imshow(mat2gray(reshape(mean(train_data(1:75,:)),[256,256])));
saveas(gcf,'meanF','png');

for i=1:78 %78 males
    image_name = strcat(male_faces(i).folder,'\',male_faces(i).name);
    image = double(imread(image_name));
    train_data(train_cnt,:) = reshape(image, [1,256*256]);
    train_cnt = train_cnt+1;
end

for i=79:88 %10 males
    image_name = strcat(male_faces(i).folder,'\',male_faces(i).name);
    image = double(imread(image_name));
    test_data(test_cnt,:) = reshape(image, [1,256*256]);
    test_cnt=test_cnt+1;
end

figure(2);
imshow(mat2gray(reshape(mean(train_data(76:153,:)),[256,256])));
saveas(gcf,'meanM','png');

save('test.mat', 'test_data', '-double');
save('train.mat', 'train_data', '-double');

load('test.mat');
load('train.mat');

average_face_vec = mean(train_data);
train_mean_subtract =  transpose(train_data - average_face_vec);

average_face_vec = mean(train_data);
train_mean_subtract =  transpose(train_data - average_face_vec);

%computing covariance matrix
[V, D] = eig(transpose(train_mean_subtract)* train_mean_subtract);
[D, ind] = sort(diag(D),'descend');
V = V(:, ind);
U = train_mean_subtract*V;
for col=1:size(U,2)
    U(:,col) = U(:,col)/norm(U(:,col));
end

first_10_eigenvectors = U(:,1:20);
train_projected_data = transpose(first_10_eigenvectors) * train_mean_subtract;
female_projected_data = transpose(train_projected_data(:,1:75));
male_projected_data = transpose(train_projected_data(:,76:153));

mean_female = mean(female_projected_data);
mean_male = mean(male_projected_data);

S_1 = length(female_projected_data)*cov(female_projected_data);
S_2 = length(male_projected_data)*cov(male_projected_data);
S_w = S_1+S_2;
optimal_line = inv(S_w)*(transpose(mean_male) - transpose(mean_female));

test_mean_subtract =  transpose(test_data - average_face_vec);
test_projected_data = transpose(first_10_eigenvectors) * test_mean_subtract;

test_projected_data = transpose(test_projected_data);

correct_f = 0;
correct_m = 0;

female_point = zeros(1,10);
male_point = zeros(1,10);

k=1;
for test_image=1:10
    f = test_projected_data(test_image,:);
    sign_f = transpose(optimal_line)*transpose(f);
    female_point(1,k) = sign_f;
    if sign_f<0
        correct_f = correct_f+1;
    end
    k = k+1;
end
k=1;
for test_image=11:20
    m = test_projected_data(test_image,:);
    sign_m = transpose(optimal_line)*transpose(m);
    male_point(1,k) = sign_m;
    if sign_m>0
        correct_m = correct_m+1;
    end
    k = k+1;
end
acc = ((correct_m + correct_f)/20.0) *100; 
disp(strcat('Accuracy of Testing Data ',num2str(acc),'%'));
figure;
plot(male_point,'b+', 'LineWidth',1.5);
hold on;
plot(female_point,'r+', 'LineWidth',1.5);
plot(ones(1, 12) * (0), 'yellow');
title('Fisher Discrimination using 20 eigenvectors', 'interpreter','latex','FontSize',15);
lg = legend('Male Testing','Female Testing');
set(lg, 'interpreter','latex','FontSize',15);
%set(lg, 'interpreter','latex','FontSize',15, 'Location', 'northeastoutside');
set(gca,'LineWidth',1);
grid on;
saveas(gcf,sprintf('Total_Reconstruction_Error'),'png');
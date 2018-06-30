%clc;
%close all;
%clear all;

cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2');
load('average_landmarks.mat'); %FROM PART 2
root1 = ('C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\face\');
cd(root1);
dir_data = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\face\';
dir_landmarks = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\landmark_87\';

train_aligned_images = zeros(150, 256*256);
test_aligned_images = zeros(27, 256*256);
 
for i =1:150
    val = sprintf('face%03d.bmp', i);
    image_name = strcat(dir_data,val);
    data = imread(image_name);
    
    val = sprintf('face%03d_87pt.dat', i);
    path = strcat(dir_landmarks,val);
    point_data = importdata(path);
    point_data = transpose(point_data(2:length(point_data)));
    
    desired_points = zeros(87,2);
    org_points = zeros(87, 2);
    cnt = 1;
    for j=1:2:174
        if ~ mod(j,2)==0
            org_points(cnt,1) = point_data(j);
            org_points(cnt,2) = point_data(j+1);
            desired_points(cnt,1) = average_landmarks(j);
            desired_points(cnt,2) = average_landmarks(j+1);
        end
        cnt = cnt+1;
    end
    warped_image = warpImage_kent(data, org_points, desired_points);
    train_aligned_images(i,:) = reshape(warped_image, [1, 256*256]);
end
cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3');
save('train_aligned_images.mat', 'train_aligned_images', '-double');

row = 1;
for i =151:177
    val = sprintf('face%03d.bmp', i);
    image_name = strcat(dir_data,val);
    data = imread(image_name);
    
    val = sprintf('face%03d_87pt.dat', i);
    path = strcat(dir_landmarks,val);
    point_data = importdata(path);
    point_data = transpose(point_data(2:length(point_data)));
    
    desired_points = zeros(87,2);
    org_points = zeros(87, 2);
    cnt = 1;
    for j=1:2:174
        if ~ mod(j,2)==0
            org_points(cnt,1)=point_data(j);
            org_points(cnt,2)=point_data(j+1);
            desired_points(cnt,1) = average_landmarks(j);
            desired_points(cnt,2) = average_landmarks(j+1);
        end
        cnt = cnt+1;
    end
    warped_image = warpImage_kent(data, org_points, desired_points);
    test_aligned_images(row,:) = reshape(warped_image, [1, 256*256]);
    
     cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3\original_points');
     save(strcat('testface_',num2str(row),'_original_points.mat'), 'org_points', '-double');
     cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3\warped_points');
     save(strcat('testface_',num2str(row),'_desired.mat'), 'desired_points', '-double');
     row = row+1;
end
cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3');
save('test_aligned_images.mat', 'test_aligned_images', '-double');

load('train_aligned_images');
load('test_aligned_images');

average_face_vec = mean(train_aligned_images);
train_mean_subtract =  transpose(train_aligned_images - average_face_vec);

%%% computing covariance matrix
[V, D] = eig(transpose(train_mean_subtract)* train_mean_subtract);
[D, ind] = sort(diag(D),'descend');
V = V(:, ind);
U = train_mean_subtract*V;
for col=1:size(U,2)
    U(:,col) = U(:,col)/norm(U(:,col));
end
%%% First 10 eigenfaces
first_10_eigenvectors = U(:,1:10);
save('first_10_eigenvectors_appearance','first_10_eigenvectors','-double');
save('eigenvalues_appearance','D','-double');

%% Reconstructed Image
for image_no=1:27
    b = (test_aligned_images(image_no,:)-average_face_vec)*(first_10_eigenvectors); % projection of test faces on eigenvectors
    reconstructed_image = average_face_vec + (b*transpose(first_10_eigenvectors));
    reconstructed_image = reshape(reconstructed_image, [256, 256]);
    
    cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2\reconstructed_points');
    load(strcat('testface_',num2str(image_no),'_reconstructed_points.mat'));
    cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3\warped_points');
    load(strcat('testface_',num2str(image_no),'_desired.mat'));
    
    new_image = warpImage_kent(reconstructed_image, desired_points, recon_points);
    cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3\reconstructed_images\');
    imwrite(mat2gray(reshape(new_image, [256, 256])), strcat('reconstructed_',num2str(image_no),'.jpeg'));

end

%% Reconstruction Error
cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part1');
load('test.mat');
X = 1:150;
Y = zeros(1,150);
for K=1:150
    first_K_eigenvectors = U(:,1:K);
    total = 0;
    disp(K);
    for image_no=1:27
        b = (test_aligned_images(image_no,:)-average_face_vec)*(first_K_eigenvectors); % projection of test faces on eigenvectors
        reconstructed_image = average_face_vec + (b*transpose(first_K_eigenvectors)); %%CHECK
        reconstructed_image = reshape(reconstructed_image, [256, 256]);
        
        cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2\reconstructed_points');
        load(strcat('testface_',num2str(image_no),'_reconstructed_points.mat'));
        cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3\warped_points');
        load(strcat('testface_',num2str(image_no),'_desired.mat'));

        new_image = double(warpImage_kent(reconstructed_image, desired_points, recon_points));
        new_image = reshape(new_image,[1,256*256]);

        for i=1:size(new_image,2)
            total = total + (new_image(1, i) - test(image_no, i))^2;
        end
    end
    error = total/(256*256*27);
    Y(K)=error;
    disp(error);
end

cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3');
plot(X,Y, 'LineWidth',1.5);
xlabel('Number of eigenfaces', 'interpreter','latex','FontSize',15);
ylabel('Reconstruction error for test image', 'interpreter','latex','FontSize',15);
title('Reconstruction Error over K eigen-warpings', 'interpreter','latex','FontSize',15);
%legend('');
grid on;
set(gca,'LineWidth',1);
saveas(gcf,sprintf('Reconstruction_Error_over_K_eigen-warpings_TEST'),'png');
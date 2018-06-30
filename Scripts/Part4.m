%clc;
%close all;
%clear all;

cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3');
load('train_aligned_images'); %FROM PART3

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
% eigenvec_A = U(:,1:10);
% eigenval_A = D(1:10,1);

cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2');
eigen_landmarks = load('first_10_eigenvectors_landmarks');
value_landmarks = load('eigenvalues_landmarks.mat');
cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part3');
eigen_appearance = load('first_10_eigenvectors_appearance');
value_appearance = load('eigenvalues_appearance.mat');

%load('average_face_vec.mat');
cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2');
load('average_landmarks.mat');

eigenvec_L = eigen_landmarks.first_10_eigenvectors;
eigenvec_A = eigen_appearance.first_10_eigenvectors;
 
eigenval_A = value_appearance.D;
eigenval_L = value_landmarks.D;
counter = 0;

set(0,'defaultfigurecolor',[1 1 1]);
L=figure('Position', [400, 400, 1500, 5000]);
zoom on;
for image_no=1:20

app = zeros(1,10);
geom = zeros(1,10);

for i=1:10
    app(i) = normrnd(0,1.0)*sqrt(eigenval_A(i)/150.0); %normrnd(0.0, 1.0) * sqrt(D_wf(j) / 150.0);
    geom(i) = normrnd(0,1.0)*sqrt(eigenval_L(i)/150.0);
end

generated_appearance_image = average_face_vec + (app*transpose(eigenvec_A));
image = uint8(reshape(generated_appearance_image,[256,256]));
generated_landmarks = average_landmarks + (geom*transpose(eigenvec_L));

desired_points = zeros(87,2);
org_points = zeros(87, 2);
cnt = 1;
for j=1:2:174
    if ~ mod(j,2)==0
        org_points(cnt,1)=average_landmarks(j);
        org_points(cnt,2)=average_landmarks(j+1);
        desired_points(cnt,1) = generated_landmarks(j);
        desired_points(cnt,2) = generated_landmarks(j+1);
    end
    cnt = cnt+1;
end
warped_image = warpImage_kent(double(image), org_points, desired_points);
warped_image = reshape(warped_image, [256, 256]);


subplot(5,4,image_no);
imshow(warped_image);
title(sprintf('Random Face %d',image_no));
axis tight;
axis off;
daspect([1 1 1]);
counter = counter +1;
end
cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part4\');
saveas(L,sprintf('RandomSublpot'),'png');
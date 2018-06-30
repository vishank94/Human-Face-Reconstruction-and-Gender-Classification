%clc;
%close all;
%clear all;

%STEP1: RENAMED (MALE) face000_87pt.dat to face103_87pt.dat since file was missing

dir = 'C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\landmark_87\';
train_landmarks = zeros(150, 174);
test_landmarks = zeros(27, 174);

% cnt = 1;
% test = zeros(27, 256*256);

for i = 1:150 
     name = sprintf('face%03d_87pt.dat', i);
     path = strcat(dir,name);
     data = importdata(path);
     data = transpose(data(2:length(data)));
     train_landmarks(i,:) = data;
     %adding landmarks for each image in each row
end

for i =151:177
     name = sprintf('face%03d_87pt.dat', i);
     path = strcat(dir,name);
     data = importdata(path);
     data = transpose(data(2:length(data)));
     test_landmarks(i-150,:) = data;
     %adding landmarks for each image in each row
end
 
save('train_landmarks.mat', 'train_landmarks', '-double');
save('test_landmarks.mat', 'test_landmarks', '-double');
load('train_landmarks.mat');
load('test_landmarks.mat');

load('train_face_average.mat') % FROM PART 1
mean_face = mat2gray(reshape(train_face_average,[256, 256]));

average_landmarks = mean(train_landmarks);
save('average_landmarks.mat', 'average_landmarks', '-double');
load('average_landmarks.mat');
train_mean_subtract =  transpose(train_landmarks - average_landmarks);

%computing covariance matrix
[V, D] = eig(transpose(train_mean_subtract)*train_mean_subtract);
[D, ind] = sort(diag(D),'descend');
V = V(:, ind);
U = train_mean_subtract*V;
for col=1:size(U,2)
    U(:,col) = U(:,col)/norm(U(:,col));
end
first_5_eigenvectors = U(:,1:5);
first_10_eigenvectors = U(:,1:10);
% save('first_10_eigenvectors_landmarks','first_10_eigenvectors','-double');
% save('eigenvalues_landmarks','D','-double');

cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2\eigen_part2\');
for eigen_no=1:5
    t = transpose(first_5_eigenvectors(:,eigen_no))+average_landmarks;
    eigen_points = zeros(87, 2);
    cnt = 1;
    for i=1:2:174
        if ~ mod(i,2)==0
            eigen_points(cnt, 1)=t(i);
            eigen_points(cnt, 2)=t(i+1);
        end
        cnt = cnt+1;
    end
    save(strcat('eigen_no',num2str(eigen_no)),'eigen_points','-double');
end

%% Displaying eigen warpings of landmarks

e1 = load('eigen_no1');
e2 = load('eigen_no2');
e3 = load('eigen_no3');
e4 = load('eigen_no4');
e5 = load('eigen_no5');
imshow(mean_face);
hold on;
plot(e1.eigen_points(:,1), e1.eigen_points(:,2), 'r.', 'MarkerSize',25); set(gca, 'Ydir', 'reverse');
plot(e2.eigen_points(:,1), e2.eigen_points(:,2), 'bx', 'MarkerSize',15); set(gca, 'Ydir', 'reverse');
plot(e3.eigen_points(:,1), e3.eigen_points(:,2), 'go', 'MarkerSize',10); set(gca, 'Ydir', 'reverse');
plot(e4.eigen_points(:,1), e4.eigen_points(:,2), 'y*', 'MarkerSize',10); set(gca, 'Ydir', 'reverse');
plot(e5.eigen_points(:,1), e5.eigen_points(:,2), 'md', 'MarkerSize',10); set(gca, 'Ydir', 'reverse');
hold off;
% h1 = axes;
% plot(e1.eigen_points(:,1), e1.eigen_points(:,2), 'r.');
% set(h1, 'Ydir', 'reverse');
% set(h1, 'XAxisLocation', 'Top');
% figure;
% h1 = axes;
% plot(e2.eigen_points(:,1), e2.eigen_points(:,2), 'r.');
% set(h1, 'Ydir', 'reverse');
% set(h1, 'XAxisLocation', 'Top');
% figure;
% h1 = axes;
% plot(e3.eigen_points(:,1), e3.eigen_points(:,2), 'r.');
% set(h1, 'Ydir', 'reverse');
% set(h1, 'XAxisLocation', 'Top');
% figure;
% h1 = axes;
% plot(e4.eigen_points(:,1), e4.eigen_points(:,2), 'r.');
% set(h1, 'Ydir', 'reverse');
% set(h1, 'XAxisLocation', 'Top');
% figure;
% h1 = axes;
% plot(e5.eigen_points(:,1), e5.eigen_points(:,2), 'r.');
% set(h1, 'Ydir', 'reverse');
% set(h1, 'XAxisLocation', 'Top');

%% Reconstructed Landmarks for part 2

for image_no=1:27
    b = (test_landmarks(image_no,:)-average_landmarks)*(first_5_eigenvectors);
    reconstructed_points = average_landmarks + (b*transpose(first_5_eigenvectors));
    recon_points = zeros(87, 2);
    cnt = 1;
    for i=1:2:174
        if ~ mod(i,2)==0
            recon_points(cnt, 1)=reconstructed_points(i);
            recon_points(cnt, 2)=reconstructed_points(i+1);
        end
        cnt = cnt+1;
    end
    root3=('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2\reconstructed_points\');
    cd(root3);
    save(strcat('testface5_',num2str(image_no),'_reconstructed_points.mat'), 'recon_points', '-double');
    cd('C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\face\');
    imshow(strcat(pwd(),'\face',num2str(150+image_no),'.bmp'));
    hold on;
    plot(recon_points(:,1), recon_points(:,2), 'r.');
    hold off;
    cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2\reconstructed_landmarks\');
    saveas( gcf, strcat('face5_',num2str(150+image_no),'.jpeg'));
end

%% Reconstructed Landmarks for part 3

 for image_no=1:27
     b = (test_landmarks(image_no,:)-average_landmarks)*(first_10_eigenvectors);
     reconstructed_points = average_landmarks + (b*transpose(first_10_eigenvectors));
     recon_points = zeros(87, 2);
     cnt = 1;
     for i=1:2:174
         if ~ mod(i,2)==0
             recon_points(cnt, 1)=reconstructed_points(i);
             recon_points(cnt, 2)=reconstructed_points(i+1);
         end
         cnt = cnt+1;
     end
     root4=('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2\reconstructed_points\');
     cd(root4);
     save(strcat('testface10_',num2str(image_no),'_reconstructed_points.mat'), 'recon_points', '-double');
     cd('C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\face\');
     imshow(strcat(pwd(),'\face',num2str(150+image_no),'.bmp'));
     hold on;
     plot(recon_points(:,1), recon_points(:,2), 'r.');
     hold off;
     cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2\reconstructed_landmarks\');
     saveas( gcf, strcat('face10_',num2str(150+image_no),'.jpeg'));
 end

 %% Plotting reconstruction error
cd('C:\Users\VishankBhatia\Desktop\231A\Project1\Part2\');
X = 1:150;
Y = zeros(1,150);
for K=1:150
    first_K_eigenvectors = U(:,1:K);
    total = 0;
    disp(K);
    for image_no=1:27
        b = (test_landmarks(image_no,:)-average_landmarks)*(first_K_eigenvectors); % projection of test faces on eigenvectors
        reconstructed_landmarks = average_landmarks + (b*transpose(first_K_eigenvectors));
        %reconstructed_image = mat2gray(reconstructed_image);
        %imshow(reshape(reconstructed_image,[256,256]));
        for i=1:size(reconstructed_landmarks,2)
            total = total + (reconstructed_landmarks(i) - test_landmarks(image_no, i))^2;
        end
    end
    error = sqrt(total/27.0);
    Y(K)=error;
    disp(error);
end
plot(X,Y, 'LineWidth',1.5);
xlabel('Number of eigenfaces', 'interpreter','latex','FontSize',15);
ylabel('Reconstruction error for test image', 'interpreter','latex','FontSize',15);
title('Reconstruction Error over K eigen-warpings', 'interpreter','latex','FontSize',15);
%legend('');
grid on;
set(gca,'LineWidth',1);
saveas(gcf,sprintf('Reconstruction_Error_over_K_eigen-warpings_TEST'),'png');
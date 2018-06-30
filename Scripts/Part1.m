%clc;
%clear all;
%close all;

root1 = ('C:\Users\VishankBhatia\Desktop\231A\Project1\face_data\face\');
cd(root1);

%STEP1: RENAMED face000.bmp to face103.bmp since file was missing

train = zeros(150, 256*256);
test = zeros(27, 256*256);
%STEP2: LOOP FOR ITERATING OVER 150 TRAINING FACES
for i = 1:150
    imagefiles = dir('*.bmp'); 
    image_name = strcat(imagefiles(i).folder,'\',imagefiles(i).name);
    data = imread(image_name);
    train(i,:) = reshape(data, [1, 256*256]);
    %each image in one row
end
 save('train.mat', 'train', '-double');

%STEP3: LOOP FOR ITERATING OVER 27 TESTING FACES
for i = 151:177
    imagefiles = dir('*.bmp'); 
    image_name = strcat(imagefiles(i).folder,'\',imagefiles(i).name);
    data = imread(image_name);
    test((i-150),:) = reshape(data, [1, 256*256]);
    %each image in one row
end
 save('test.mat', 'test', '-double');
 
%STEP4: LOADING .mat FILES FOR TRAIN AND TEST
load('./train.mat');
load('./test.mat');

%STEP5: CALCULATING AVERAGE FACE VECTOR AND SUBTRACT AVERAGE FACE VECTOR
train_face_average = mean(train);
train_face_average_subtract = train - train_face_average;
train_face_average_subtract_col = transpose(train_face_average_subtract);
save('train_face_average.mat', 'train_face_average', '-double');


%STEP6: COMPUTING COVARIANCE MATRIX
[vec_col_train_temp, val_train] = eig(transpose(train_face_average_subtract_col) * train_face_average_subtract_col);
vec_col_train = train_face_average_subtract_col * vec_col_train_temp;

for col = 1:size(vec_col_train,2)
    vec_col_train(:,col) = vec_col_train(:,col)/norm(vec_col_train(:,col));
end

first_20_evecs = vec_col_train(:,1:20);
first_10_evecs = vec_col_train(:,1:10);
save('first_10_evecs_appearance','first_10_evecs','-double');
save('evals_appearance','val_train','-double');

%STEP7: DISPLAY FIRST 20 EIGEN-FACES
%LOOP FOR GENERATING A SUBPLOT OF EIGENFACES
root2 = ('C:\Users\VishankBhatia\Desktop\231A\Project1\Part1\');
cd(root2);
set(0,'defaultfigurecolor',[1 1 1]);
I=figure('Position', [400, 400, 1500, 5000]);
zoom on;
for i = 1:20
    %figure;
    trained = first_20_evecs(:,i);
    trained = mat2gray(reshape(trained,[256 256]));
    %imshow(trained);
    subplot(4,5,i);
    %title(sprintf('Face_%d',i));
    imshow(trained);
    title(sprintf('Face %d',i));
    axis tight;
    axis off;
    daspect([1 1 1]);
end
saveas(I,sprintf('Subplot_First20EigenFaces'),'png');

%STEP8: CALCULATING RECONSTRUCTION ERROR
Xaxis = 1:20;
Yaxis = zeros(1,150);

% %STEP9: 
% for test_image = 1:27
%         %normalising test face image
%         test_face = (test(test_image,:) - train_face_average);
%         %projecting the test face on eigenspace
%         reconstructed_image = train_face_average + (first_K_evecs * test_face * transpose(first_K_evecs));
%         reconstructed_image = mat2gray(reconstructed_image);
%         imshow(reshape(reconstructed_image,[256,256]));
%         for i = 1:size(reconstructed_image,2)
%             total = total + (reconstructed_image(i) - test(test_image, i)) ^ 2;
%         end
% end

for K = 1:150
    first_K_evecs = vec_col_train(:,1:K);
    total = 0;
    %disp(K);
    
    for image_no = 1:27
        %normalising test face image
        test_face = (test(image_no,:) - train_face_average)*first_K_evecs;
        reconstructed_image = train_face_average + (test_face*transpose(first_K_evecs));
        %reconstructed_image = mat2gray(reconstructed_image);
        %imshow(reshape(reconstructed_image,[256,256]));
        
        for i = 1:size(reconstructed_image,2)
            total = total + (reconstructed_image(i) - test(image_no, i))^2;
        end
        
    end
    error = total/(256*256*27);
    Yaxis(K) = error;
    %disp(error);
end
Yaxis = Yaxis(:,1:20);
K = figure(2);
plot(Xaxis,Yaxis, 'LineWidth',1.5);
xlabel('Number of eigenfaces', 'interpreter','latex','FontSize',15);
ylabel('Reconstruction error for test image', 'interpreter','latex','FontSize',15);
title('Total Reconstruction Error', 'interpreter','latex','FontSize',15);
%legend('');
grid on;
set(gca,'LineWidth',1);
saveas(K,sprintf('Total_Reconstruction_Error'),'png');
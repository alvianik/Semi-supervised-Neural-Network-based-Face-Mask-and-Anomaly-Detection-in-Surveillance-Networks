clc; clear all;

%jpgFiles = dir('U:\Workspace\IntruderDet\SUPRIYO\work\*.jpeg');
jpgFiles = dir('F:\thesis\matlab-deep-learning-mtcnn-face-detection-a0a8ecc\code\mtcnn\+mtcnn\toTestCannyR\*.jpg');

for k = 1:length(jpgFiles)
    k
    filename = jpgFiles(k).name;
    

    cd 'F:\thesis\matlab-deep-learning-mtcnn-face-detection-a0a8ecc\code\mtcnn\+mtcnn\toTestCannyR\'; 
    a1 = imread(filename); cd 'F:\thesis\matlab-deep-learning-mtcnn-face-detection-a0a8ecc\code\mtcnn\+mtcnn';
    %a1 = imadjust(a);
    %a1 = histeq(a);
    %a1 = adapthisteq(a);
    %a1=dwt2(a,'haar');
    [m n]=size(a1);
    a1_expand=[];
    for r=1:m
        a1_expand=[a1_expand a1(r,:)];
    end
    P1(k,:)=[a1_expand];
    size(a1_expand)
    
    a2=imresize(a1,0.1,'bil'); %10% bilinear interpolated
    [m n]=size(a2);
    a2_expand=[];
    for r=1:m
        a2_expand=[a2_expand a2(r,:)];
    end
    P2(k,:)=[a2_expand];
     
end
A = double(P1)./double(repmat(sqrt(sum(P1.*P1,2)+eps),1,size(P1,2)));
A

save('F:\thesis\matlab-deep-learning-mtcnn-face-detection-a0a8ecc\code\mtcnn\+mtcnn\toTestCannyR\Exp0.mat', 'P1', 'P2', 'A');

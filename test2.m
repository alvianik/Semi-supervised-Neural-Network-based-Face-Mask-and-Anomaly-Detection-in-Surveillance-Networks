%{.
 clc; clear all;
jpgFiles = dir('E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn\dataset2\*.jpg');

for k = 1:length(jpgFiles)
    k
    filename = jpgFiles(k).name;
    

    cd 'E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn\dataset2\'; 
    im = imread(filename);
    cd 'E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn';

    %im = imread("F:\thesis\mask.jpg");
    [bboxes, scores, landmarks] = mtcnn.detectFaces(im);
    %fprintf("Found %d faces.\n", numel(scores));
    if numel(scores)==1
        imcropped = imcrop(im, bboxes);
        %imshow(imcropped);
        cd 'E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn\cropped2';
        imwrite(imcropped, filename);
    else
        cd 'E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn\cropped2';
        imwrite(im, filename);
    end
end

%}
%{
im = imread("F:\mask-myths.jpg");
[bboxes, scores, landmarks] = mtcnn.detectFaces(im);
fprintf("Found %d faces.\n", numel(scores));
%}

    


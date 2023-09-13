%{.
 clc; clear all;
jpgFiles = dir('E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn\datasetR2\*.jpg');

for k = 1:length(jpgFiles)
    k
    filename = jpgFiles(k).name;
    

    cd 'E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn\datasetR2\'; 
    im = imread(filename);
    cd 'E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn';

    %im = imread("F:\thesis\mask.jpg");
    [bboxes, scores, landmarks] = mtcnn.detectFaces(im);
    %fprintf("Found %d faces.\n", numel(scores));
    if numel(scores)==1
        bboxes
        imcropped = imcrop(im, bboxes);
        %imshow(imcropped);
        cd 'E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn\croppedR2\';
        imwrite(imcropped, filename);
    else
        bboxes
        for k = 1 : size(bboxes, 1)
            b1 = bboxes(k,:)
            imcropped = imcrop(im, b1);
            cd 'E:\Anik Alvi\unsupervised-face-mask-detection\mtcnn-face-detection\code\mtcnn\croppedR2\';
            filename
            filename1 = filename(1:end-4);
            filename2 = strcat(filename1, string(k), ".jpg")
            imwrite(imcropped, filename2);
        end
    end
end

%}
%{
im = imread("F:\mask-myths.jpg");
[bboxes, scores, landmarks] = mtcnn.detectFaces(im);
fprintf("Found %d faces.\n", numel(scores));
%}

    


%%% Experiment 14 %%%
%%% Investigate performance of KOAD as a function of thresholds nu1 and nu2%%%

clc; clear; close all;

% Load value of optimal sigma choice %
%load Exp1.mat;
%load U:\Workspace\IntruderDet\SUPRIYO\New500Images\Exp1.mat X sigma_star;
%load %%% Experiment 14 %%%
%%% Investigate performance of KOAD as a function of thresholds nu1 and nu2%%%

clc; clear; close all;

% Load value of optimal sigma choice %
%load Exp1.mat;
%load U:\Workspace\IntruderDet\SUPRIYO\New500Images\Exp1.mat X sigma_star;
load ('F:\thesis\matlab-deep-learning-mtcnn-face-detection-a0a8ecc\code\mtcnn\+mtcnn\resizedR2\Exp0.mat', 'P1', 'P2', 'A');

X = A;
[T D] = size(X);

kernelChoice = 2;
sigma = 5.5;
%%%%%%%%%%%%%%%%%%%%
%sigma = 2*sigma_star; %Using sigma = 2*sigma_star, and NOT sigma = sigma_star; Only for KOAD; for KEAD and KPCA, sigma = sigma_star;
%%%%%%%%%%%%%%%%%%%%

% nu1 = 0.10;
% nu2 = 0.59;

el = 10;  %Parameters for resolving orange alarm
epsilon = 0.5; 
L = 20;  %Parameters for dropping obsolete elements
d = 1; 


%anomalies = [14 20:22 30:32 34 40:45 51 61:62 78 94:95 105:106 111:114 135 152 162 164];
%anomalies = [21, 32, 53, 74, 104, 116, 127, 148, 169, 185];
anomalies = [3, 4, 6, 11, 12, 13, 14, 20,22, 23, 25, 28, 34, 37, 40, 44, 46, 58, 59, 85, 90, 99, 108, 112, 119, 121, 124, 128, 129, 141, 145, 152, 156, 167, 170, 175, 176, 178, 190, 191,196, 201, 204];
index_actual=zeros(T,1); index_actual(anomalies) = 1; actual = sum(index_actual);

deltaStore_save ={};
false_KOAD = [];
det_KOAD = [];
clear det_KOAD false_KOAD;
nu1 = 0.4;
nu2 = 0.6;
%cd 'F:\thesis\New folder'
%cd 'F:\thesis\matlab-deep-learning-mtcnn-face-detection-a0a8ecc\code\mtcnn\+mtcnn'
cd 'F:\thesis\matlab-deep-learning-mtcnn-face-detection-a0a8ecc\code'
[Red1 Red2 deltaStore Error] = KOAD(X, nu1, nu2, kernelChoice, sigma, d, L, epsilon, el);
deltaStore
x = 1:1:205;
y = deltaStore;
a = ones(T,1) * nu1;
b = ones(T,1) * nu2;
figure;
stem(x,y, 'k');
hold on;
xlabel('Timestep, t');
ylabel('KOAD Detection Statistic, δt');
xlim([0 205]);
plot(x,a, 'g', 'Linestyle', '--');
plot(x,b, 'r', 'Linestyle', '--');
hold off;

deltaStore_save = {deltaStore};
index_det_KOAD = zeros(T,1); index_det_KOAD(Red1) = 1; index_det_KOAD(Red2) = 1; %Count both Red1 and Red2 as detected
det_KOAD = [length(find(index_actual==1 & index_det_KOAD==1))];
index_false_KOAD = zeros(T,1); index_false_KOAD(find(index_actual==0 & index_det_KOAD==1)) = 1;
false_KOAD = [sum(index_false_KOAD)];

detRate_KOAD = (det_KOAD./actual) * 100
FDR_KOAD = (false_KOAD./(det_KOAD+false_KOAD)) * 100
false_alarms_KOAD = (false_KOAD./(T-actual)) * 100
    %scatter(false_KOAD(n1,:)./(T-length(anomalies)), det_KOAD(n1,:)./length(anomalies));
    %title(nu1);
%     figure;
%     plot(false_KOAD(n1,:)./(T-length(anomalies)), det_KOAD(n1,:)./length(anomalies));
%     axis([0 1 0 1]);
%     title(nu1);
%for n1=1:length(nu1_set)


%save C:\Users\tarem\LabComp\Workspace\IntruderDet\SUPRIYO\New500Images\Exp14.mat;

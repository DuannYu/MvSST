% �������ֶ���ָ��������ྫ��
[ARI,NMI,ACC] = Clustering_measure(label,gt);
% ���룺label: your labels
% gt: the real labels

% ������������������ָ��
addpath 'Measures'
ARI = RandIndex(label,gt);
NMI_1 = MutualInfo(label,gt);
ACC =  Accuracy(label,gt);
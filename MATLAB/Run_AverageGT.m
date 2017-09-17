clear all
close all
%%
datapath = 'D:\Qian\Dataset\Segmentation\BSR_bsds500\BSR\BSDS500\data\groundTruth\train\';
impath = 'D:\Qian\Dataset\Segmentation\BSR_bsds500\BSR\BSDS500\data\images\train\';
% savepath = [datapath '\train\'];
savepath = 'D:\Qian\Dataset\Segmentation\BSR_bsds500\BSR\BSDS500\data\mask\train\';
mkdir(savepath)
file_list = ls([datapath '*.mat']);
%%
for i = 1:length(file_list)
    load([datapath file_list(i,:)]);
    
    [pathstr,name,ext] = fileparts(file_list(i,:)); 
    cur_im = [name, '.jpg'];
    im = imread([impath cur_im]);
    
    cur_gt = [];
    se = strel('disk',1);
    for k = 1:length(groundTruth)
        cur_gt(:,:,k) = double(imdilate(groundTruth{k}.Boundaries, se));
    end
    GT = uint8(sum(cur_gt,3) > 2);
%     Mask = true(size(GT));
    Mask = uint8(zeros(size(GT)));
    Mask(GT > 0) = 1;
    negative_id = find(GT == 0);
    rand_id = randperm(length(negative_id));
    Mask(negative_id(rand_id(1:nnz(GT)))) = 1;
%     imwrite(uint8(Mask)*255, [savepath name, '.jpg'],'jpg');
    save([savepath file_list(i,:)], 'Mask');
%     figure(1);
% subplot(2,2,1);imagesc(GT); axis equal; colormap gray
% subplot(2,2,2);imagesc(sum(cur_gt,3)); axis equal; colormap gray
% subplot(2,2,3);imagesc(im); axis equal
% pause()

end


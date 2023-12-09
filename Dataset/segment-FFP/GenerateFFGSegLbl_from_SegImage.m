clear all;
close all;
clc;

cd C:\research\POLSAR_data_processing\FullyCN\segment-FFP;

img=imread('Seg1.PPM');
figure; imshow(img);
[m n p]=size(img);
Pro = zeros(m,n);
PatchNum  = 0;
for i=1:m
    i
    for j=1:n       
        if (Pro(i,j) == 0)    
            PatchNum = PatchNum + 1;
            queue_head=1;       %队列头
            queue_tail=1;       %队列尾
            neighbour=[-1 -1;-1 0;-1 1;0 -1;0 1;1 -1;1 0;1 1];  %和当前像素坐标相加得到八个邻域坐标
            %neighbour=[-1 0;1 0;0 1;0 -1];     %四邻域用的
            q{queue_tail}=[i j];
            Pro(i,j) = PatchNum;
            queue_tail=queue_tail+1;
            [ser1 ser2]=size(neighbour);
            while queue_head~=queue_tail
                pix=q{queue_head};
                for k=1:ser1
                    pix1=pix+neighbour(k,:);
                    if pix1(1)>=1 && pix1(2)>=1 &&pix1(1)<=m && pix1(2)<=n
                        if (img(pix1(1),pix1(2),1)==img(pix(1),pix(2),1)) && (img(pix1(1),pix1(2),2)==img(pix(1),pix(2),2)) && (img(pix1(1),pix1(2),3)==img(pix(1),pix(2),3) && (Pro(pix1(1), pix1(2))==0)) 
                            Pro(pix1(1), pix1(2)) = PatchNum;
                            q{queue_tail}=[pix1(1) pix1(2)];                            
                            queue_tail=queue_tail+1;
                        end
                    end
                end
                queue_head=queue_head+1;
            end
        end
    end
end
figure(1);
imshow(mat2gray(Pro));


im = [];
Lbl = mod(Pro, 20);
for i=1:750
    for j=1:1024
        switch single(Lbl(i,j))
            case 0,
                im(i,j,:) = [1 1 1];  %%白
            case 1,
                im(i,j,:) = [0 0 255]/255;  %% 蓝 Water
            case 2,
                im(i,j,:) = [150 0 0]/255;  %% 深棕色 Barley
            case 3,
                im(i,j,:) = [90 0 226]/255;  %% 深蓝 Peas
            case 4,
                im(i,j,:) = [255 0 0]/255;   %% 红色 Stembeans
            case 5,
                im(i,j,:) = [182 1 255]/255;  %% 紫色 Beet
            case 6,
                im(i,j,:) = [0 127 76]/255;  %% 深绿 Forest
            case 7,
                im(i,j,:) = [175 135 77]/255;  %% 淡棕色 Bare soil
            case 8,
                im(i,j,:) = [0 255 0]/255;  %% 绿色 Grasses
            case 9,
                im(i,j,:) = [255 127 0]/255;  %% 橘色 RapeSeed
            case 10,
                im(i,j,:) = [0 255 255]/255;  %% 青蓝色 Lucerne
            case 11,
                im(i,j,:) = [193 191 255]/255;  %% 淡紫色 Wheat 2
            case 12,
                im(i,j,:) = [255 183 227]/255;  %% 粉色 Wheat 1
            case 13,
                im(i,j,:) = [255 220 162]/255; %% 肉色 Buildinds
            case 14,
                im(i,j,:) = [255 255 0]/255;  %% 黄色 Potatoes
            case 15,
                im(i,j,:) = [186 255 194]/255;  %% 淡绿色 Wheat 3
            case 16,
                im(i,j,:) = [0 51 0]/255;  %% 粉色 Wheat 1
            case 17,
                im(i,j,:) = [128 0 128]/255; %% 肉色 Buildinds
            case 18,
                im(i,j,:) = [128 128 128]/255;  %% 黄色 Potatoes
            case 19,
                im(i,j,:) = [0 51 102]/255;  %% 淡绿色 Wheat 3                
        end
    end
end

figure;
imshow(im);

FFGSegLbl = Pro;
save FFGSegLbl.mat FFGSegLbl;

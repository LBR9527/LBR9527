%2017200603035 刘播瑞

load('MNISTData.mat'); %加载文件


%初始化各层权值

W1=randn(9,9,20); %初始化卷积层权值，20组9*9滤波器（9*9*20）

W3=(2*rand(100,2000)-1)/20; 
%初始化隐层权重W3，2000个输入，100个输出，并降低初始权值大小，降低学习成本

W4=(2*rand(10,100)-1)/10;
%初始化输出层权重W4，100个输入，10个输出，降低初始权值大小，降低学习成本


%训练主函数CNN

for epoch = 1 %SGD训练1轮所需时间约为8分钟
[W1,W3,W4] = CNN(W1,W3,W4,X_Train,D_Train);
end


% 训练过程完成，代入测试数据集数据测试训练正确率

N = length(D_Test); %提取测试数据集的长度(数量)

d_comp = zeros(1,N); %创建1行N列的向量d_comp

for k = 1:N %测试数据数量决定测试循环次数
    
X= X_Test(:, :, k); %从测试数据集中提取一个28*28的图像X

for k2 = 1:20
V1(:,:,k2) = filter2(W1(:,:,k2),X,'valid'); %卷积层对图像X滤波得到20*20*20特征图FeatureMap
end                                         %共20组数据通过循环一一赋值给V1                      

Y1 = ReLU(V1);  %卷积层激活函数ReLU

Y2 = Pool(Y1); %池化层对输入Y1进行2*2平均池化，得到20组10*10矩阵（10*10*20）

y2 = reshape(Y2,[],1);  %将10*10*20矩阵Y2合并为2000*1列向量y2

v3 = W3*y2; y3 = ReLU(v3); %计算隐层输出y3，隐层激活函数ReLU

v = W4*y3; y = Softmax(v); %计算输出层输出y，多分类输出层激活函数Softmax

[~, i] = max(y); %找到y向量中的最大元素，i为其位置索引

d_comp(k) = i; %保存CNN的计算值(识别出的数字)

end

[~, d_true] = max(D_Test); %将单热编码变回相应数字，存入d_true (1xN维向量)

acc = sum(d_comp==d_true); %统计正确识别的总数

fprintf('Accuracy is %f\n', acc/N); %输出正确率


function [W1,W3,W4] = CNN(W1,W3,W4,X_Train,D_Train)

alpha = 0.01;

for k = 1:60000 %训练数据集60000组
    
X= X_Train(:,:,k); %提取训练集中一个输入X

for k2 = 1:20
V1(:,:,k2)=filter2(W1(:,:,k2),X,'valid'); %卷积层对图像滤波得到20*20*20特征图FeatureMap
end                                       %共20组数据通过循环一一赋值给V1 

Y1 = ReLU(V1); %卷积层激活函数ReLU

Y2 = Pool(Y1); %池化层对输入Y1进行2*2平均池化，得到20组10*10矩阵（10*10*20）

y2 = reshape(Y2,[],1); %将10*10*20矩阵Y2合并为2000*1列向量y2

v3 = W3*y2; y3 = ReLU(v3); %计算隐层输出y3，隐层激活函数ReLU

v = W4*y3; y = Softmax(v); %计算输出层输出y，多分类输出层激活函数Softmax

%BP训练更新权值

d=D_Train(:,k); %提取1个训练集正确输出d(10*1列向量)

e=d-y; %误差列向量10*1

delta=e; %输出层delta(交叉嫡+Softmax)

e3=W4'*delta; 

delta3=(v3>0).*e3; %对激活函数ReLU求导

e2=W3'*delta3;

E2=reshape(e2,size(Y2)); %还原e2至Y2大小

E1 = zeros(size(Y1));  %逆向2*2平均池化操作，继续传递误差E1
E2_4= E2/4;            %生成20*20*20矩阵E1
E1(1:2:end,1:2:end,:)= E2_4;    
E1(1:2:end,2:2:end,:)= E2_4;
E1(2:2:end,1:2:end,:)= E2_4;
E1(2:2:end,2:2:end,:)= E2_4;

delta1=(V1>0).*E1; %对激活函数ReLU求导

for k2=1:20
dW1(:,:,k2)=alpha*filter2(delta1(:,:,k2),X,'valid'); %卷积层对图像滤波得到20*20*20特征图FeatureMap
end                                                  %通过循环一一赋值给卷积层更新量dW1              

W1= W1 + dW1;  %更新各层权值

W3= W3 + alpha*delta3*y2';

W4= W4 + alpha*delta*y3';

end
end

function y =Softmax(x)
ex = exp(x); %若x是向量，则ex是同尺寸向量
y = ex./sum(ex); %y也是同尺寸向量
end

function y=ReLU(x)
y=max(0,x);
end

%池化层函数
function y=Pool(x) 
y = (x(1:2:end,1:2:end,:)+x(2:2:end,1:2:end,:)+x(1:2:end,2:2:end,:)+x(2:2:end,2:2:end,:))/4; %池化层对输入x进行2*2平均池化
end


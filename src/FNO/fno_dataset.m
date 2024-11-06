clear;
clc;
close all

% 参数设置
L = 10; N = 256;
dx = L / N;
z = [-L/2 : dx: L/2-dx]';

dt = 0.0001;
tmax = 1;
nmax = round(tmax / dt);
k = [0: N/2-1 -N/2:-1]' * 2 * pi / L;
k2 = k.^2;

% 高斯过程的设置
num_samples = 500; % 样本数量
u_all = zeros(N, num_samples); % 存储所有样本的u
covariance_function = @(x1, x2) exp(-0.5 * (x1 - x2).^2); % 定义协方差函数

% 使用高斯过程生成初始条件
for i = 1:num_samples
    % 构建协方差矩阵
    K = zeros(N, N);
    for j = 1:N
        for m = 1:N
            K(j, m) = covariance_function(z(j), z(m));
        end
    end
    % 生成高斯过程样本
    u_gp = mvnrnd(zeros(N, 1), K)'; % 从零均值的高斯分布中采样
    u_gp(z < -5) = u_gp(z > 5); % 在-5处和5处的值相等
    u_all(:, i) = u_gp; % 保存样本
end

uu_all = zeros(num_samples, N, 201);

% 迭代计算每个初始条件的演化
for sample_idx = 1:num_samples
    sample_idx
    tt = 0;
    u = u_all(:, sample_idx); % 获取当前样本的初始条件
    uu = u;
    
    for nn = 1 : nmax
        % nn
        du1 = 1i * (ifft(0.5 * k2 .* fft(u)) - u .* u .* conj(u));
        v = u + 0.5 * du1 * dt;
        du2 = 1i * (ifft(0.5 * k2 .* fft(v)) - v .* v .* conj(v));
        v = u + 0.5 * du2 * dt;
        du3 = 1i * (ifft(0.5 * k2 .* fft(v)) - v .* v .* conj(v));
        v = u + du3 * dt;
        du4 = 1i * (ifft(0.5 * k2 .* fft(v)) - v .* v .* conj(v));
        u = u + (du1 + 2 * du2 + 2 * du3 + du4) * dt / 6;
        
        if mod(nn, round(nmax/200)) == 0
            uu = [uu u];
            tt = [tt 0 + nn * dt];
        end
    end
    uu_all(sample_idx, :, :) = uu;
    % 绘制结果
    % figure(sample_idx)
    % mesh(z, tt, abs(uu'));
    % title(['Sample ' num2str(sample_idx)]);
    % xlabel('z');
    % ylabel('Time');
    % zlabel('|u|');
end
save('NLS', 'tt', 'z', 'uu_all' );

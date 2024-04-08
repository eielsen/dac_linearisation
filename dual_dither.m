if 1
    clear
    SYSTEM = 7;
else
    SYSTEM = 0;
end

N = 1e6;
Fs = 1e6;
Ts = 1/Fs;

% v = randn(1,N);

%% specifying S(omega) = G*G'; Fourier transform of R(tau)

switch SYSTEM
    case 1 % random filter alt. 1
        g = randn(1,5);
        G_tf = tf(g,1,Ts,'Variable','z^-1'); % phi realisation
        S_num = conv(g,fliplr(g));
        S_den = [0 0 0 0 1];
    case 2 % random filter alt. 2
        G = drss(7);
        G_tf = tf(G);
        S = G*G';
        S_tf = tf(S); S.Ts = Ts;
        S_num = S_tf.Numerator{:};
        S_den = S_tf.Denominator{:};
    case 3
        g = [1 -1];
        G_tf = tf(g,1,Ts,'Variable','z^-1');
        S_num = conv(g,fliplr(g));
        S_den = 1;
    case 4
        g = [1 1];
        G_tf = tf(g,1,Ts,'Variable','z^-1');
        S_num = conv(g,fliplr(g));
        S_den = 1;
    case 5
        g = ones(1,10)/10;
        G_tf = tf(g,1,Ts,'Variable','z^-1');
        S_num = conv(g,fliplr(g));
        S_den = 1;
    case 6
        Nh = 2047;
        n = (0:Nh-1)-(Nh-1)/2;
        fc = 0.5/2; % cutoff frequency = 0.5
        g_lp = 2*fc*sinc(2*fc*n).*kaiser(Nh,50).'; % windowed sinc function
        %figure; freqz(h_lp)
        
        g = -g_lp; % spectral inversion
        g((Nh+1)/2) = g((Nh+1)/2) + 1;
        %figure; freqz(g), return

        Gtf = tf(g,1,Ts,'Variable','z^-1');

        S_num = conv(g,fliplr(g));
        S_den = 1;
    case 7 %
        Nh = 63;
        g = fir1(Nh,[0.5 0.8]);
        G_tf = tf(g,1,Ts,'Variable','z^-1');

        S_num = conv(g,fliplr(g));
        S_den = 1;
        
        
        %Fc = 100e3;
        %Fn = Fs/2;
        %[b,a] = butter(4, [Fc, 2*Fc]/Fn);  % Butterworth filter
        %G = tf(b,a,1/Fs);
        %Gc = tf([1 0.9 0.25],[1 0.001 2.5]); % tf([1 0.01 0.1],[0.1 0.01 1]); % tf([0.1 0.01],[0.01 0.1]); % tf([1 0.9],[0.05 1]);
        %Gc = balreal(Gc);
        %G = c2d(Gc,Ts,'zoh');
        %S = G*G';
        
        %S_tf = tf(S);
        %S_num = S_tf.Numerator{:};
        %S_den = S_tf.Denominator{:};
        

    case 8 % Valid PSD-shaping examples (Phi_(w) >= 0 for all w, condition is met)
        PSD_Example_Data = readmatrix('Valid_PSD_Shaping_Examples.txt');
        rand_index = size(PSD_Example_Data);
        rand_index = randi(rand_index(1)/2);
        S_num = PSD_Example_Data(2*rand_index-1,:);
        S_den = PSD_Example_Data(2*rand_index,:);  
    case 9 % Invalid PSD-shaping examples (Phi_(w) >= 0 for all w, condition is violated)
        PSD_Example_Data = readmatrix('Invalid_PSD_Shaping_Examples.txt');
        rand_index = size(PSD_Example_Data);
        rand_index = randi(rand_index(1)/2);
        S_num = PSD_Example_Data(2*rand_index-1,:);
        S_den = PSD_Example_Data(2*rand_index,:); 
    otherwise

end

M = 1024; % no. of frequency samples

%% Non-linearities chain parameters

B = 7; % Number of quantization bits
V_max = 10; % Dynamic range - upper limit
V_min = -10; % Dynamic range - lower limit
FSR = V_max-V_min; % Full Scale Range
LSB = FSR/(2^B-1); % Least Sitgnificant Bit
Quant_L = V_min:LSB:V_max; % Quantized Output Levels (2^B)
T = Quant_L(1:end-1)+0.5*LSB; % Quantization Thresholds (Mid-rise, 2^B-1)
Var_v = 1; % Initial dist. variance
Var_Eq = LSB^2/12; % Quantization Error variance (Output stage - Uniform Quantizer)

pdf_initial = makedist('Normal','mu',0,'sigma',sqrt(Var_v)); % Initial distribution


Amp = 50*LSB; % Amp value for a desired Uniform PDF [LSB]
pdf_des = makedist('Uniform',-Amp,Amp); % Desired dither probablity deistribution (RPDF)
Var_y = pdf_des.var; 


%% frequency response of S(omega)

% M = 1024; % no. of frequency samples
w = linspace(-pi,pi,M); % sample whole circle
S_fr = freqz(S_num,S_den,w); % frequency response samples

% if SYSTEM==7
%     S_fr = freqresp(S,w,'rad/s');
%     S_fr = S_fr(:);
%     S_fr = circshift(S_fr,M/2);
% end

w = linspace(0,2*pi,M); % sample whole circle

%% determining the norm/variance and analytical S(omega)

% The supplied norm() function in MATLAB fails 
S_fr_2norm = sum(abs(S_fr).*mean(diff(w)))/(2*pi);
S_fr_ = (S_fr/S_fr_2norm)*Var_y; % scale response to a feasible variance at the output of the non-linear element [v+qUniform: (LSB^2/12)]

% var(y) = norm(G)^2, when y = G v, and v unity variance white noise


%% PSD S(w)

S_fr = abs(S_fr);
N_fir = 256;
[R_,R_win,R_beta] = fir_filter_ls(S_fr,N_fir);
S_fir_fr = S_num; %freqz(R_win,R_beta,w); % frequency response samples

% figure(45)
% plot(w/(2*pi), 10*log10(abs(S_fr)), w/(2*pi), 10*log10(abs(S_fir_fr)))

%% compute phi
R_win_ = 0.999*R_win/(norm(R_win,Inf)+sqrt(eps));

% phi = sin(pi/2*R_win_); % eq (16c) in Sondhi 1983 [Binary]
phi = 2*sin((pi/6)*R_win_); % eq (16a) in Sondhi 1983 [Uniform]

% plot(phi,R_win_,p,R)

Phi_fr = freqz(phi,1,w); % frequency response

switch 1
    case 1 % use FFT/IFFT to synth. H (same as Sondhi - 1983)
        phi_ = circshift(phi,128);
        % figure(50), stem(phi_)

        Phi = fft(phi_);

        % if(real(Phi)==abs(real(Phi)))
        %     Table_num = table(S_num); % Examples of well-behaved desired PSDs
        %     Table_denum = table(S_den); % Examples of well-behaved desired PSDs
        %     writetable(Table_num, 'Dither_Gen_HP_PSD_Shaping_Examples.txt','WriteMode','Append','WriteVariableNames',false); % Record randomly generated feasible PSD examples.
        %     writetable(Table_denum, 'Dither_Gen_HP_PSD_Shaping_Examples.txt','WriteMode','Append','WriteVariableNames',false);
        % else
        %     Table_num = table(S_num); % Examples of ill-behaved PSDs
        %     Table_denum = table(S_den); % Examples of ill-behaved PSDs
        %     writetable(Table_num, 'Dither_Gen_Invalid_PSD_Shaping_Examples.txt','WriteMode','Append','WriteVariableNames',false); % Record randomly generated feasible PSD examples.
        %     writetable(Table_denum, 'Dither_Gen_Invalid_PSD_Shaping_Examples.txt','WriteMode','Append','WriteVariableNames',false);
        % end

        % Th = w(real(Phi)<0) & w(S_fr_<=Var_Eq);
        
        Phi(real(Phi)<0) = 0; % (Filter Approx. Eq(18) in Sondhi - 1983) ***

        mus = sqrt(abs(Phi));

        figure(55)
        subplot(2,1,1)
        plot(real(fft(circshift(R_win_,128)))), hold on, plot(real(Phi)), plot(real(mus)), hold off
        str = '$PSD [dB/Hz]$';
        ylabel(str,'Interpreter','latex')
        legend('$Re[S]$','$Re[\Phi]$','$Re[\mu]$','interpreter','latex')
        subplot(2,1,2)
        plot(imag(fft(circshift(R_win_,128)))), hold on, plot(imag(Phi)), plot(imag(mus)), hold off
        str = '$\arg{PSD} [deg]$';
        ylabel(str,'Interpreter','latex')
        legend('$Im[S]$','$Im[\Phi]$','$Im[\mu]$','interpreter','latex')
        str = '$\tau$';
        xlabel(str,'Interpreter','latex')

        

        h = ifft(real(mus)); % real(mus) excludes real(Phi) < 0
        h = circshift(h,128);

        % figure(60), stem(h)
    case 2 % use LS on frequency response to synth. H
        
        mus = sqrt(real(Phi_fr));
        [h_alpha,h_alpha_win,h_beta] = fir_filter_ls(mus,1000);

        h = h_alpha_win;
end

Phi_fr_2norm = sum(abs(Phi_fr).*mean(diff(w)))/(2*pi); % should be 1

figure(100)
%plot(w/(2*pi), 10*log10(abs(R_fir_fr)), w/(2*pi), 10*log10(abs(phi_fir_fr)))
plot(w/(2*pi), 10*log10(abs(S_fr_)))
hold on
plot(w/(2*pi), 10*log10(Var_y*abs(Phi_fr)))
plot(w/(2*pi),10*log10(Var_y)*ones(numel(w/(2*pi)),1),'--','Color',[0.4660 0.6740 0.1880],'LineWidth',1.5)
plot(w/(2*pi),10*log10(Var_Eq)*ones(numel(w/(2*pi)),1),'-.','Color',[0.9290 0.6940 0.1250],'LineWidth',1.5)
hold off
grid on
legend('$S(\omega)$','$\Phi(\omega)$','NSD noise floor','SD noise floor','Location','best','interpreter','latex')

figure(200)
stem(R_win_)
hold on
stem(phi)
hold off
str = '$\tau$';
xlabel(str,'Interpreter','latex')
str = '$ACF$';
ylabel(str,'Interpreter','latex')
legend('$R(\tau)$','$\rho(\tau)$','interpreter','latex')

%% generate coloured input to non-lin.

v = random(pdf_initial, 1, N);

switch 1
    case 1 % use corrected filter
        x_ = filter(h,1,v);
    case 2 % use "original" filter
        x_ = filter(g,1,v);
end

v_hat = sqrt(var(v)/var(x_))*x_; % normalise (should not be neccessary)
x_ = pdf_des.icdf(pdf_initial.cdf(v_hat));

% % Plot Output Histogram & PSD
figure(5)
histogram(x_/LSB,'Normalization', 'pdf'); title('PDF-PSD Shaped Dither Ditribution');xlabel('Amplitude [LSB]');ylabel('PDF');
% 
figure(6)
WIN = kaiser(N/250,10);
plot(w(w < pi)/(2*pi), 10*log10(2*abs(S_fr_(w < pi))))
% trapz(w(w < pi)/(2*pi),2*abs(S_fr_(w < pi))) % Area under the PSD curve - Compare to Var_y (Should/ Must add up)
hold on
[Pdd,Fd] = pwelch(x_,WIN,[],[],[],'psd','onesided'); % Fy(10*log10(Pyy)>=10*log10(2*Var_Eq))
% trapz(Fy,Pyy) % Area under the PSD curve - Compare to Var_y (Should/ Must add up)
plot(Fd,10*log10(Pdd))
plot(Fd,10*log10(2*Var_y)*ones(numel(Fd),1),'--','Color',[0.4660 0.6740 0.1880],'LineWidth',1.5)
hold off
grid on
legend('Desired','Acheived','NSD power level','Location','best','interpreter','latex')
str = 'Normalized frequency';
xlabel(str,'Interpreter','latex')
str = '$PSD [dB/Hz]$';
ylabel(str,'Interpreter','latex')
str = 'Desired vs. Acheived single-sided output PSD $S(\omega)$';
title(str,'Interpreter','latex')

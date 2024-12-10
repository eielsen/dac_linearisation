function [alpha,alpha_win,beta] = fir_filter_ls(H,p)
% Synthesise FIR filter from complex values frequency response samples
% 
%  Inputs:
%  H - frequency response samples from 0 to 2*pi
%  p - desired FIR filter length
%  Outputs:
%  alpha     - FIR filter coefficients
%  alpha_win - windows FIR filter coefficients
%  beta      - denominator that can be used to null the phase when computing
%              the frequency response

% AA Eielsen - 2023

switch nargin
    case 2

    case 1
        p = 50; % filter length
    otherwise
        HM = drss(5); % random LTI system prototype/reference

        HMtf = tf(HM);
        bref = HMtf.Numerator{:};
        aref = HMtf.Denominator{:};

        M = 1e3; % no. of frequency samples
        w = linspace(0,2*pi,M); % sample whole circle
        H = freqz(bref,aref,w); % frequency response samples

        p = 50; % filter length
end

%% generate FIR filter approximation

M = length(H); % no. of frequency samples

X = zeros(M,p);

if mod(p,2) % odd
    q = (p-1)/2;
else % even
    q = p/2;
end

K = 2*pi/M;
ps = 0:p-1;
pq = K*(ps-q);

for k = 0:M-1 % iterate over frequency samples
    X(k+1,:) = exp(-1i*k*pq);
end

b = H(:);

% min(||b - X*alpha||)
alpha = real(X\b); % real valued FIR coefficients by least-squares
alpha = alpha(:); % ensure column vector
alpha = -alpha; % spectral inversion
alpha_win = hann(numel(alpha)).*alpha; % reduce ripples due to implicit rectangular windowing

beta = [zeros(1,q) 1];

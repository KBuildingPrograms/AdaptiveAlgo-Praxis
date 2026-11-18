%Matlab Based File
WT221 = parquetread("V408A 221 4 units pleth only 6.11.24.parquet");
MU222 = parquetread("V408A 222 4 units pleth only 6.11.24.parquet");


UK230 = parquetread("4 unit no implant pleth V408A 230 6.13.24.parquet");
%larger files load fine :)

%%
%Stage one, filtering

function filtered_data = preprocessing(unfiltered_data)
    shift = -((unfiltered_data) - mean(unfiltered_data,2))/std(unfiltered_data); %Normalization
    h = [1/2,1/2];
    binomial_coeff_filter = conv(conv(h,h),h);
    binomialMA = filter(binomial_coeff_filter,1,shift); %Reduction of variation (based on standard for data in lab)
    [b,a] = butter(2,6); %weak low pass filter with cutoff around the maximum for breath rate in mice
    filtered_data = filter(b,a,binomialMA(:,2));
end

%Stage two - unknown reducing
N = 3*60*2000; %length of overall segment, minutes times seconds times sampling rate
function [W1,w2,w3,w4a,w4b,b1,b2,b3,b4] = initialize_weights(L1,L2,L3,L4,batch) %initializing weights for each layer
    w1 = randn(1,L1); %Convolution prep
    W1 = zeros(batch-L1+1,batch);
    b1 = zeros(batch-L1+1,1);
    w1_alt = [w1,zeros(1,batch-L1+1)];
    w1_alt2 = [w1(1),zeros(1,batch-L1+1)];
    W1 = toeplitz(w1_alt,w1_alt2);

    w2 = randn(L2,batch-L1+1);
    b2 = zeros(L2,1);

    w3 = randn(L3,L2);
    b3 = randn(L3,1);

    w4a = randn(L4,L3);
    b4 = randn(L4,1);

    w4b = randn(L4,L3);
end

function output = forward_prop(filtered_data,backup_data,labels)
    batch = 2000; %taking out data in chunks, 1/18th based on current data length
    L1 = 1000;
    L2 = 100;
    L3 = 2; %Heart-rate and mass are being included as potential explanations for the content
    L4 = 1;

    [W1,w2,w3,w4a,w4b,b1,b2,b3,b4] = initialize_weights(L1,L2,L3,L4,batch); 
    
    

end


%Final stage sigmoid
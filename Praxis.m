%Matlab Based File
WT221 = parquetread("V408A 221 4 units pleth only 6.11.24.parquet");
MU222 = parquetread("V408A 222 4 units pleth only 6.11.24.parquet");


UK230 = parquetread("4 unit no implant pleth V408A 230 6.13.24.parquet");
%larger files load fine :)
%%
%Stage one, filtering
function filtered_data = preprocessing(unfiltered_data)
    shift = -((unfiltered_data) - mean(unfiltered_data,2));
    h = [1/2,1/2];
    binomial_coeff_filter = conv(conv(h,h),h);
    binomialMA = filter(binomial_coeff_filter,1,shift);
    [b,a] = butter(2,6);
    filtered_data = filter(b,a,binomialMA(:,2));
end
%Stage two - unknown reducing

%Final stage sigmoid
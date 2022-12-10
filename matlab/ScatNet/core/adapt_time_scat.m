function S_adapt_time = adapt_time_scat(U1, S1, Wop, adapt_options)
% Compute the adaptive time scattering
% Created by Changhong Wang

S_adapt_time = NaN*ones(adapt_options.moduIdx(2)-adapt_options.moduIdx(1)+1,...
                length(adapt_options.domIdx)); 
    
for k=1:length(adapt_options.domIdx)
    if adapt_options.domIdx(k) > adapt_options.maxDecmpIdx
        continue
    elseif adapt_options.domIdx(k) < 1
        decomposeIdx = 1;
    else
        decomposeIdx = adapt_options.domIdx(k);
    end

    hop = adapt_options.T/2^adapt_options.oversampling/2^U1.meta.resolution(decomposeIdx);
    if (k-1)*hop+adapt_options.T/2^U1.meta.resolution(decomposeIdx) > ...
            length(U1.signal{1,decomposeIdx})
        U1.signal{1,decomposeIdx} = [U1.signal{1,decomposeIdx};...
            NaN*ones((k-1)*hop+adapt_options.T/2^U1.meta.resolution(decomposeIdx)-...
            length(U1.signal{1,decomposeIdx}),1)]; % pad into sig_len with NaNs
    end
    U_temp.signal = {U1.signal{1,decomposeIdx}((k-1)*hop+1:(k-1)*hop+...
        adapt_options.T/2^U1.meta.resolution(decomposeIdx))}; % signal within one T in time
    U_temp.meta.bandwidth = U1.meta.bandwidth(decomposeIdx); % meta info
    U_temp.meta.resolution = U1.meta.resolution(decomposeIdx);
    U_temp.meta.j = U1.meta.j(decomposeIdx);
    [~, V] = Wop{2}(U_temp);  % U1 => U2
    U_temp = modulus_layer(V); % U2
    S_temp = Wop{2}(U_temp); 
    S_out = [S_temp.signal{:}].';
    
    S{1}.signal = {S1{1}.signal{1}(k)};   % low of the waveform of current frame
    S{1}.meta.bandwidth = S1{1}.meta.bandwidth;
    S{1}.meta.resolution = S1{1}.meta.resolution;
    S{2}.signal = {S1{2}.signal{decomposeIdx}(k)}; 
    S{2}.meta.j = S1{2}.meta.j(decomposeIdx);
    S{2}.meta.bandwidth = S1{2}.meta.bandwidth(decomposeIdx);
    S{2}.meta.resolution = S1{2}.meta.resolution(decomposeIdx);

    if isempty(S_temp.signal{1,1})
        continue
    else
        modu = mean([S_temp.signal{:}].',2); % get mean due to oversampling
        S_temp.signal = [num2cell(modu,length(modu))]';
        S{3} = S_temp;   
        S_out = log_scat(renorm_scat(S)); % S2, norm necessary
        S_out = [S_out{1, 3}.signal{:}].';
        S_adapt_time(:,k) = S_out(adapt_options.moduIdx(1):adapt_options.moduIdx(2));
    end
    clear S
end

end